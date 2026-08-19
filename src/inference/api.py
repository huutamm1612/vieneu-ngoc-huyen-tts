from __future__ import annotations

import logging
import re
import shutil
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

from .audio import assemble_final_wav
from .batching import validate_batches
from .config import InferenceConfig
from .generation import run_generation
from .modeling import InferenceWorker, close_workers, encode_reference, load_workers
from .preprocessing import (
    StoryPreprocessor,
    finalize_normalization_unit,
    prepare_normalization_unit,
    read_text_file,
)
from .types import InferenceResult, PreparedBatches

LOGGER = logging.getLogger(__name__)


def prepare_batches(
    *,
    text: str | None = None,
    input_path: str | Path | None = None,
    min_chars: int = 80,
    target_chars: int = 128,
    max_chars: int = 156,
    batch_size: int = 128,
    max_length_gap: int = 12,
    preprocessor: StoryPreprocessor | None = None,
) -> PreparedBatches:
    """Normalize Vietnamese text, create chunks/jobs, and length-bucket them."""
    if (text is None) == (input_path is None):
        raise ValueError("Provide exactly one of text or input_path")
    source_name = "text"
    if input_path is not None:
        source = Path(input_path).expanduser().resolve()
        text = read_text_file(source)
        source_name = source.name
    assert text is not None
    processor = preprocessor or StoryPreprocessor()
    batches, _ = processor.prepare(
        text,
        min_chars=min_chars,
        target_chars=target_chars,
        max_chars=max_chars,
        batch_size=batch_size,
        max_length_gap=max_length_gap,
        source_name=source_name,
    )
    return validate_batches(batches)


class TTSInference:
    """Portable VieNeu-TTS inference engine with lazy model loading."""

    def __init__(
        self,
        config: InferenceConfig | str | Path | None = None,
        *,
        preprocessor: StoryPreprocessor | None = None,
    ) -> None:
        if config is None:
            resolved = InferenceConfig()
        elif isinstance(config, InferenceConfig):
            resolved = InferenceConfig.from_dict(config.as_dict())
        else:
            resolved = InferenceConfig.from_yaml(config)
        resolved.validate()
        self.config = resolved
        self._preprocessor = preprocessor
        self._workers: list[InferenceWorker] = []
        self._reference_signature: tuple[str, int, int, str] | None = None

    @property
    def preprocessor(self) -> StoryPreprocessor:
        if self._preprocessor is None:
            self._preprocessor = StoryPreprocessor()
        return self._preprocessor

    @property
    def workers(self) -> list[InferenceWorker]:
        if not self._workers:
            self._seed_everything()
            self._workers = load_workers(self.config)
        return self._workers

    def _seed_everything(self) -> None:
        import torch

        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _normalized_reference_phonemes(self, reference_text: str) -> str:
        if not reference_text.strip():
            raise ValueError("reference_text cannot be empty")
        unit = prepare_normalization_unit(unicodedata.normalize("NFC", reference_text))
        unit = self.preprocessor.normalizer.normalize(unit)
        unit = finalize_normalization_unit(unit)
        if re.search(r"\d", unit) or "<" in unit or ">" in unit:
            raise ValueError("reference_text must normalize to text without digits or angle brackets")
        return self.preprocessor.phonemize([unit])[0]

    def set_reference(self, reference_audio: str | Path, reference_text: str) -> None:
        source = Path(reference_audio).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Reference audio not found: {source}")
        stat = source.stat()
        signature = (str(source), stat.st_size, stat.st_mtime_ns, reference_text.strip())
        if signature == self._reference_signature:
            return
        if self._workers and any(worker.codec_encoder_released for worker in self._workers):
            LOGGER.info("Reference changed after codec encoder release; reloading inference workers")
            self.close()
        reference_phonemes = self._normalized_reference_phonemes(reference_text)
        encode_reference(
            self.workers,
            reference_audio=source,
            reference_phonemes=reference_phonemes,
            config=self.config,
        )
        self._reference_signature = signature

    def prepare_batches(
        self,
        *,
        text: str | None = None,
        input_path: str | Path | None = None,
    ) -> PreparedBatches:
        return prepare_batches(
            text=text,
            input_path=input_path,
            min_chars=self.config.min_chars,
            target_chars=self.config.target_chars,
            max_chars=self.config.max_chars,
            batch_size=self.config.batch_size,
            max_length_gap=self.config.max_length_gap,
            preprocessor=self.preprocessor,
        )

    def infer(
        self,
        *,
        output_path: str | Path,
        reference_audio: str | Path,
        reference_text: str,
        batches: PreparedBatches | None = None,
        text: str | None = None,
        input_path: str | Path | None = None,
    ) -> InferenceResult:
        """Generate all batches and atomically write exactly one final WAV."""
        if batches is None:
            batches = self.prepare_batches(text=text, input_path=input_path)
        else:
            if text is not None or input_path is not None:
                raise ValueError("Do not pass text/input_path when batches is provided")
            batches = validate_batches(batches)
        destination = Path(output_path).expanduser().resolve()
        if destination.suffix.casefold() != ".wav":
            raise ValueError("output_path must end with .wav")
        destination.parent.mkdir(parents=True, exist_ok=True)
        self.set_reference(reference_audio, reference_text)
        segment_directory = Path(
            tempfile.mkdtemp(prefix=f".{destination.stem}-segments-", dir=destination.parent)
        )
        succeeded = False
        try:
            metadata, generation_seconds = run_generation(
                self.workers,
                batches,
                segment_directory,
                self.config,
            )
            sample_rate, duration, metadata = assemble_final_wav(
                metadata,
                destination,
                self.config,
            )
            if not self.config.keep_segments:
                for item in metadata:
                    item.pop("segment_path", None)
            succeeded = True
            return InferenceResult(
                output_path=str(destination),
                sample_rate=sample_rate,
                segments=len(metadata),
                duration_seconds=duration,
                inference_seconds=generation_seconds,
                rtf=generation_seconds / max(duration, 1e-9),
                devices=[str(worker.device) for worker in self.workers],
                metadata=metadata,
                temporary_segments_kept=self.config.keep_segments,
                temporary_segment_directory=str(segment_directory) if self.config.keep_segments else None,
            )
        finally:
            if not self.config.keep_segments:
                shutil.rmtree(segment_directory, ignore_errors=True)
            elif not succeeded:
                LOGGER.warning("Keeping failed inference segments at %s", segment_directory)

    def close(self) -> None:
        if self._workers:
            close_workers(self._workers)
        self._reference_signature = None

    def __enter__(self) -> "TTSInference":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def infer(
    *,
    model: str | None = None,
    output_path: str | Path,
    reference_audio: str | Path,
    reference_text: str,
    batches: PreparedBatches | None = None,
    text: str | None = None,
    input_path: str | Path | None = None,
    config: InferenceConfig | str | Path | None = None,
    **overrides: Any,
) -> InferenceResult:
    """One-call API that loads, runs, and releases the inference engine."""
    if config is None:
        resolved = InferenceConfig()
    elif isinstance(config, InferenceConfig):
        resolved = InferenceConfig.from_dict(config.as_dict())
    else:
        resolved = InferenceConfig.from_yaml(config)
    if model is not None:
        overrides["model"] = model
    resolved.apply_overrides(**overrides)
    with TTSInference(resolved) as engine:
        return engine.infer(
            output_path=output_path,
            reference_audio=reference_audio,
            reference_text=reference_text,
            batches=batches,
            text=text,
            input_path=input_path,
        )
