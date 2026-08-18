from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .artifacts import atomic_write_json, utc_now
from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)

TEXT_PROMPT_START = "<|TEXT_PROMPT_START|>"
TEXT_PROMPT_END = "<|TEXT_PROMPT_END|>"
SPEECH_GENERATION_START = "<|SPEECH_GENERATION_START|>"
SPEECH_GENERATION_END = "<|SPEECH_GENERATION_END|>"


@dataclass(frozen=True, slots=True)
class MetadataRow:
    filename: str
    text: str


def read_pipe_metadata(path: str | Path) -> list[MetadataRow]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Metadata file not found: {source}")
    rows: list[MetadataRow] = []
    seen: set[str] = set()
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("|", maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Invalid metadata line {line_number} in {source}: expected filename|text")
            filename, text = (part.strip() for part in parts)
            if not filename or not text:
                raise ValueError(f"Empty filename/text at line {line_number} in {source}")
            if filename in seen:
                raise ValueError(f"Duplicate filename in {source}: {filename}")
            seen.add(filename)
            rows.append(MetadataRow(filename=filename, text=text))
    if not rows:
        raise ValueError(f"Metadata file is empty: {source}")
    return rows


def audit_dataset(config: PipelineConfig) -> dict[str, Any]:
    root = config.dataset_root
    audio_root = root / config.data.audio_directory
    train_path = root / config.data.train_metadata
    eval_path = root / config.data.eval_metadata
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_root}")

    train_rows = read_pipe_metadata(train_path)
    eval_rows = read_pipe_metadata(eval_path)
    train_names = {row.filename for row in train_rows}
    eval_names = {row.filename for row in eval_rows}
    overlap = sorted(train_names & eval_names)
    if overlap:
        raise ValueError(f"Train/eval filename leakage detected ({len(overlap)} files), first: {overlap[0]}")

    missing = [
        row.filename
        for row in [*train_rows, *eval_rows]
        if not (audio_root / row.filename).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Metadata references {len(missing)} missing audio files; first: {missing[0]}")

    active_names = train_names | eval_names
    wav_names = {path.name for path in audio_root.glob("*.wav")}
    orphaned = sorted(wav_names - active_names)
    duplicate_texts = _duplicate_text_count([*train_rows, *eval_rows])
    train_texts = {_normalize_text(row.text) for row in train_rows}
    eval_texts = {_normalize_text(row.text) for row in eval_rows}
    cross_split_text_overlap = sorted(train_texts & eval_texts)

    report = {
        "audited_at": utc_now(),
        "dataset_root": str(root),
        "audio_root": str(audio_root),
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "referenced_samples": len(active_names),
        "wav_files": len(wav_names),
        "missing_audio": len(missing),
        "orphaned_wav": len(orphaned),
        "orphaned_wav_examples": orphaned[:20],
        "cross_split_filename_overlap": len(overlap),
        "cross_split_normalized_text_overlap": len(cross_split_text_overlap),
        "cross_split_normalized_text_examples": cross_split_text_overlap[:10],
        "duplicate_normalized_text_groups": duplicate_texts,
    }
    return report


def _duplicate_text_count(rows: Iterable[MetadataRow]) -> int:
    counts: dict[str, int] = {}
    for row in rows:
        normalized = _normalize_text(row.text)
        counts[normalized] = counts.get(normalized, 0) + 1
    return sum(1 for count in counts.values() if count > 1)


def _normalize_text(text: str) -> str:
    return " ".join(text.casefold().split())


def dataset_fingerprint(config: PipelineConfig) -> str:
    digest = hashlib.sha256()
    root = config.dataset_root
    for relative in (
        config.data.train_metadata,
        config.data.eval_metadata,
        "manifest.csv",
        "processing_config.json",
        "processing_stats.json",
    ):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
    fingerprint_fields = {
        "base_model": config.model.base_model,
        "model_revision": config.model.model_revision,
        "codec_model": config.model.codec_model,
        "codec_revision": config.model.codec_revision,
        "codec_input_sample_rate": config.data.codec_input_sample_rate,
        "max_sequence_length": config.model.max_sequence_length,
        "format_version": 1,
    }
    digest.update(json.dumps(fingerprint_fields, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:20]


def format_training_sequence(phones: str, codes: list[int]) -> str:
    code_tokens = "".join(f"<|speech_{code}|>" for code in codes)
    return (
        f"{TEXT_PROMPT_START}{phones}{TEXT_PROMPT_END}"
        f"{SPEECH_GENERATION_START}{code_tokens}{SPEECH_GENERATION_END}"
    )


def encoded_dataset_paths(config: PipelineConfig) -> dict[str, Path]:
    """Return the deterministic shared-Volume paths for an encoded dataset."""
    fingerprint = dataset_fingerprint(config)
    cache_root = config.output_root / "cache" / "encoded" / fingerprint
    return {
        "root": cache_root,
        "train": cache_root / "train.jsonl",
        "eval": cache_root / "eval.jsonl",
        "complete": cache_root / "COMPLETE.json",
    }


def require_encoded_dataset(config: PipelineConfig) -> dict[str, Path]:
    """Load a completed NeuCodec cache without importing or running NeuCodec."""
    paths = encoded_dataset_paths(config)
    required = (paths["complete"], paths["train"], paths["eval"])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "NeuCodec cache is not complete. Run the NeuCodec preparation stage first. "
            f"Missing: {', '.join(missing)}"
        )
    LOGGER.info("Using completed encoded dataset cache: %s", paths["root"])
    return {key: paths[key] for key in ("root", "train", "eval")}


def prepare_encoded_dataset(
    config: PipelineConfig,
    progress_callback: Callable[[], None] | None = None,
) -> dict[str, Path]:
    """Encode every train/eval WAV once and cache the result on the output Volume."""
    paths = encoded_dataset_paths(config)
    cache_root = paths["root"]
    train_output = paths["train"]
    eval_output = paths["eval"]
    complete_marker = paths["complete"]
    fingerprint = cache_root.name

    if config.data.rebuild_encoded_cache and cache_root.exists():
        from .artifacts import safe_rmtree

        safe_rmtree(cache_root, config.output_root)

    if complete_marker.is_file() and train_output.is_file() and eval_output.is_file():
        LOGGER.info("Using encoded dataset cache: %s", cache_root)
        return {"root": cache_root, "train": train_output, "eval": eval_output}

    cache_root.mkdir(parents=True, exist_ok=True)
    audit_report = audit_dataset(config)
    atomic_write_json(cache_root / "audit_report.json", audit_report)

    try:
        import torch
        import torchaudio
        import soundfile as sf
        from neucodec import NeuCodec
        from sea_g2p import SEAPipeline
        from transformers import AutoTokenizer
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "Training dependency import failed while loading Torch/NeuCodec/Transformers: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    tokenizer_kwargs: dict[str, Any] = {
        "trust_remote_code": config.model.trust_remote_code,
    }
    if config.model.model_revision:
        tokenizer_kwargs["revision"] = config.model.model_revision
    tokenizer = AutoTokenizer.from_pretrained(config.model.base_model, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    _validate_special_tokens(tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_description = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    LOGGER.info("NeuCodec encoding device: %s (%s)", device, device_description)
    codec_kwargs: dict[str, Any] = {}
    if config.model.codec_revision:
        codec_kwargs["revision"] = config.model.codec_revision
    codec = NeuCodec.from_pretrained(config.model.codec_model, **codec_kwargs).to(device)
    codec.eval()
    phonemizer = SEAPipeline(lang="vi")

    split_specs = (
        ("train", config.dataset_root / config.data.train_metadata, train_output),
        ("eval", config.dataset_root / config.data.eval_metadata, eval_output),
    )
    split_reports: dict[str, Any] = {}
    for split_name, metadata_path, output_path in split_specs:
        rows = read_pipe_metadata(metadata_path)
        expected_names = {row.filename for row in rows}
        if output_path.is_file():
            finalized_names = _read_completed_filenames(output_path)
            if finalized_names != expected_names:
                raise ValueError(
                    f"Incomplete finalized {split_name} cache at {output_path}: "
                    f"found {len(finalized_names)}/{len(expected_names)} samples"
                )
            LOGGER.info("Reusing completed %s encoded split: %s", split_name, output_path)
            split_reports[split_name] = {
                "metadata": str(metadata_path),
                "output": str(output_path),
                "samples": len(finalized_names),
                "expected_samples": len(rows),
                "errors": 0,
                "reused": True,
            }
            continue
        partial_path = output_path.with_suffix(".partial.jsonl")
        completed_names = _read_completed_filenames(partial_path)
        errors: list[dict[str, str]] = []
        written = len(completed_names)
        LOGGER.info(
            "Encoding %s split: %d total, %d already cached",
            split_name,
            len(rows),
            written,
        )
        with partial_path.open("a", encoding="utf-8", newline="\n") as output_handle:
            for row in tqdm(rows, desc=f"NeuCodec {split_name}", unit="audio"):
                if row.filename in completed_names:
                    continue
                audio_path = config.dataset_root / config.data.audio_directory / row.filename
                should_commit = False
                try:
                    waveform_np, source_sample_rate = sf.read(
                        audio_path,
                        dtype="float32",
                        always_2d=True,
                    )
                    waveform = torch.from_numpy(waveform_np.T).float()
                    waveform = waveform.mean(dim=0, keepdim=True)
                    if source_sample_rate != config.data.codec_input_sample_rate:
                        waveform = torchaudio.functional.resample(
                            waveform,
                            source_sample_rate,
                            config.data.codec_input_sample_rate,
                        )
                    codec_input = waveform.unsqueeze(0).to(device)
                    with torch.inference_mode():
                        code_tensor = codec.encode_code(codec_input)
                    codes = [int(value) for value in code_tensor.detach().reshape(-1).cpu().tolist()]
                    if not codes:
                        raise ValueError("NeuCodec returned no codes")
                    if min(codes) < 0 or max(codes) >= 65536:
                        raise ValueError(f"NeuCodec code outside [0, 65535]: {min(codes)}..{max(codes)}")

                    phones = phonemizer.run(row.text, punc_norm=True)
                    sequence = format_training_sequence(phones, codes)
                    token_ids = tokenizer.encode(sequence)
                    token_count = len(token_ids)
                    if token_count > config.model.max_sequence_length:
                        raise ValueError(
                            f"sequence has {token_count} tokens, limit is {config.model.max_sequence_length}"
                        )
                    duration = waveform.shape[-1] / config.data.codec_input_sample_rate
                    payload = {
                        "filename": row.filename,
                        "text": row.text,
                        "phones": phones,
                        "codes": codes,
                        "duration_seconds": round(float(duration), 6),
                        "token_count": token_count,
                    }
                    output_handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
                    output_handle.flush()
                    written += 1
                    completed_names.add(row.filename)
                    should_commit = written % config.data.encoded_commit_interval == 0
                except Exception as exc:  # noqa: BLE001 - the filename and error are persisted
                    error = {"filename": row.filename, "error": str(exc)}
                    errors.append(error)
                    LOGGER.exception("Failed to encode %s", row.filename)
                    if config.data.fail_on_data_error:
                        atomic_write_json(cache_root / f"{split_name}_errors.json", {"errors": errors})
                        raise RuntimeError(f"Failed to encode {row.filename}: {exc}") from exc
                if should_commit and progress_callback is not None:
                    os.fsync(output_handle.fileno())
                    progress_callback()

        if errors:
            atomic_write_json(cache_root / f"{split_name}_errors.json", {"errors": errors})
        if written != len(rows) and config.data.fail_on_data_error:
            raise RuntimeError(f"Encoded {written}/{len(rows)} {split_name} samples")
        if written == 0:
            raise RuntimeError(f"No {split_name} samples were encoded successfully")
        os.replace(partial_path, output_path)
        if progress_callback is not None:
            progress_callback()
        split_reports[split_name] = {
            "metadata": str(metadata_path),
            "output": str(output_path),
            "samples": written,
            "expected_samples": len(rows),
            "errors": len(errors),
            "reused": False,
        }

    report = {
        "completed_at": utc_now(),
        "fingerprint": fingerprint,
        "codec_model": config.model.codec_model,
        "codec_revision": config.model.codec_revision,
        "codec_input_sample_rate": config.data.codec_input_sample_rate,
        "base_model_tokenizer": config.model.base_model,
        "base_model_revision": config.model.model_revision,
        "max_sequence_length": config.model.max_sequence_length,
        "splits": split_reports,
    }
    atomic_write_json(cache_root / "encoding_report.json", report)
    atomic_write_json(complete_marker, report)
    import gc

    del codec, tokenizer, phonemizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"root": cache_root, "train": train_output, "eval": eval_output}


def _read_completed_filenames(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    completed: set[str] = set()
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    valid_lines: list[str] = []
    needs_repair = False
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                payload = json.loads(line)
                filename = payload["filename"]
                if filename in completed:
                    raise ValueError(f"Duplicate filename in partial encoded cache: {filename}")
                completed.add(filename)
                valid_lines.append(line if line.endswith("\n") else line + "\n")
            except (json.JSONDecodeError, KeyError) as exc:
                if line_number != len(lines):
                    raise ValueError(f"Corrupt partial encoded cache at {path}:{line_number}") from exc
                needs_repair = True
                break
    if needs_repair:
        temporary = path.with_suffix(path.suffix + ".repair")
        temporary.write_text("".join(valid_lines), encoding="utf-8", newline="\n")
        os.replace(temporary, path)
        LOGGER.warning("Removed an incomplete trailing cache row from %s", path)
    return completed


def _validate_special_tokens(tokenizer: Any) -> None:
    unk_id = tokenizer.unk_token_id
    for token in (TEXT_PROMPT_START, TEXT_PROMPT_END, SPEECH_GENERATION_START, SPEECH_GENERATION_END):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or (unk_id is not None and token_id == unk_id):
            raise ValueError(f"Base tokenizer is missing required special token: {token}")


class EncodedTTSDataset:
    def __init__(self, path: str | Path, tokenizer: Any, max_sequence_length: int) -> None:
        try:
            from torch.utils.data import Dataset as TorchDataset
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to construct the training dataset") from exc
        # Registering with TorchDataset is not required by DataLoader; this keeps imports lazy.
        del TorchDataset
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                payload = json.loads(line)
                if payload["token_count"] > max_sequence_length:
                    raise ValueError(f"Encoded row exceeds max length at {self.path}:{line_number}")
                self.rows.append(payload)
        if not self.rows:
            raise ValueError(f"Encoded dataset is empty: {self.path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        row = self.rows[index]
        sequence = format_training_sequence(row["phones"], row["codes"])
        input_ids = self.tokenizer.encode(sequence)
        if len(input_ids) > self.max_sequence_length:
            raise ValueError(f"Sample {row['filename']} exceeds max sequence length after cache creation")
        speech_start_id = self.tokenizer.convert_tokens_to_ids(SPEECH_GENERATION_START)
        try:
            speech_start_index = input_ids.index(speech_start_id)
        except ValueError as exc:
            raise ValueError(f"Speech generation start token missing in {row['filename']}") from exc
        labels = [-100] * len(input_ids)
        labels[speech_start_index:] = input_ids[speech_start_index:]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long),
        }


class DynamicTTSCollator:
    def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8) -> None:
        self.pad_token_id = pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_length = max(feature["input_ids"].numel() for feature in features)
        padded_length = int(math.ceil(max_length / self.pad_to_multiple_of) * self.pad_to_multiple_of)
        batch_size = len(features)
        input_ids = torch.full((batch_size, padded_length), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, padded_length), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, padded_length), dtype=torch.long)
        for row_index, feature in enumerate(features):
            length = feature["input_ids"].numel()
            input_ids[row_index, :length] = feature["input_ids"]
            labels[row_index, :length] = feature["labels"]
            attention_mask[row_index, :length] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
