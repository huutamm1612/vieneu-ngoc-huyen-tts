from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
import unicodedata
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator


DEFAULT_DATASET_ID = "pnnbao-ump/ngochuyen_voice"
DEFAULT_DATASET_REVISION = "1ebbfbd1fb828dfd41bff8b1645ad9e56cfc614a"
DEFAULT_SPLIT = "train"
DEFAULT_PREFIX = "ngoc_base01"
DEFAULT_INDEX_WIDTH = 5
DEFAULT_EVAL_RATIO = 0.02
DEFAULT_SEED = 3407
DEFAULT_MIN_DURATION = 3.0
DEFAULT_MAX_DURATION = 15.0
DEFAULT_SAMPLE_RATE = 24_000
DEFAULT_CHANNELS = 1
DEFAULT_SUBTYPE = "PCM_16"

ALLOWED_PUNCTUATION = frozenset("!,.?…")
QUOTE_CHARACTERS = frozenset(
    {
        '"',
        "'",
        "`",
        "«",
        "´",
        "»",
        "‘",
        "’",
        "“",
        "”",
        "„",
        "‟",
        "‹",
        "›",
    }
)
PREFIX_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
TRAILING_INDEX_PATTERN = re.compile(r"(?P<index>\d+)$")

MANIFEST_FIELDS = (
    "file_name",
    "transcription",
    "transcription_raw",
    "duration",
    "source_sample_rate",
    "source_channels",
    "output_sample_rate",
    "output_channels",
    "output_subtype",
    "source_index",
    "source_file_name",
)


@dataclass(frozen=True, slots=True)
class ProcessingOptions:
    dataset_id: str
    split: str
    requested_revision: str
    resolved_revision: str
    output_root: Path
    prefix: str
    index_width: int
    eval_ratio: float
    seed: int
    min_duration: float
    max_duration: float
    target_sample_rate: int
    cache_dir: Path | None


@dataclass(frozen=True, slots=True)
class TextResult:
    raw: str
    after_punctuation: str
    normalized: str | None
    forbidden_characters: tuple[str, ...]
    reasons: tuple[str, ...]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Ngoc Huyen dataset from Hugging Face and reproduce the local "
            "audio/ + metadata files used by this project. Existing samples are never deleted "
            "or overwritten; identical files are reused and conflicts stop the run."
        )
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--revision",
        default=DEFAULT_DATASET_REVISION,
        help=(
            "Hugging Face dataset revision. The default is the immutable revision used by "
            "the current training dataset; pass main explicitly to process the latest revision."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Output dataset root (default: the directory containing this script).",
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--index-width", type=int, default=DEFAULT_INDEX_WIDTH)
    parser.add_argument("--eval-ratio", type=float, default=DEFAULT_EVAL_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION)
    parser.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION)
    parser.add_argument("--target-sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face cache directory. Source shards remain in this cache.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Audit an already generated output without downloading the source dataset.",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not str(args.dataset_id).strip() or not str(args.split).strip():
        raise ValueError("dataset-id and split cannot be empty")
    if not str(args.revision).strip():
        raise ValueError("revision cannot be empty")
    if not PREFIX_PATTERN.fullmatch(args.prefix):
        raise ValueError(
            "prefix must contain only lowercase ASCII letters, digits, '-' or '_', "
            "and start with a letter or digit"
        )
    if args.index_width <= 0:
        raise ValueError("index-width must be positive")
    if not 0.0 <= args.eval_ratio < 1.0:
        raise ValueError("eval-ratio must be in [0, 1)")
    if args.min_duration <= 0 or args.max_duration <= args.min_duration:
        raise ValueError("expected 0 < min-duration < max-duration")
    if args.target_sample_rate <= 0:
        raise ValueError("target-sample-rate must be positive")


def _require_hf_api() -> Any:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Missing huggingface-hub. From the repository root run: "
            "python -m pip install -e \".[data]\""
        ) from exc
    return HfApi


def _require_runtime_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import numpy as np
        import soundfile as sf
        import soxr
        from datasets import Audio, load_dataset
        from sea_g2p import Normalizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dataset dependencies. From the repository root run: "
            "python -m pip install -e \".[data]\""
        ) from exc
    return np, sf, soxr, Audio, load_dataset, Normalizer


def resolve_revision(dataset_id: str, requested_revision: str, hf_api_type: Any) -> str:
    info = hf_api_type().dataset_info(dataset_id, revision=requested_revision)
    revision = str(getattr(info, "sha", "") or "").strip()
    if not revision:
        raise RuntimeError(f"Hugging Face did not resolve a commit SHA for {dataset_id}")
    return revision


def source_basename(value: str) -> str:
    normalized = str(value).strip().replace("\\", "/")
    name = PurePosixPath(normalized).name
    if not name or name in {".", ".."}:
        raise ValueError(f"Invalid source file_name: {value!r}")
    return name


def target_filename(source_file_name: str, prefix: str, index_width: int) -> str:
    source_name = source_basename(source_file_name)
    source_path = Path(source_name)
    if source_path.suffix.casefold() != ".wav":
        raise ValueError(f"Expected a WAV source file: {source_file_name}")
    match = TRAILING_INDEX_PATTERN.search(source_path.stem)
    if match is None:
        raise ValueError(f"Source filename has no trailing numeric ID: {source_file_name}")
    index = int(match.group("index"))
    return f"{prefix}_{index:0{index_width}d}.wav"


def forbidden_characters(text: str, *, allow_digits: bool) -> tuple[str, ...]:
    forbidden: set[str] = set()
    for character in text:
        if character.isspace():
            continue
        category = unicodedata.category(character)[0]
        if category in {"L", "M"} or (allow_digits and category == "N"):
            continue
        if character in ALLOWED_PUNCTUATION:
            continue
        forbidden.add(character)
    return tuple(sorted(forbidden))


def prepare_text(raw_value: Any, normalizer: Any) -> TextResult:
    raw = unicodedata.normalize("NFC", str(raw_value or "")).strip()
    after_punctuation = "".join(
        character for character in raw if character not in QUOTE_CHARACTERS
    )
    after_punctuation = re.sub(r"[;:]", ". ", after_punctuation)
    after_punctuation = re.sub(r"\s+", " ", after_punctuation).strip()
    after_punctuation = re.sub(r"\s+([!,.?…])", r"\1", after_punctuation)
    after_punctuation = re.sub(r"([!,.?…])(?=\S)", r"\1 ", after_punctuation)
    before_forbidden = forbidden_characters(after_punctuation, allow_digits=True)
    if not after_punctuation:
        return TextResult(
            raw=raw,
            after_punctuation=after_punctuation,
            normalized=None,
            forbidden_characters=(),
            reasons=("empty_transcription",),
        )
    if before_forbidden:
        return TextResult(
            raw=raw,
            after_punctuation=after_punctuation,
            normalized=None,
            forbidden_characters=before_forbidden,
            reasons=("forbidden_character_before_number_normalization",),
        )

    normalized_value = normalizer.normalize(after_punctuation)
    if not isinstance(normalized_value, str):
        raise TypeError(f"sea-g2p Normalizer returned {type(normalized_value).__name__}, expected str")
    normalized = unicodedata.normalize("NFC", normalized_value).lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if normalized.endswith(","):
        normalized = normalized.rstrip(" ,") + "."
    elif normalized and not normalized.endswith((".", "!", "?")):
        normalized += "."
    after_forbidden = forbidden_characters(normalized, allow_digits=False)
    reasons: list[str] = []
    if not normalized:
        reasons.append("empty_after_number_normalization")
    if any(character.isdigit() for character in normalized):
        reasons.append("digit_after_number_normalization")
    if after_forbidden:
        reasons.append("forbidden_character_after_number_normalization")
    if "|" in normalized or "\r" in normalized or "\n" in normalized:
        reasons.append("metadata_delimiter_or_newline_after_normalization")
    return TextResult(
        raw=raw,
        after_punctuation=after_punctuation,
        normalized=normalized if not reasons else None,
        forbidden_characters=after_forbidden,
        reasons=tuple(reasons),
    )


def _audio_payload(audio_value: Any) -> bytes:
    if not isinstance(audio_value, dict):
        raise TypeError("audio must be a decode=False dictionary")
    payload = audio_value.get("bytes")
    path = audio_value.get("path")
    if payload is not None:
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError("audio bytes have an unsupported type")
        return bytes(payload)
    if path:
        source = Path(str(path)).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Decoded audio path does not exist: {source}")
        return source.read_bytes()
    raise ValueError("audio has neither bytes nor path")


def process_audio(
    audio_value: Any,
    *,
    target_sample_rate: int,
    np: Any,
    sf: Any,
    soxr: Any,
) -> tuple[bytes, float, int, int]:
    source_payload = _audio_payload(audio_value)
    info = sf.info(io.BytesIO(source_payload))
    audio, source_sample_rate = sf.read(
        io.BytesIO(source_payload),
        dtype="float32",
        always_2d=True,
    )
    if audio.size == 0 or audio.shape[0] == 0:
        raise ValueError("decoded audio is empty")
    source_channels = int(audio.shape[1])
    duration = float(audio.shape[0] / float(source_sample_rate))
    if (
        int(source_sample_rate) == target_sample_rate
        and source_channels == DEFAULT_CHANNELS
        and str(info.format).upper() == "WAV"
        and str(info.subtype).upper() == DEFAULT_SUBTYPE
    ):
        return source_payload, duration, int(source_sample_rate), source_channels

    mono = np.asarray(audio, dtype=np.float32).mean(axis=1)
    if int(source_sample_rate) != target_sample_rate:
        mono = soxr.resample(mono, int(source_sample_rate), target_sample_rate, quality="HQ")
    mono = np.asarray(mono, dtype=np.float32).reshape(-1, 1)
    output = io.BytesIO()
    sf.write(
        output,
        mono,
        target_sample_rate,
        format="WAV",
        subtype=DEFAULT_SUBTYPE,
    )
    return output.getvalue(), duration, int(source_sample_rate), source_channels


def _payload_digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_new_or_identical(path: Path, payload: bytes) -> bool:
    """Create path without overwriting it; reuse an existing byte-identical file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file():
            raise FileExistsError(f"Output path exists and is not a file: {path}")
        existing = path.read_bytes()
        if existing != payload:
            raise FileExistsError(
                f"Refusing to overwrite conflicting file: {path} "
                f"(existing sha256={_payload_digest(existing)}, new sha256={_payload_digest(payload)})"
            )
        return False

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            existing = path.read_bytes()
            if existing != payload:
                raise FileExistsError(f"Output appeared with different content: {path}")
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")


def _pipe_metadata_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(f"{row['file_name']}|{row['transcription']}\n" for row in rows).encode("utf-8")


def _manifest_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=MANIFEST_FIELDS, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def select_eval_names(rows: list[dict[str, Any]], eval_ratio: float, seed: int) -> set[str]:
    eval_count = int(len(rows) * eval_ratio)
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return {rows[index]["file_name"] for index in indices[:eval_count]}


def _special_character_counts(text_rejects: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for rejection in text_rejects:
        counts.update(rejection.get("forbidden_characters", ()))
    result: dict[str, int] = {}
    for character, count in counts.most_common():
        display = (
            character
            if character.isprintable() and not character.isspace()
            else character.encode("unicode_escape").decode("ascii")
        )
        name = unicodedata.name(character, "UNKNOWN")
        result[f"U+{ord(character):04X} '{display}' {name}"] = count
    return result


def _processing_config(options: ProcessingOptions) -> dict[str, Any]:
    return {
        "dataset_id": options.dataset_id,
        "dataset_split": options.split,
        "dataset_revision": options.resolved_revision,
        "allowed_punctuation": ["!", ",", ".", "?", "…"],
        "quote_characters_removed": sorted(QUOTE_CHARACTERS),
        "semicolon_and_colon_replacement": ".",
        "numbers_to_words": "sea-g2p Normalizer(lang='vi')",
        "target_sample_rate": options.target_sample_rate,
        "target_channels": DEFAULT_CHANNELS,
        "target_subtype": DEFAULT_SUBTYPE,
        "min_duration": options.min_duration,
        "max_duration": options.max_duration,
        "eval_ratio": options.eval_ratio,
        "seed": options.seed,
        "whisper_used": False,
    }


def _source_text(options: ProcessingOptions) -> bytes:
    value = (
        f"Derived from: {options.dataset_id}\n"
        f"Revision: {options.resolved_revision}\n"
        f"Source: https://huggingface.co/datasets/{options.dataset_id}\n"
        "License: CC BY-NC 4.0\n"
        "Whisper/ASR used: no\n"
    )
    return value.encode("utf-8")


def _readme_text() -> bytes:
    return (
        "ngochuyen_story_tts_clean\n\n"
        "metadata.csv: train rows in filename.wav|text format.\n"
        "metadata_eval.csv: eval rows in filename.wav|text format.\n"
        "manifest.csv: processed metadata and raw transcript audit.\n"
        "audio/: only audio referenced by the clean metadata.\n"
        "text_rejects.json: samples rejected by text rules.\n"
        "audio_rejects.json: samples rejected by audio rules.\n"
    ).encode("utf-8")


def _dataset_rows(dataset: Any) -> Iterator[tuple[int, dict[str, Any]]]:
    required = {"audio", "transcription", "file_name"}
    columns = set(getattr(dataset, "column_names", ()))
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"Hugging Face dataset is missing columns: {', '.join(missing)}")
    for source_index in range(len(dataset)):
        row = dataset[source_index]
        if not isinstance(row, dict):
            raise TypeError(f"Dataset row {source_index} is not a mapping")
        yield source_index, row


def existing_complete_report(options: ProcessingOptions) -> dict[str, Any] | None:
    required = (
        "metadata.csv",
        "metadata_eval.csv",
        "manifest.csv",
        "text_rejects.json",
        "audio_rejects.json",
        "processing_config.json",
        "processing_stats.json",
        "SOURCE.txt",
    )
    if any(not (options.output_root / name).is_file() for name in required):
        return None
    try:
        config = json.loads((options.output_root / "processing_config.json").read_text(encoding="utf-8"))
        stats = json.loads((options.output_root / "processing_stats.json").read_text(encoding="utf-8"))
        expected_config = {
            "dataset_id": options.dataset_id,
            "dataset_split": options.split,
            "dataset_revision": options.resolved_revision,
            "target_sample_rate": options.target_sample_rate,
            "target_channels": DEFAULT_CHANNELS,
            "target_subtype": DEFAULT_SUBTYPE,
            "min_duration": options.min_duration,
            "max_duration": options.max_duration,
            "eval_ratio": options.eval_ratio,
            "seed": options.seed,
        }
        if any(config.get(key) != value for key, value in expected_config.items()):
            return None
        report = audit_output(options.output_root)
        if report["extra_wav_preserved"] != 0:
            return None
        if int(stats.get("audio_valid_samples", -1)) != report["referenced_audio"]:
            return None
        if int(stats.get("train_samples", -1)) != report["train_samples"]:
            return None
        if int(stats.get("eval_samples", -1)) != report["eval_samples"]:
            return None
        train_names = {name for name, _ in _read_pipe_metadata(options.output_root / "metadata.csv")}
        eval_names = {name for name, _ in _read_pipe_metadata(options.output_root / "metadata_eval.csv")}
        expected_prefix = f"{options.prefix}_"
        if any(not name.startswith(expected_prefix) for name in train_names | eval_names):
            return None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    report.update(
        {
            "already_complete": True,
            "source_cache_preserved": True,
            "message": "Compatible dataset already exists; download and processing were skipped.",
        }
    )
    return report


def process_dataset(options: ProcessingOptions) -> dict[str, Any]:
    print(f"Dataset       : {options.dataset_id}")
    print(f"Revision      : {options.resolved_revision}")
    print(f"Split         : {options.split}")
    print(f"Output        : {options.output_root}")
    print("Safety        : no existing sample will be deleted or overwritten")
    print()

    completed = existing_complete_report(options)
    if completed is not None:
        return completed

    np, sf, soxr, Audio, load_dataset, normalizer_type = _require_runtime_dependencies()

    dataset = load_dataset(
        options.dataset_id,
        split=options.split,
        revision=options.resolved_revision,
        cache_dir=str(options.cache_dir) if options.cache_dir else None,
    )
    dataset = dataset.cast_column("audio", Audio(decode=False))
    normalizer = normalizer_type(lang="vi")
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError('Missing tqdm. Run: python -m pip install -e ".[data]"') from exc

    audio_root = options.output_root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    text_rejects: list[dict[str, Any]] = []
    audio_rejects: list[dict[str, Any]] = []
    source_names: set[str] = set()
    target_names: set[str] = set()
    created_audio = 0
    reused_audio = 0

    progress = tqdm(_dataset_rows(dataset), total=len(dataset), desc="Prepare dataset", unit="audio")
    for source_index, row in progress:
        source_file_name = str(row.get("file_name") or "")
        source_name = source_basename(source_file_name)
        source_key = source_name.casefold()
        if source_key in source_names:
            raise ValueError(f"Duplicate source filename at row {source_index}: {source_name}")
        source_names.add(source_key)

        output_name = target_filename(source_name, options.prefix, options.index_width)
        target_key = output_name.casefold()
        if target_key in target_names:
            raise ValueError(f"Duplicate output filename at row {source_index}: {output_name}")
        target_names.add(target_key)

        text = prepare_text(row.get("transcription"), normalizer)
        if text.reasons:
            text_rejects.append(
                {
                    "source_index": source_index,
                    "source_file_name": source_file_name,
                    "file_name": source_name,
                    "transcription_raw": text.raw,
                    "transcription_after_punctuation": text.after_punctuation,
                    "forbidden_characters": list(text.forbidden_characters),
                    "reasons": list(text.reasons),
                }
            )
            continue
        assert text.normalized is not None

        audio_payload, duration, source_sample_rate, source_channels = process_audio(
            row.get("audio"),
            target_sample_rate=options.target_sample_rate,
            np=np,
            sf=sf,
            soxr=soxr,
        )
        if not options.min_duration <= duration <= options.max_duration:
            audio_rejects.append(
                {
                    "source_index": source_index,
                    "source_file_name": source_file_name,
                    "file_name": source_name,
                    "transcription_raw": text.raw,
                    "transcription": text.normalized,
                    "duration": duration,
                    "source_sample_rate": source_sample_rate,
                    "source_channels": source_channels,
                    "reasons": ["duration_outside_range"],
                }
            )
            continue

        if write_new_or_identical(audio_root / output_name, audio_payload):
            created_audio += 1
        else:
            reused_audio += 1
        accepted.append(
            {
                "file_name": output_name,
                "transcription": text.normalized,
                "transcription_raw": text.raw,
                "duration": duration,
                "source_sample_rate": source_sample_rate,
                "source_channels": source_channels,
                "output_sample_rate": options.target_sample_rate,
                "output_channels": DEFAULT_CHANNELS,
                "output_subtype": DEFAULT_SUBTYPE,
                "source_index": source_index,
                "source_file_name": f"audio/{output_name}",
            }
        )
        progress.set_postfix(
            valid=len(accepted),
            text_reject=len(text_rejects),
            audio_reject=len(audio_rejects),
        )

    if not accepted:
        raise RuntimeError("No valid samples were produced")
    eval_names = select_eval_names(accepted, options.eval_ratio, options.seed)
    train_rows = [row for row in accepted if row["file_name"] not in eval_names]
    eval_rows = [row for row in accepted if row["file_name"] in eval_names]
    total_hours = sum(float(row["duration"]) for row in accepted) / 3600.0
    stats = {
        "source_samples": len(dataset),
        "text_valid_samples": len(dataset) - len(text_rejects),
        "text_rejected_samples": len(text_rejects),
        "audio_valid_samples": len(accepted),
        "audio_rejected_samples": len(audio_rejects),
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "total_audio_hours": total_hours,
        "special_character_counts": _special_character_counts(text_rejects),
    }

    outputs = {
        options.output_root / "metadata.csv": _pipe_metadata_bytes(train_rows),
        options.output_root / "metadata_eval.csv": _pipe_metadata_bytes(eval_rows),
        options.output_root / "manifest.csv": _manifest_bytes(accepted),
        options.output_root / "text_rejects.json": _json_bytes(text_rejects),
        options.output_root / "audio_rejects.json": _json_bytes(audio_rejects),
        options.output_root / "processing_config.json": _json_bytes(_processing_config(options)),
        options.output_root / "processing_stats.json": _json_bytes(stats),
        options.output_root / "SOURCE.txt": _source_text(options),
        options.output_root / "README.txt": _readme_text(),
    }
    created_metadata = 0
    reused_metadata = 0
    for path, payload in outputs.items():
        if write_new_or_identical(path, payload):
            created_metadata += 1
        else:
            reused_metadata += 1

    audit = audit_output(options.output_root)
    audit.update(
        {
            "source_samples": len(dataset),
            "text_rejected_samples": len(text_rejects),
            "audio_rejected_samples": len(audio_rejects),
            "created_audio": created_audio,
            "reused_audio": reused_audio,
            "created_metadata_files": created_metadata,
            "reused_metadata_files": reused_metadata,
            "source_cache_preserved": True,
        }
    )
    return audit


def _read_pipe_metadata(path: Path) -> list[tuple[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required metadata file not found: {path}")
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("|", maxsplit=1)
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                raise ValueError(f"Invalid metadata row at {path}:{line_number}")
            filename, text = (part.strip() for part in parts)
            if filename in seen:
                raise ValueError(f"Duplicate metadata filename at {path}:{line_number}: {filename}")
            seen.add(filename)
            rows.append((filename, text))
    if not rows:
        raise ValueError(f"Metadata file is empty: {path}")
    return rows


def audit_output(output_root: Path) -> dict[str, Any]:
    root = output_root.expanduser().resolve()
    audio_root = root / "audio"
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_root}")
    train_rows = _read_pipe_metadata(root / "metadata.csv")
    eval_rows = _read_pipe_metadata(root / "metadata_eval.csv")
    train_names = {filename for filename, _ in train_rows}
    eval_names = {filename for filename, _ in eval_rows}
    overlap = train_names & eval_names
    if overlap:
        raise ValueError(f"Train/eval filename overlap: {sorted(overlap)[0]}")
    referenced = train_names | eval_names
    missing = sorted(filename for filename in referenced if not (audio_root / filename).is_file())
    if missing:
        raise FileNotFoundError(f"Metadata references missing audio: {missing[0]}")
    wav_names = {path.name for path in audio_root.glob("*.wav")}
    extra = sorted(wav_names - referenced)
    return {
        "output_root": str(root),
        "train_samples": len(train_rows),
        "eval_samples": len(eval_rows),
        "referenced_audio": len(referenced),
        "wav_files": len(wav_names),
        "extra_wav_preserved": len(extra),
        "extra_wav_examples": extra[:10],
    }


def build_options(args: argparse.Namespace, resolved_revision: str) -> ProcessingOptions:
    return ProcessingOptions(
        dataset_id=str(args.dataset_id).strip(),
        split=str(args.split).strip(),
        requested_revision=str(args.revision).strip(),
        resolved_revision=resolved_revision,
        output_root=args.output_root.expanduser().resolve(),
        prefix=args.prefix,
        index_width=args.index_width,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        target_sample_rate=args.target_sample_rate,
        cache_dir=args.cache_dir.expanduser().resolve() if args.cache_dir else None,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_args(args)
        if args.verify_only:
            report = audit_output(args.output_root)
        else:
            hf_api_type = _require_hf_api()
            revision = resolve_revision(args.dataset_id, args.revision, hf_api_type)
            options = build_options(args, revision)
            options.output_root.mkdir(parents=True, exist_ok=True)
            report = process_dataset(options)
        print()
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Existing samples and Hugging Face cache were preserved.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, TypeError, ValueError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
