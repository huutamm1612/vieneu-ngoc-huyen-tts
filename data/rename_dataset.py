from __future__ import annotations

import argparse
import codecs
import csv
import io
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PREFIX = "ngoc_base01"
DEFAULT_METADATA_FILES = ("metadata.csv", "metadata_eval.csv")
PREFIX_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
TRAILING_INDEX_PATTERN = re.compile(r"(?P<index>\d+)$")


@dataclass(frozen=True, slots=True)
class RenamePlan:
    dataset_root: Path
    audio_directory: Path
    prefix: str
    mapping: dict[str, str]
    rewritten_files: dict[Path, bytes]

    @property
    def changed_mapping(self) -> dict[str, str]:
        return {old: new for old, new in self.mapping.items() if old != new}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rename the current WAV dataset while keeping metadata.csv, "
            "metadata_eval.csv, and manifest.csv synchronized."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Dataset directory (default: the directory containing this script).",
    )
    parser.add_argument(
        "--audio-directory",
        default="audio",
        help="WAV directory relative to dataset root (default: audio).",
    )
    parser.add_argument(
        "--prefix",
        default=DEFAULT_PREFIX,
        help=f"New filename prefix (default: {DEFAULT_PREFIX}).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the plan. Without this flag the script is read-only.",
    )
    return parser.parse_args(argv)


def _casefold_duplicates(values: list[str]) -> list[str]:
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for value in values:
        key = value.casefold()
        if key in seen and seen[key] != value:
            duplicates.append(value)
        elif key in seen:
            duplicates.append(value)
        else:
            seen[key] = value
    return duplicates


def _build_mapping(audio_files: list[Path], prefix: str) -> dict[str, str]:
    if not PREFIX_PATTERN.fullmatch(prefix):
        raise ValueError(
            "prefix must contain only lowercase ASCII letters, digits, '-' or '_', "
            "and must start with a letter or digit"
        )

    mapping: dict[str, str] = {}
    used_indices: dict[int, str] = {}
    widths: list[int] = []
    parsed: list[tuple[Path, int, int]] = []
    for audio_path in audio_files:
        match = TRAILING_INDEX_PATTERN.search(audio_path.stem)
        if match is None:
            raise ValueError(
                f"Cannot preserve the numeric ID because the filename has no trailing number: "
                f"{audio_path.name}"
            )
        digits = match.group("index")
        index = int(digits)
        if index in used_indices:
            raise ValueError(
                f"Duplicate numeric ID {index}: {used_indices[index]} and {audio_path.name}"
            )
        used_indices[index] = audio_path.name
        widths.append(len(digits))
        parsed.append((audio_path, index, len(digits)))

    output_width = max(5, max(widths, default=5))
    for audio_path, index, _ in parsed:
        mapping[audio_path.name] = f"{prefix}_{index:0{output_width}d}.wav"

    duplicate_targets = _casefold_duplicates(list(mapping.values()))
    if duplicate_targets:
        raise ValueError(f"The rename plan contains duplicate targets: {duplicate_targets[0]}")
    return mapping


def _decode_utf8(path: Path) -> tuple[str, bool]:
    payload = path.read_bytes()
    has_bom = payload.startswith(codecs.BOM_UTF8)
    try:
        return payload.decode("utf-8-sig"), has_bom
    except UnicodeDecodeError as exc:
        raise ValueError(f"Expected an UTF-8 file: {path}") from exc


def _encode_utf8(text: str, with_bom: bool) -> bytes:
    payload = text.encode("utf-8")
    return codecs.BOM_UTF8 + payload if with_bom else payload


def _rewrite_pipe_metadata(path: Path, mapping: dict[str, str]) -> tuple[bytes, set[str]]:
    text, has_bom = _decode_utf8(path)
    output: list[str] = []
    referenced: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(keepends=True), start=1):
        line = raw_line.rstrip("\r\n")
        newline = raw_line[len(line) :]
        if not line:
            output.append(raw_line)
            continue
        parts = line.split("|", maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid metadata row at {path}:{line_number}; expected filename|text")
        filename, transcript = (part.strip() for part in parts)
        if not filename or not transcript:
            raise ValueError(f"Empty filename or transcript at {path}:{line_number}")
        if filename not in mapping:
            raise ValueError(f"Metadata references a WAV that is not in the audio directory: {filename}")
        if filename in referenced:
            raise ValueError(f"Duplicate filename in {path}: {filename}")
        referenced.add(filename)
        output.append(f"{mapping[filename]}|{transcript}{newline}")
    if not referenced:
        raise ValueError(f"Metadata file has no rows: {path}")
    return _encode_utf8("".join(output), has_bom), referenced


def _replace_path_basename(value: str, mapping: dict[str, str]) -> str:
    normalized = value.replace("\\", "/")
    directory, separator, filename = normalized.rpartition("/")
    if filename not in mapping:
        return value
    replacement = mapping[filename]
    return f"{directory}{separator}{replacement}" if separator else replacement


def _rewrite_manifest(path: Path, mapping: dict[str, str]) -> tuple[bytes, set[str]]:
    text, has_bom = _decode_utf8(path)
    newline = "\r\n" if "\r\n" in text else "\n"
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None or "file_name" not in reader.fieldnames:
        raise ValueError(f"Manifest must contain a file_name column: {path}")
    rows = list(reader)
    referenced: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        filename = (row.get("file_name") or "").strip()
        if filename not in mapping:
            raise ValueError(f"Unknown manifest file_name at {path}:{row_number}: {filename}")
        if filename in referenced:
            raise ValueError(f"Duplicate file_name in {path}: {filename}")
        referenced.add(filename)
        row["file_name"] = mapping[filename]
        if "source_file_name" in row and row["source_file_name"]:
            row["source_file_name"] = _replace_path_basename(row["source_file_name"], mapping)

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=reader.fieldnames, lineterminator=newline)
    writer.writeheader()
    writer.writerows(rows)
    return _encode_utf8(buffer.getvalue(), has_bom), referenced


def create_plan(dataset_root: Path, audio_directory: str, prefix: str) -> RenamePlan:
    root = dataset_root.expanduser().resolve()
    audio_root = (root / audio_directory).resolve()
    if root != audio_root and root not in audio_root.parents:
        raise ValueError("audio-directory must stay inside dataset-root")
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Audio directory not found: {audio_root}")

    audio_files = sorted(audio_root.glob("*.wav"), key=lambda path: path.name.casefold())
    if not audio_files:
        raise ValueError(f"No WAV files found in {audio_root}")
    source_names = [path.name for path in audio_files]
    if _casefold_duplicates(source_names):
        raise ValueError("The audio directory contains case-insensitive duplicate filenames")

    mapping = _build_mapping(audio_files, prefix)
    source_name_set = set(source_names)
    for old_name, new_name in mapping.items():
        target = audio_root / new_name
        if old_name != new_name and target.exists():
            raise FileExistsError(f"Rename target already exists: {target}")

    rewritten_files: dict[Path, bytes] = {}
    metadata_references: set[str] = set()
    for relative_path in DEFAULT_METADATA_FILES:
        metadata_path = root / relative_path
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Required metadata file not found: {metadata_path}")
        rewritten, referenced = _rewrite_pipe_metadata(metadata_path, mapping)
        rewritten_files[metadata_path] = rewritten
        overlap = metadata_references & referenced
        if overlap:
            raise ValueError(f"Train/eval filename overlap: {sorted(overlap)[0]}")
        metadata_references.update(referenced)

    if metadata_references != source_name_set:
        missing_from_metadata = sorted(source_name_set - metadata_references)
        missing_audio = sorted(metadata_references - source_name_set)
        raise ValueError(
            "Audio and metadata are not synchronized before rename. "
            f"Unreferenced WAVs: {len(missing_from_metadata)}; missing WAVs: {len(missing_audio)}"
        )

    manifest_path = root / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Required manifest file not found: {manifest_path}")
    rewritten_manifest, manifest_references = _rewrite_manifest(manifest_path, mapping)
    if manifest_references != source_name_set:
        raise ValueError("manifest.csv and the audio directory do not contain the same filenames")
    rewritten_files[manifest_path] = rewritten_manifest

    return RenamePlan(
        dataset_root=root,
        audio_directory=audio_root,
        prefix=prefix,
        mapping=mapping,
        rewritten_files=rewritten_files,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _mapping_payload(mapping: dict[str, str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(("old_name", "new_name"))
    for old_name, new_name in sorted(mapping.items(), key=lambda item: item[0].casefold()):
        if old_name != new_name:
            writer.writerow((old_name, new_name))
    return buffer.getvalue().encode("utf-8")


def apply_plan(plan: RenamePlan) -> None:
    changed = plan.changed_mapping
    if not changed:
        return

    original_metadata = {path: path.read_bytes() for path in plan.rewritten_files}
    temporary_audio: dict[str, Path] = {}
    mapping_path = plan.dataset_root / "rename_mapping.csv"
    mapping_existed = mapping_path.exists()
    original_mapping = mapping_path.read_bytes() if mapping_existed else None

    try:
        _atomic_write(mapping_path, _mapping_payload(changed))

        for old_name in changed:
            source = plan.audio_directory / old_name
            temporary = plan.audio_directory / f".rename-{uuid.uuid4().hex}.tmp"
            source.rename(temporary)
            temporary_audio[old_name] = temporary

        for old_name, new_name in changed.items():
            temporary = temporary_audio[old_name]
            target = plan.audio_directory / new_name
            if target.exists():
                raise FileExistsError(f"Rename target appeared during apply: {target}")
            temporary.rename(target)

        for path, payload in plan.rewritten_files.items():
            _atomic_write(path, payload)
    except BaseException:
        for old_name, new_name in reversed(list(changed.items())):
            old_path = plan.audio_directory / old_name
            new_path = plan.audio_directory / new_name
            temporary = temporary_audio.get(old_name)
            if not old_path.exists():
                if new_path.exists():
                    new_path.rename(old_path)
                elif temporary is not None and temporary.exists():
                    temporary.rename(old_path)
        for path, payload in original_metadata.items():
            _atomic_write(path, payload)
        if mapping_existed and original_mapping is not None:
            _atomic_write(mapping_path, original_mapping)
        else:
            mapping_path.unlink(missing_ok=True)
        raise


def print_plan(plan: RenamePlan, apply: bool) -> None:
    changed = plan.changed_mapping
    print(f"Dataset root : {plan.dataset_root}")
    print(f"Audio folder : {plan.audio_directory}")
    print(f"WAV files    : {len(plan.mapping)}")
    print(f"Rename count : {len(changed)}")
    print(f"Prefix       : {plan.prefix}")
    print()
    for old_name, new_name in list(changed.items())[:10]:
        print(f"  {old_name} -> {new_name}")
    if len(changed) > 10:
        print(f"  ... and {len(changed) - 10} more")
    print()
    if apply:
        print("Rename completed. metadata.csv, metadata_eval.csv, and manifest.csv are synchronized.")
        print(f"Mapping saved: {plan.dataset_root / 'rename_mapping.csv'}")
    else:
        print("Dry run only; no files were changed.")
        print("Run again with --apply after reviewing this plan.")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan = create_plan(args.dataset_root, args.audio_directory, args.prefix)
        if args.apply:
            apply_plan(plan)
        print_plan(plan, args.apply)
        return 0
    except (OSError, ValueError, csv.Error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
