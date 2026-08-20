from __future__ import annotations

import argparse
import os
import secrets
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence


INPUT_VOLUME = "tts-inference-inputs"
OUTPUT_VOLUME = "tts-inference-results"
DEFAULT_MODEL = "/mnt/tts-results/runs/ngoc-a10-v1/phase3_finetune/final"
DEFAULT_REFERENCE_AUDIO = "/mnt/tts-dataset/dataset/audio/ngochuyen_00769.wav"
DEFAULT_REFERENCE_TEXT = (
    "Vì thế, Đồng chí luôn được các Đồng chí lãnh đạo cấp cao của Đảng "
    "tin tưởng, đánh giá cao."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a local TXT file, run VieNeu-TTS inference on Modal, wait for "
            "completion, and atomically download the final WAV."
        )
    )
    parser.add_argument("input_txt", type=Path, help="Local input TXT path")
    parser.add_argument("output_wav", type=Path, help="Local destination WAV path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model path on the training Volume")
    parser.add_argument(
        "--reference-audio",
        default=DEFAULT_REFERENCE_AUDIO,
        help="Reference WAV path inside a mounted Modal Volume",
    )
    parser.add_argument("--reference-text", default=DEFAULT_REFERENCE_TEXT)
    parser.add_argument("--gpu", default="A10")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-runtime-batch-size", type=int, default=0)
    parser.add_argument("--env", default="main", help="Modal environment")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output_wav if it already exists",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print commands without contacting Modal",
    )
    return parser


def _find_modal() -> str:
    modal = shutil.which("modal.exe") or shutil.which("modal")
    if modal is None:
        raise RuntimeError(
            "Modal CLI was not found. Activate the project environment and install modal==1.5.4."
        )
    return modal


def _job_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{secrets.token_hex(4)}"


def _print_command(command: Sequence[str]) -> None:
    print(f"> {subprocess.list2cmdline(list(command))}", flush=True)


def _run(command: Sequence[str], *, cwd: Path, dry_run: bool) -> None:
    _print_command(command)
    if dry_run:
        return
    completed = subprocess.run(list(command), cwd=cwd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}: "
            f"{subprocess.list2cmdline(list(command))}"
        )


def _ensure_input_volume(modal: str, environment: str, project_root: Path) -> None:
    lookup = subprocess.run(
        [modal, "volume", "ls", "--env", environment, INPUT_VOLUME],
        cwd=project_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if lookup.returncode == 0:
        return
    _run(
        [modal, "volume", "create", "--env", environment, INPUT_VOLUME],
        cwd=project_root,
        dry_run=False,
    )


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    input_path = args.input_txt.expanduser().resolve()
    output_path = args.output_wav.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input TXT not found: {input_path}")
    if input_path.suffix.casefold() != ".txt":
        raise ValueError(f"Input must end with .txt: {input_path}")
    if input_path.stat().st_size == 0:
        raise ValueError(f"Input TXT is empty: {input_path}")
    if output_path.suffix.casefold() != ".wav":
        raise ValueError(f"Output must end with .wav: {output_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it after a successful run."
        )
    if not 1 <= args.num_gpus <= 8:
        raise ValueError("--num-gpus must be between 1 and 8")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_runtime_batch_size < 0:
        raise ValueError("--max-runtime-batch-size cannot be negative")
    if not args.model.strip():
        raise ValueError("--model cannot be empty")
    if not args.reference_audio.strip() or not args.reference_text.strip():
        raise ValueError("Reference audio and reference text cannot be empty")
    return input_path, output_path


def _download_atomically(
    modal: str,
    environment: str,
    remote_output: str,
    output_path: Path,
    project_root: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{secrets.token_hex(6)}.partial")
    try:
        _run(
            [
                modal,
                "volume",
                "get",
                "--env",
                environment,
                "--force",
                OUTPUT_VOLUME,
                f"/{remote_output}",
                str(temporary),
            ],
            cwd=project_root,
            dry_run=False,
        )
        if not temporary.is_file() or temporary.stat().st_size <= 44:
            raise RuntimeError(f"Downloaded WAV is missing or invalid: {temporary}")
        with temporary.open("rb") as handle:
            header = handle.read(12)
        if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise RuntimeError(f"Downloaded file is not a RIFF/WAVE audio file: {temporary}")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        input_path, output_path = _validate_args(args)
        modal = _find_modal()
        project_root = Path(__file__).resolve().parents[1]
        modal_app = project_root / "cloud" / "modal_inference.py"
        if not modal_app.is_file():
            raise FileNotFoundError(f"Modal inference app not found: {modal_app}")

        job_id = _job_id()
        remote_input = f"jobs/{job_id}/input.txt"
        remote_output = f"jobs/{job_id}/output.wav"

        print(f"Job ID        : {job_id}")
        print(f"Local input   : {input_path}")
        print(f"Local output  : {output_path}")
        print(f"Remote input  : {INPUT_VOLUME}:/{remote_input}")
        print(f"Remote output : {OUTPUT_VOLUME}:/{remote_output}")
        print(f"GPU           : {args.gpu} x {args.num_gpus}")
        print(flush=True)

        if not args.dry_run:
            print("[1/3] Checking the Modal input Volume...", flush=True)
            _ensure_input_volume(modal, args.env, project_root)
        else:
            print("[1/3] Dry run: input Volume lookup skipped.", flush=True)

        print("[1/3] Uploading input TXT...", flush=True)
        upload_command = [
            modal,
            "volume",
            "put",
            "--env",
            args.env,
            "--force",
            INPUT_VOLUME,
            str(input_path),
            f"/{remote_input}",
        ]
        _run(upload_command, cwd=project_root, dry_run=args.dry_run)

        print("[2/3] Running Modal inference and waiting for completion...", flush=True)
        inference_command = [
            modal,
            "run",
            "--detach",
            "--env",
            args.env,
            str(modal_app),
            "--model",
            args.model,
            "--input-path",
            remote_input,
            "--output",
            remote_output,
            "--reference-audio",
            args.reference_audio,
            "--reference-text",
            args.reference_text,
            "--gpu",
            args.gpu,
            "--num-gpus",
            str(args.num_gpus),
            "--batch-size",
            str(args.batch_size),
            "--max-runtime-batch-size",
            str(args.max_runtime_batch_size),
        ]
        _run(inference_command, cwd=project_root, dry_run=args.dry_run)

        print("[3/3] Downloading the final WAV...", flush=True)
        if args.dry_run:
            download_command = [
                modal,
                "volume",
                "get",
                "--env",
                args.env,
                "--force",
                OUTPUT_VOLUME,
                f"/{remote_output}",
                str(output_path),
            ]
            _print_command(download_command)
            print("Dry run completed; no files were uploaded, generated, or downloaded.")
        else:
            _download_atomically(
                modal,
                args.env,
                remote_output,
                output_path,
                project_root,
            )
            print()
            print(f"Inference completed: {output_path}")
            print(f"Remote backup kept at {OUTPUT_VOLUME}:/{remote_output}")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. The remote job may continue because Modal was started with --detach.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
