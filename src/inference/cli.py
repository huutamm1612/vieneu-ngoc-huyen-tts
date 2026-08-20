from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .api import TTSInference
from .batching import validate_batches
from .config import InferenceConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Portable VieNeu-TTS inference")
    parser.add_argument("--config", help="Inference YAML file")
    parser.add_argument("--model", help="Final full model directory or Hugging Face repository")
    parser.add_argument("--model-revision")
    parser.add_argument("--adapter", help="Optional legacy LoRA adapter directory/repository")
    parser.add_argument("--base-model", help="Base model used with --adapter")
    parser.add_argument("--codec-model")
    parser.add_argument("--codec-revision")
    parser.add_argument("--reference-audio", required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--output", required=True, help="Final .wav path")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="UTF-8/UTF-16/CP1258 story TXT")
    source.add_argument("--text", help="Text to synthesize")
    source.add_argument("--batches-json", help="Prepared batch JSON produced by user code")

    parser.add_argument("--devices", help="auto, cpu, mps, cuda:0, or comma-separated CUDA devices")
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"))
    parser.add_argument("--min-chars", type=int)
    parser.add_argument("--target-chars", type=int)
    parser.add_argument("--max-chars", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-length-gap", type=int)
    parser.add_argument("--max-runtime-batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--min-new-tokens", type=int)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-compile", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--keep-segments", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--speed", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def _config_from_args(args: argparse.Namespace) -> InferenceConfig:
    config = InferenceConfig.from_yaml(args.config) if args.config else InferenceConfig()
    overrides: dict[str, Any] = {
        "model": args.model,
        "model_revision": args.model_revision,
        "adapter": args.adapter,
        "base_model": args.base_model,
        "codec_model": args.codec_model,
        "codec_revision": args.codec_revision,
        "devices": args.devices,
        "num_gpus": args.num_gpus,
        "dtype": args.dtype,
        "min_chars": args.min_chars,
        "target_chars": args.target_chars,
        "max_chars": args.max_chars,
        "batch_size": args.batch_size,
        "max_length_gap": args.max_length_gap,
        "max_runtime_batch_size": args.max_runtime_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "max_retries": args.max_retries,
        "do_sample": args.do_sample,
        "enable_compile": args.enable_compile,
        "keep_segments": args.keep_segments,
        "show_progress": args.show_progress,
        "speed": args.speed,
        "seed": args.seed,
    }
    config.apply_overrides(**overrides)
    return config


def _read_batches(path: str | None):
    if path is None:
        return None
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Batch JSON not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    return validate_batches(payload)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = _config_from_args(args)
    batches = _read_batches(args.batches_json)
    with TTSInference(config) as engine:
        result = engine.infer(
            output_path=args.output,
            reference_audio=args.reference_audio,
            reference_text=args.reference_text,
            batches=batches,
            text=args.text,
            input_path=args.input,
        )
    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
