from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from .config import load_config
from .data import audit_dataset
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train VieNeu-TTS through LoRA, safe merge, and partial fine-tuning.",
    )
    parser.add_argument("--config", required=True, help="YAML pipeline configuration")
    parser.add_argument("--run-name", help="Override run.name")
    parser.add_argument("--dataset-root", help="Override data.dataset_root")
    parser.add_argument("--output-root", help="Override run.output_root")
    parser.add_argument("--phase1-lr", type=float, help="Override phase 1 learning rate")
    parser.add_argument("--phase1-epochs", type=float, help="Override phase 1 epochs")
    parser.add_argument("--phase3-lr", type=float, help="Override phase 3 learning rate")
    parser.add_argument("--phase3-epochs", type=float, help="Override phase 3 epochs")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume runtime checkpoints")
    parser.add_argument(
        "--rebuild-encoded-cache",
        action="store_true",
        help="Re-encode all WAV files even if a compatible cache exists",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate config and raw dataset without loading training models",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = load_config(Path(args.config))
    config.apply_overrides(
        run_name=args.run_name,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        phase1_learning_rate=args.phase1_lr,
        phase1_epochs=args.phase1_epochs,
        phase3_learning_rate=args.phase3_lr,
        phase3_epochs=args.phase3_epochs,
    )
    if args.no_resume:
        config.run.resume = False
    if args.rebuild_encoded_cache:
        config.data.rebuild_encoded_cache = True
    config.validate()

    result = audit_dataset(config) if args.validate_only else run_pipeline(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
