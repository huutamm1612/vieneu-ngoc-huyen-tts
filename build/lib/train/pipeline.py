from __future__ import annotations

import logging
import os
import platform
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable

from .artifacts import (
    PipelineState,
    atomic_write_json,
    mark_phase_complete,
    phase_complete,
    phase_final_dir,
    read_json,
    safe_rmtree,
    utc_now,
)
from .config import PipelineConfig, dump_config, load_config
from .data import prepare_encoded_dataset, require_encoded_dataset
from .modeling import gpu_report
from .phases import merge_phase2, read_phase_report, train_phase1_lora, train_phase3_finetune

LOGGER = logging.getLogger(__name__)
CommitCallback = Callable[[], None]


def run_pipeline(
    config: PipelineConfig,
    *,
    commit_callback: CommitCallback | None = None,
) -> dict[str, Any]:
    """Run preparation and all training phases in the current environment."""
    prepare_pipeline_data(config, commit_callback=commit_callback)
    return run_training_pipeline(config, commit_callback=commit_callback)


def prepare_pipeline_data(
    config: PipelineConfig,
    *,
    commit_callback: CommitCallback | None = None,
) -> dict[str, Any]:
    """Resolve versions and build the reusable NeuCodec cache."""
    state = _initialize_pipeline(
        config,
        commit_callback=commit_callback,
        environment_filename="preparation_environment.json",
    )

    try:
        state.begin("prepare_data")
        encoded_paths = prepare_encoded_dataset(config, progress_callback=commit_callback)
        preparation_report = read_json(encoded_paths["root"] / "encoding_report.json", {})
        state.complete("prepare_data", str(encoded_paths["root"]), preparation_report)
        _commit(commit_callback)
        return {
            "run_name": config.run.name,
            "status": "prepared",
            "prepared_at": utc_now(),
            "run_root": str(config.run_root),
            "encoded_cache": str(encoded_paths["root"]),
        }
    except BaseException as exc:
        state.fail(exc)
        _commit(commit_callback)
        raise


def run_training_pipeline(
    config: PipelineConfig,
    *,
    commit_callback: CommitCallback | None = None,
) -> dict[str, Any]:
    """Run all three training phases from an already completed encoded cache."""
    state = _initialize_pipeline(
        config,
        commit_callback=commit_callback,
        environment_filename="environment.json",
    )

    try:
        encoded_paths = require_encoded_dataset(config)
        preparation_report = read_json(encoded_paths["root"] / "encoding_report.json", {})
        state.complete("prepare_data", str(encoded_paths["root"]), preparation_report)
        _commit(commit_callback)

        phases = (
            (
                "phase1_lora",
                lambda: train_phase1_lora(config, encoded_paths, commit_callback),
            ),
            ("phase2_merge", lambda: merge_phase2(config, encoded_paths)),
            (
                "phase3_finetune",
                lambda: train_phase3_finetune(config, encoded_paths, commit_callback),
            ),
        )
        for phase_name, runner in phases:
            if phase_complete(config.run_root, phase_name):
                final_dir = phase_final_dir(config.run_root, phase_name)
                report = read_phase_report(final_dir)
                LOGGER.info("Skipping completed phase: %s", phase_name)
                state.complete(phase_name, str(final_dir), report)
                _commit(commit_callback)
                _cleanup_phase_runtime(config, phase_name, commit_callback)
                continue

            state.begin(phase_name)
            final_dir, report = runner()
            mark_phase_complete(config.run_root, phase_name, report)
            state.complete(phase_name, str(final_dir), report)
            _commit(commit_callback)
            _cleanup_phase_runtime(config, phase_name, commit_callback)

        final_artifact = phase_final_dir(config.run_root, "phase3_finetune")
        _prune_unrequested_artifacts(config)
        state.finish(str(final_artifact))
        summary = {
            "run_name": config.run.name,
            "status": "completed",
            "completed_at": utc_now(),
            "run_root": str(config.run_root),
            "final_model": str(final_artifact),
            "phase1_adapter": _artifact_or_none(config, "phase1_lora"),
            "phase2_merged_model": _artifact_or_none(config, "phase2_merge"),
            "encoded_cache": str(encoded_paths["root"]),
        }
        atomic_write_json(config.run_root / "pipeline_summary.json", summary)
        _commit(commit_callback)
        return summary
    except BaseException as exc:
        state.fail(exc)
        _commit(commit_callback)
        raise


def _initialize_pipeline(
    config: PipelineConfig,
    *,
    commit_callback: CommitCallback | None,
    environment_filename: str,
) -> PipelineState:
    config.validate()
    _seed_everything(config.run.seed)
    config.run_root.mkdir(parents=True, exist_ok=True)
    resolved_config_path = config.run_root / "resolved_config.yaml"
    _resolve_hub_revisions(config, resolved_config_path)
    config.validate()
    state = PipelineState(
        config.run_root / "pipeline_state.json",
        config.run.name,
        config.signature(),
    )
    dump_config(config, resolved_config_path)
    atomic_write_json(config.run_root / environment_filename, _environment_report())
    _commit(commit_callback)
    return state


def _seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def _resolve_hub_revisions(config: PipelineConfig, resolved_config_path: Path) -> None:
    if resolved_config_path.is_file():
        previous = load_config(resolved_config_path)
        if previous.model.base_model != config.model.base_model:
            raise ValueError("Existing run uses a different base model")
        if previous.model.codec_model != config.model.codec_model:
            raise ValueError("Existing run uses a different codec model")
        if not _is_commit_sha(config.model.model_revision):
            config.model.model_revision = previous.model.model_revision
        if not _is_commit_sha(config.model.codec_revision):
            config.model.codec_revision = previous.model.codec_revision

    unresolved = []
    if not _is_commit_sha(config.model.model_revision):
        unresolved.append(("model_revision", config.model.base_model, config.model.model_revision))
    if not _is_commit_sha(config.model.codec_revision):
        unresolved.append(("codec_revision", config.model.codec_model, config.model.codec_revision))
    if not unresolved:
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface-hub is required to resolve immutable model revisions") from exc

    api = HfApi()
    for attribute, repo_id, requested_revision in unresolved:
        revision = api.model_info(repo_id=repo_id, revision=requested_revision).sha
        if not revision:
            raise RuntimeError(f"Hugging Face did not return an immutable revision for {repo_id}")
        setattr(config.model, attribute, revision)
        LOGGER.info("Resolved %s to %s", repo_id, revision)


def _is_commit_sha(revision: str | None) -> bool:
    return revision is not None and re.fullmatch(r"[0-9a-fA-F]{40}", revision) is not None
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "recorded_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        report["gpu"] = gpu_report()
    except ImportError:
        report["gpu"] = {"cuda_available": False, "reason": "PyTorch is not installed"}
    for package_name in ("transformers", "peft", "accelerate", "modal", "sea_g2p", "neucodec"):
        try:
            from importlib.metadata import version

            report.setdefault("packages", {})[package_name] = version(package_name.replace("_", "-"))
        except Exception:  # noqa: BLE001 - environment report must not stop training
            report.setdefault("packages", {})[package_name] = None
    return report


def _commit(callback: CommitCallback | None) -> None:
    if callback is not None:
        callback()


def _prune_unrequested_artifacts(config: PipelineConfig) -> None:
    choices = (
        (config.artifacts.keep_phase1_final, "phase1_lora"),
        (config.artifacts.keep_phase2_final, "phase2_merge"),
    )
    for keep, phase_name in choices:
        path = phase_final_dir(config.run_root, phase_name)
        if not keep and path.exists():
            safe_rmtree(path, config.run_root)


def _artifact_or_none(config: PipelineConfig, phase_name: str) -> str | None:
    path = phase_final_dir(config.run_root, phase_name)
    return str(path) if path.is_dir() else None


def _cleanup_phase_runtime(
    config: PipelineConfig,
    phase_name: str,
    commit_callback: CommitCallback | None,
) -> None:
    runtime_dir = config.run_root / ".runtime" / phase_name
    if config.run.cleanup_runtime_checkpoints and runtime_dir.exists():
        safe_rmtree(runtime_dir, config.run_root)
        _commit(commit_callback)
