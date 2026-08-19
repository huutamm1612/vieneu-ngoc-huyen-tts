from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, target)


def read_json(path: str | Path, default: Any = None) -> Any:
    source = Path(path)
    if not source.is_file():
        return default
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def phase_final_dir(run_root: Path, phase_name: str) -> Path:
    return run_root / phase_name / "final"


def phase_complete(run_root: Path, phase_name: str) -> bool:
    final_dir = phase_final_dir(run_root, phase_name)
    return final_dir.is_dir() and (final_dir / "COMPLETE.json").is_file()


def mark_phase_complete(run_root: Path, phase_name: str, report: dict[str, Any]) -> None:
    final_dir = phase_final_dir(run_root, phase_name)
    final_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        final_dir / "COMPLETE.json",
        {
            "phase": phase_name,
            "completed_at": utc_now(),
            **report,
        },
    )


def safe_rmtree(path: str | Path, allowed_root: str | Path) -> None:
    target = Path(path).resolve()
    root = Path(allowed_root).resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"Refusing to delete path outside allowed root: {target}")
    if target.exists():
        shutil.rmtree(target)


class PipelineState:
    def __init__(self, path: Path, run_name: str, config_signature: str) -> None:
        self.path = path
        existing = read_json(path)
        if existing is None:
            self.data: dict[str, Any] = {
                "run_name": run_name,
                "config_signature": config_signature,
                "status": "created",
                "current_phase": None,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "phases": {},
            }
            self.save()
        else:
            if existing.get("run_name") != run_name:
                raise ValueError("Existing pipeline state belongs to a different run")
            if existing.get("config_signature") != config_signature:
                raise ValueError(
                    "The run name already exists with a different resolved configuration. "
                    "Choose a new run name or restore the original configuration."
                )
            self.data = existing

    def save(self) -> None:
        self.data["updated_at"] = utc_now()
        atomic_write_json(self.path, self.data)

    def begin(self, phase: str) -> None:
        self.data["status"] = "running"
        self.data["current_phase"] = phase
        phase_state = self.data["phases"].setdefault(phase, {})
        phase_state["status"] = "running"
        phase_state["started_at"] = phase_state.get("started_at", utc_now())
        self.save()

    def complete(self, phase: str, artifact: str, report: dict[str, Any] | None = None) -> None:
        phase_state = self.data["phases"].setdefault(phase, {})
        phase_state.update(
            {
                "status": "completed",
                "completed_at": utc_now(),
                "artifact": artifact,
            }
        )
        if report:
            phase_state["report"] = report
        self.data["current_phase"] = None
        self.save()

    def finish(self, final_artifact: str) -> None:
        self.data["status"] = "completed"
        self.data["current_phase"] = None
        self.data["completed_at"] = utc_now()
        self.data["final_artifact"] = final_artifact
        self.save()

    def fail(self, error: BaseException) -> None:
        self.data["status"] = "failed"
        self.data["last_error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "recorded_at": utc_now(),
        }
        phase = self.data.get("current_phase")
        if phase:
            phase_state = self.data["phases"].setdefault(phase, {})
            phase_state["status"] = "failed"
            phase_state["last_error"] = self.data["last_error"]
        self.save()
