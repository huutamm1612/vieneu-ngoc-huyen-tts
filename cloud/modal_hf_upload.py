from __future__ import annotations

import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

import modal


APP_NAME = "vieneu-tts-hf-upload"
RESULTS_VOLUME_NAME = "tts-training-results"
HF_SECRET_NAME = "huggingface-secret"

REMOTE_RESULTS_ROOT = PurePosixPath("/mnt/tts-results")
DEFAULT_RUN_NAME = "ngoc-a10-v1"
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(
        "huggingface-hub>=0.34,<1",
        "hf-xet>=1.1,<2",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
)

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=False)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])


def _validate_repo_id(repo_id: str) -> str:
    value = repo_id.strip()
    parts = value.split("/")
    if len(parts) != 2 or any(not part or part in {".", ".."} for part in parts):
        raise ValueError("repo_id must have the form 'username-or-org/model-name'")
    if any(char.isspace() or char in {"\\", ":"} for char in value):
        raise ValueError("repo_id contains an unsupported character")
    return value


def _validate_run_name(run_name: str) -> str:
    value = run_name.strip()
    if not RUN_NAME_PATTERN.fullmatch(value):
        raise ValueError(
            "run_name must start with a letter or number and contain only letters, "
            "numbers, dots, underscores, or hyphens"
        )
    return value


def _phase3_directory(run_name: str) -> Path:
    safe_run_name = _validate_run_name(run_name)
    return Path(REMOTE_RESULTS_ROOT / "runs" / safe_run_name / "phase3_finetune" / "final")


def _inspect_artifact(model_dir: Path) -> dict[str, Any]:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Phase 3 final directory was not found: {model_dir}")

    required_files = ["COMPLETE.json", "config.json", "tokenizer.json"]
    missing = [name for name in required_files if not (model_dir / name).is_file()]
    has_weights = (model_dir / "model.safetensors").is_file() or (
        model_dir / "model.safetensors.index.json"
    ).is_file()
    if not has_weights:
        missing.append("model.safetensors or model.safetensors.index.json")
    if missing:
        raise FileNotFoundError(
            f"Phase 3 artifact is incomplete at {model_dir}; missing: {', '.join(missing)}"
        )

    try:
        complete_payload = json.loads((model_dir / "COMPLETE.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid COMPLETE.json in {model_dir}: {exc}") from exc

    files = [path for path in model_dir.rglob("*") if path.is_file()]
    return {
        "directory": str(model_dir),
        "file_count": len(files),
        "size_bytes": sum(path.stat().st_size for path in files),
        "complete": complete_payload,
    }


@app.function(
    image=image,
    secrets=[hf_secret],
    cpu=2.0,
    memory=2 * 1024,
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=2, initial_delay=1.0),
    single_use_containers=True,
    volumes={str(REMOTE_RESULTS_ROOT): results_volume},
)
def upload_phase3_to_huggingface(
    repo_id: str,
    run_name: str = DEFAULT_RUN_NAME,
    private: bool = True,
    commit_message: str = "Upload Phase 3 final VieNeu-TTS model",
) -> dict[str, Any]:
    from huggingface_hub import HfApi

    safe_repo_id = _validate_repo_id(repo_id)
    safe_run_name = _validate_run_name(run_name)
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            f"Modal Secret '{HF_SECRET_NAME}' does not contain a non-empty HF_TOKEN"
        )

    # A fresh single-use container normally sees the newest Volume snapshot. Reloading
    # here also makes the behavior explicit if Modal later changes container reuse.
    results_volume.reload()
    model_dir = _phase3_directory(safe_run_name)
    artifact = _inspect_artifact(model_dir)

    api = HfApi(token=token)
    account = api.whoami(token=token)
    account_name = str(account.get("name") or account.get("fullname") or "unknown")

    repo_url = api.create_repo(
        repo_id=safe_repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
        token=token,
    )
    repo_info = api.model_info(repo_id=safe_repo_id, token=token)
    actual_private = bool(repo_info.private)
    if actual_private != private:
        requested = "private" if private else "public"
        actual = "private" if actual_private else "public"
        raise RuntimeError(
            f"Repository {safe_repo_id} is {actual}, but this upload requested {requested}. "
            "Change repository visibility explicitly on Hugging Face, then retry."
        )

    print(
        f"Uploading {artifact['file_count']} files "
        f"({artifact['size_bytes'] / (1024**3):.3f} GiB) from {model_dir}"
    )
    print(f"Hugging Face account: {account_name}")
    print(f"Destination: {safe_repo_id} ({'private' if private else 'public'})")

    commit = api.upload_folder(
        repo_id=safe_repo_id,
        repo_type="model",
        folder_path=model_dir,
        commit_message=commit_message.strip() or "Upload Phase 3 final VieNeu-TTS model",
        token=token,
        ignore_patterns=["**/.cache/**", "**/__pycache__/**", "*.tmp"],
    )
    result = {
        "status": "uploaded",
        "repo_id": safe_repo_id,
        "repo_url": str(repo_url),
        "private": actual_private,
        "run_name": safe_run_name,
        "source": str(model_dir),
        "file_count": artifact["file_count"],
        "size_bytes": artifact["size_bytes"],
        "commit_url": str(commit.commit_url),
        "commit_oid": commit.oid,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


@app.local_entrypoint()
def main(
    repo_id: str,
    run_name: str = DEFAULT_RUN_NAME,
    public: bool = False,
    commit_message: str = "Upload Phase 3 final VieNeu-TTS model",
) -> None:
    upload_phase3_to_huggingface.remote(
        repo_id=repo_id,
        run_name=run_name,
        private=not public,
        commit_message=commit_message,
    )
