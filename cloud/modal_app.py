from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import modal

APP_NAME = "vieneu-tts-three-phase"
DATASET_VOLUME_NAME = "tts-dataset"
RESULTS_VOLUME_NAME = "tts-training-results"
HF_CACHE_VOLUME_NAME = "tts-hf-cache"
HF_SECRET_NAME = "huggingface-secret"

REMOTE_REPO = PurePosixPath("/repo")
REMOTE_DATASET_ROOT = PurePosixPath("/mnt/tts-dataset/dataset")
REMOTE_RESULTS_ROOT = PurePosixPath("/mnt/tts-results")
REMOTE_HF_CACHE = PurePosixPath("/mnt/hf-cache")

LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .uv_pip_install(
        "accelerate==1.12.0",
        "huggingface-hub>=0.34,<1",
        "neucodec==0.0.6",
        "numpy>=2.0.2,<3",
        "peft==0.18.0",
        "PyYAML==6.0.2",
        "safetensors>=0.5,<1",
        "sea-g2p==0.8.4",
        "soundfile>=0.13,<1",
        "soxr>=0.5,<1",
        "torchao==0.13.0",
        "transformers==4.57.6",
        "tqdm>=4.67,<5",
    )
    .run_commands(
        "python -c \"from importlib.metadata import version; import torch, torchao; "
        "from neucodec import NeuCodec; "
        "assert torch.__version__.startswith('2.8.0'); "
        "print('dependency smoke test passed:', torch.__version__, version('torchao'))\""
    )
    .env(
        {
            "PYTHONPATH": str(REMOTE_REPO / "src"),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": str(REMOTE_HF_CACHE),
            "HF_HUB_CACHE": str(REMOTE_HF_CACHE / "hub"),
            "TORCH_HOME": str(REMOTE_HF_CACHE / "torch"),
            "HF_XET_HIGH_PERFORMANCE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_dir(LOCAL_PROJECT_ROOT / "src", str(REMOTE_REPO / "src"), copy=True)
    .add_local_dir(LOCAL_PROJECT_ROOT / "configs", str(REMOTE_REPO / "configs"), copy=True)
)

app = modal.App(APP_NAME)
dataset_volume = modal.Volume.from_name(DATASET_VOLUME_NAME, create_if_missing=False)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])


@app.function(
    image=image,
    secrets=[hf_secret],
    gpu="T4",
    cpu=4.0,
    memory=6 * 1024,
    timeout=24 * 60 * 60,
    startup_timeout=30 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=0.0),
    single_use_containers=True,
    volumes={
        str(REMOTE_DATASET_ROOT.parent): dataset_volume,
        str(REMOTE_RESULTS_ROOT): results_volume,
        str(REMOTE_HF_CACHE): hf_cache_volume,
    },
)
def prepare_neucodec_t4(
    config_name: str,
    run_name: str,
    phase1_lr: float = 0.0,
    phase1_epochs: float = 0.0,
    phase3_lr: float = 0.0,
    phase3_epochs: float = 0.0,
) -> dict[str, object]:
    import logging
    import sys

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("NeuCodec preparation requires the Modal T4 GPU, but CUDA is unavailable")

    sys.path.insert(0, str(REMOTE_REPO / "src"))
    from train.config import load_config
    from train.pipeline import prepare_pipeline_data

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    safe_config_name = Path(config_name).name
    if safe_config_name != config_name:
        raise ValueError("config_name must be a filename from the bundled configs directory")
    config_path = Path(REMOTE_REPO / "configs" / safe_config_name)
    config = load_config(config_path)
    config.apply_overrides(
        run_name=run_name,
        dataset_root=str(REMOTE_DATASET_ROOT),
        output_root=str(REMOTE_RESULTS_ROOT),
        phase1_learning_rate=phase1_lr or None,
        phase1_epochs=phase1_epochs or None,
        phase3_learning_rate=phase3_lr or None,
        phase3_epochs=phase3_epochs or None,
    )

    def commit_volumes() -> None:
        results_volume.commit()
        hf_cache_volume.commit()

    result = prepare_pipeline_data(config, commit_callback=commit_volumes)
    commit_volumes()
    return result


@app.function(
    image=image,
    secrets=[hf_secret],
    gpu="A10",
    cpu=4.0,
    memory=6 * 1024,
    timeout=24 * 60 * 60,
    startup_timeout=30 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=0.0),
    single_use_containers=True,
    volumes={
        str(REMOTE_DATASET_ROOT.parent): dataset_volume,
        str(REMOTE_RESULTS_ROOT): results_volume,
        str(REMOTE_HF_CACHE): hf_cache_volume,
    },
)
def train_three_phase(
    config_name: str,
    run_name: str,
    phase1_lr: float = 0.0,
    phase1_epochs: float = 0.0,
    phase3_lr: float = 0.0,
    phase3_epochs: float = 0.0,
) -> dict[str, object]:
    import logging
    import sys

    sys.path.insert(0, str(REMOTE_REPO / "src"))
    from train.config import load_config
    from train.pipeline import run_training_pipeline

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    safe_config_name = Path(config_name).name
    if safe_config_name != config_name:
        raise ValueError("config_name must be a filename from the bundled configs directory")
    config_path = Path(REMOTE_REPO / "configs" / safe_config_name)
    config = load_config(config_path)
    config.apply_overrides(
        run_name=run_name,
        dataset_root=str(REMOTE_DATASET_ROOT),
        output_root=str(REMOTE_RESULTS_ROOT),
        phase1_learning_rate=phase1_lr or None,
        phase1_epochs=phase1_epochs or None,
        phase3_learning_rate=phase3_lr or None,
        phase3_epochs=phase3_epochs or None,
    )

    def commit_volumes() -> None:
        results_volume.commit()
        hf_cache_volume.commit()

    result = run_training_pipeline(config, commit_callback=commit_volumes)
    commit_volumes()
    return result


@app.local_entrypoint()
def main(
    config: str = "pipeline_3phase.yaml",
    run_name: str = "ngoc-a10-v1",
    phase1_lr: float = 0.0,
    phase1_epochs: float = 0.0,
    phase3_lr: float = 0.0,
    phase3_epochs: float = 0.0,
) -> None:
    print("Stage 1/2: preparing NeuCodec cache on T4 (4 CPU cores, 6 GiB RAM)...")
    preparation = prepare_neucodec_t4.remote(
        config_name=config,
        run_name=run_name,
        phase1_lr=phase1_lr,
        phase1_epochs=phase1_epochs,
        phase3_lr=phase3_lr,
        phase3_epochs=phase3_epochs,
    )
    print(json.dumps(preparation, ensure_ascii=False, indent=2))
    print("Stage 2/2: starting all three training phases on A10...")
    result = train_three_phase.remote(
        config_name=config,
        run_name=run_name,
        phase1_lr=phase1_lr,
        phase1_epochs=phase1_epochs,
        phase3_lr=phase3_lr,
        phase3_epochs=phase3_epochs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
