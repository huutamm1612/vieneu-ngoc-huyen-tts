from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import modal

APP_NAME = "vieneu-tts-inference"
DATASET_VOLUME_NAME = "tts-dataset"
TRAINING_RESULTS_VOLUME_NAME = "tts-training-results"
INFERENCE_INPUTS_VOLUME_NAME = "tts-inference-inputs"
INFERENCE_RESULTS_VOLUME_NAME = "tts-inference-results"
HF_CACHE_VOLUME_NAME = "tts-hf-cache"
HF_SECRET_NAME = "huggingface-secret"

REMOTE_REPO = PurePosixPath("/repo")
REMOTE_DATASET = PurePosixPath("/mnt/tts-dataset/dataset")
REMOTE_TRAINING_RESULTS = PurePosixPath("/mnt/tts-results")
REMOTE_INPUTS = PurePosixPath("/mnt/inference-inputs")
REMOTE_OUTPUTS = PurePosixPath("/mnt/inference-results")
REMOTE_HF_CACHE = PurePosixPath("/mnt/hf-cache")
LOCAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL = str(REMOTE_TRAINING_RESULTS / "runs/ngoc-a10-v1/phase3_finetune/final")
DEFAULT_REFERENCE_AUDIO = str(REMOTE_DATASET / "audio/ngochuyen_00769.wav")
DEFAULT_REFERENCE_TEXT = (
    "Vì thế, Đồng chí luôn được các Đồng chí lãnh đạo cấp cao của Đảng "
    "tin tưởng, đánh giá cao."
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libsndfile1")
    .uv_pip_install(
        "torch==2.8.0",
        "torchaudio==2.8.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .uv_pip_install(
        "huggingface-hub>=0.34,<1",
        "librosa==0.11.0",
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
training_results_volume = modal.Volume.from_name(TRAINING_RESULTS_VOLUME_NAME, create_if_missing=False)
inference_inputs_volume = modal.Volume.from_name(INFERENCE_INPUTS_VOLUME_NAME, create_if_missing=True)
inference_results_volume = modal.Volume.from_name(INFERENCE_RESULTS_VOLUME_NAME, create_if_missing=True)
hf_cache_volume = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])


def _safe_output_path(relative_path: str) -> Path:
    root = Path(str(REMOTE_OUTPUTS)).resolve()
    candidate = (root / relative_path.lstrip("/")).resolve()
    if candidate == root or root not in candidate.parents or candidate.suffix.casefold() != ".wav":
        raise ValueError("output must be a relative .wav path inside the inference result Volume")
    return candidate


@app.function(
    image=image,
    secrets=[hf_secret],
    cpu=4.0,
    memory=6 * 1024,
    timeout=24 * 60 * 60,
    startup_timeout=30 * 60,
    single_use_containers=True,
    volumes={
        str(REMOTE_DATASET.parent): dataset_volume,
        str(REMOTE_TRAINING_RESULTS): training_results_volume,
        str(REMOTE_INPUTS): inference_inputs_volume,
        str(REMOTE_OUTPUTS): inference_results_volume,
        str(REMOTE_HF_CACHE): hf_cache_volume,
    },
)
def inference_job(
    model: str,
    input_path: str,
    text: str,
    output: str,
    reference_audio: str,
    reference_text: str,
    num_gpus: int,
    batch_size: int,
    max_runtime_batch_size: int,
) -> dict[str, object]:
    import logging
    import sys

    sys.path.insert(0, str(REMOTE_REPO / "src"))
    from inference import InferenceConfig, TTSInference

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if bool(input_path) == bool(text):
        raise ValueError("Provide exactly one of input_path or text")
    resolved_input: str | None = None
    if input_path:
        input_root = Path(str(REMOTE_INPUTS)).resolve()
        candidate = (input_root / input_path.lstrip("/")).resolve()
        if input_root not in candidate.parents or not candidate.is_file():
            raise FileNotFoundError(f"Input TXT is not inside the input Volume: {candidate}")
        resolved_input = str(candidate)
    output_path = _safe_output_path(output)
    config = InferenceConfig.from_yaml(REMOTE_REPO / "configs/inference.yaml")
    config.apply_overrides(
        model=model,
        devices="auto",
        num_gpus=num_gpus,
        batch_size=batch_size,
        max_runtime_batch_size=max_runtime_batch_size or None,
    )
    with TTSInference(config) as engine:
        result = engine.infer(
            text=text or None,
            input_path=resolved_input,
            output_path=output_path,
            reference_audio=reference_audio,
            reference_text=reference_text,
        )
    inference_results_volume.commit()
    hf_cache_volume.commit()
    return result.as_dict()


@app.local_entrypoint()
def main(
    model: str = DEFAULT_MODEL,
    input_path: str = "story.txt",
    text: str = "",
    output: str = "story_complete.wav",
    reference_audio: str = DEFAULT_REFERENCE_AUDIO,
    reference_text: str = DEFAULT_REFERENCE_TEXT,
    gpu: str = "A10",
    num_gpus: int = 1,
    batch_size: int = 128,
    max_runtime_batch_size: int = 0,
) -> None:
    if num_gpus < 1:
        raise ValueError("num_gpus must be positive")
    gpu_spec = gpu if num_gpus == 1 else f"{gpu}:{num_gpus}"
    result = inference_job.with_options(gpu=gpu_spec).remote(
        model=model,
        input_path="" if text else input_path,
        text=text,
        output=output,
        reference_audio=reference_audio,
        reference_text=reference_text,
        num_gpus=num_gpus,
        batch_size=batch_size,
        max_runtime_batch_size=max_runtime_batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
