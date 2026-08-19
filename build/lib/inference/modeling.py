from __future__ import annotations

import gc
import json
import logging
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import InferenceConfig

LOGGER = logging.getLogger(__name__)

TEXT_REPLACE = "<|TEXT_REPLACE|>"
SPEECH_REPLACE = "<|SPEECH_REPLACE|>"
TEXT_PROMPT_START = "<|TEXT_PROMPT_START|>"
TEXT_PROMPT_END = "<|TEXT_PROMPT_END|>"
SPEECH_GENERATION_START = "<|SPEECH_GENERATION_START|>"
SPEECH_GENERATION_END = "<|SPEECH_GENERATION_END|>"
CHAT_TEMPLATE = "user: Convert the text to speech:<|TEXT_REPLACE|>\nassistant:<|SPEECH_REPLACE|>"
SPEECH_CODE_COUNT = 65_536


@dataclass(slots=True)
class InferenceWorker:
    worker_id: int
    device: Any
    dtype: Any
    model: Any
    tokenizer: Any
    codec: Any
    pad_id: int
    text_replace_id: int
    speech_replace_id: int
    text_prompt_start_id: int
    text_prompt_end_id: int
    speech_start_id: int
    speech_end_id: int
    speech_token_min: int
    speech_token_max: int
    context_limit: int
    sample_rate: int = 24_000
    reference_phonemes: str | None = None
    reference_codes: list[int] = field(default_factory=list)
    reference_code_token_ids: list[int] = field(default_factory=list)
    codec_encoder_released: bool = False
    compile_disabled: bool = False

    def build_prompt(self, target_phonemes: str) -> list[int]:
        if not self.reference_phonemes or not self.reference_codes:
            raise RuntimeError("Reference audio has not been encoded")
        combined = f"{self.reference_phonemes} {target_phonemes.strip()}".strip()
        input_ids = self.tokenizer.encode(combined, add_special_tokens=False)
        ids = self.tokenizer.encode(CHAT_TEMPLATE)
        try:
            text_index = ids.index(self.text_replace_id)
        except ValueError as exc:
            raise RuntimeError("Tokenizer did not preserve the TEXT_REPLACE placeholder") from exc
        ids = (
            ids[:text_index]
            + [self.text_prompt_start_id]
            + list(input_ids)
            + [self.text_prompt_end_id]
            + ids[text_index + 1 :]
        )
        try:
            speech_index = ids.index(self.speech_replace_id)
        except ValueError as exc:
            raise RuntimeError("Tokenizer did not preserve the SPEECH_REPLACE placeholder") from exc
        return ids[:speech_index] + [self.speech_start_id] + self.reference_code_token_ids


def resolve_devices(config: InferenceConfig) -> list[str]:
    import torch

    requested = config.devices.strip()
    if requested and requested.lower() != "auto":
        devices = [value.strip() for value in requested.split(",") if value.strip()]
        if not devices:
            raise ValueError("devices cannot be empty")
        for device in devices:
            parsed = torch.device(device)
            if parsed.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA devices were requested but CUDA is unavailable")
                index = parsed.index if parsed.index is not None else 0
                if index >= torch.cuda.device_count():
                    raise RuntimeError(f"CUDA device index {index} is unavailable")
            elif parsed.type == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested but is unavailable")
            elif parsed.type not in {"cuda", "cpu", "mps"}:
                raise ValueError(f"Unsupported device: {device}")
        return devices

    if torch.cuda.is_available():
        available = torch.cuda.device_count()
        if config.num_gpus > available:
            raise RuntimeError(f"num_gpus={config.num_gpus}, but only {available} CUDA devices are visible")
        return [f"cuda:{index}" for index in range(config.num_gpus)]
    if config.num_gpus > 1:
        raise RuntimeError("Multiple workers require CUDA GPUs")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return ["mps"]
    return ["cpu"]


def resolve_dtype(config: InferenceConfig, device: Any):
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if config.dtype != "auto":
        dtype = mapping[config.dtype]
        if device.type != "cuda" and dtype != torch.float32:
            LOGGER.warning("Using float32 on %s instead of %s", device, config.dtype)
            return torch.float32
        return dtype
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(device)
        if capability[0] >= 8 and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        # The supplied Kaggle workflow intentionally uses FP32 on T4-class GPUs.
        return torch.float32
    return torch.float32


def _hub_kwargs(revision: str | None, path_or_repo: str, trust_remote_code: bool) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if revision and not Path(path_or_repo).expanduser().exists():
        kwargs["revision"] = revision
    return kwargs


def _adapter_base_model(adapter: str) -> str | None:
    path = Path(adapter).expanduser()
    config_path = path / "adapter_config.json"
    if not config_path.is_file():
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    value = payload.get("base_model_name_or_path")
    return str(value) if value else None


def _special_token_id(tokenizer: Any, token: str) -> int:
    value = tokenizer.convert_tokens_to_ids(token)
    if value is None or (tokenizer.unk_token_id is not None and value == tokenizer.unk_token_id):
        raise ValueError(f"Tokenizer is missing required token: {token}")
    return int(value)


def validate_tokenizer(tokenizer: Any) -> dict[str, int]:
    values = {
        "text_replace_id": _special_token_id(tokenizer, TEXT_REPLACE),
        "speech_replace_id": _special_token_id(tokenizer, SPEECH_REPLACE),
        "text_prompt_start_id": _special_token_id(tokenizer, TEXT_PROMPT_START),
        "text_prompt_end_id": _special_token_id(tokenizer, TEXT_PROMPT_END),
        "speech_start_id": _special_token_id(tokenizer, SPEECH_GENERATION_START),
        "speech_end_id": _special_token_id(tokenizer, SPEECH_GENERATION_END),
        "speech_token_min": _special_token_id(tokenizer, "<|speech_0|>"),
        "speech_token_max": _special_token_id(tokenizer, "<|speech_65535|>"),
    }
    minimum = values["speech_token_min"]
    if values["speech_token_max"] != minimum + SPEECH_CODE_COUNT - 1:
        raise RuntimeError("Tokenizer speech token IDs are not a contiguous 65,536-token range")
    for code in (0, 1, 255, 4096, 65_535):
        actual = _special_token_id(tokenizer, f"<|speech_{code}|>")
        if actual != minimum + code:
            raise RuntimeError(f"Tokenizer speech token range is not contiguous at code {code}")
    return values


def _load_tokenizer(config: InferenceConfig, source: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source,
        **_hub_kwargs(config.model_revision, source, config.trust_remote_code),
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _load_backbone(config: InferenceConfig, device: Any, dtype: Any):
    from transformers import AutoModelForCausalLM

    model_source = config.model
    if config.adapter:
        model_source = config.base_model or _adapter_base_model(config.adapter) or config.model
    kwargs = _hub_kwargs(config.model_revision, model_source, config.trust_remote_code)
    kwargs.update({"dtype": dtype, "low_cpu_mem_usage": True, "attn_implementation": "sdpa"})
    model = AutoModelForCausalLM.from_pretrained(model_source, **kwargs).to(device)
    if config.adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("PEFT is required to load an adapter") from exc
        LOGGER.info("Merging inference adapter %s into %s", config.adapter, model_source)
        model = PeftModel.from_pretrained(model, config.adapter, is_trainable=False)
        model = model.merge_and_unload(safe_merge=True)
    model.eval().requires_grad_(False)
    model.config.use_cache = True
    model.config.pad_token_id = None  # set after the tokenizer is validated
    return model, model_source


def _context_limit(model: Any, configured: int) -> int:
    candidates = [
        getattr(model.config, "max_position_embeddings", None),
        getattr(model.config, "n_positions", None),
        getattr(model.config, "max_sequence_length", None),
    ]
    available = [int(value) for value in candidates if isinstance(value, (int, float)) and value > 0]
    return min([configured, *available]) if available else configured


def load_workers(config: InferenceConfig) -> list[InferenceWorker]:
    import torch
    from neucodec import NeuCodec

    config.validate()
    device_names = resolve_devices(config)
    tokenizer_source = config.model
    if config.adapter:
        tokenizer_source = config.base_model or _adapter_base_model(config.adapter) or config.model
    tokenizer = _load_tokenizer(config, tokenizer_source)
    token_ids = validate_tokenizer(tokenizer)
    workers: list[InferenceWorker] = []
    try:
        for worker_id, device_name in enumerate(device_names):
            device = torch.device(device_name)
            dtype = resolve_dtype(config, device)
            LOGGER.info("Loading inference worker %d on %s with %s", worker_id, device, dtype)
            context = torch.cuda.device(device) if device.type == "cuda" else nullcontext()
            with context:
                model, _ = _load_backbone(config, device, dtype)
                codec_kwargs: dict[str, Any] = {}
                if config.codec_revision and not Path(config.codec_model).expanduser().exists():
                    codec_kwargs["revision"] = config.codec_revision
                codec = NeuCodec.from_pretrained(config.codec_model, **codec_kwargs).to(device)
                codec.eval().requires_grad_(False)
            model.config.pad_token_id = int(tokenizer.pad_token_id)
            workers.append(
                InferenceWorker(
                    worker_id=worker_id,
                    device=device,
                    dtype=dtype,
                    model=model,
                    tokenizer=tokenizer,
                    codec=codec,
                    pad_id=int(tokenizer.pad_token_id),
                    context_limit=_context_limit(model, config.max_context),
                    **token_ids,
                )
            )
    except Exception:
        close_workers(workers)
        raise
    return workers


def _resample_reference(path: str | Path, device: Any) -> tuple[Any, float]:
    import soundfile as sf
    import torch
    import torchaudio

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Reference audio not found: {source}")
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    if audio.shape[1] != 1:
        LOGGER.warning("Reference audio has %d channels; averaging to mono", audio.shape[1])
    waveform = torch.from_numpy(audio.T).float().mean(dim=0, keepdim=True)
    duration = waveform.shape[-1] / float(sample_rate)
    if sample_rate != 16_000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
    return waveform.unsqueeze(0).to(device), duration


def encode_reference(
    workers: list[InferenceWorker],
    *,
    reference_audio: str | Path,
    reference_phonemes: str,
    config: InferenceConfig,
) -> list[int]:
    import torch

    if not workers:
        raise RuntimeError("No inference workers are loaded")
    if not reference_phonemes.strip():
        raise ValueError("reference_text produced empty phonemes")
    primary = workers[0]
    if primary.codec_encoder_released:
        raise RuntimeError("Codec encoder was released; reload TTSInference before changing the reference")
    waveform, duration = _resample_reference(reference_audio, primary.device)
    if not config.reference_min_seconds <= duration <= config.reference_max_seconds:
        raise ValueError(
            f"Reference audio duration is {duration:.2f}s; expected "
            f"{config.reference_min_seconds:.1f}-{config.reference_max_seconds:.1f}s"
        )
    context = torch.cuda.device(primary.device) if primary.device.type == "cuda" else nullcontext()
    with context, torch.inference_mode():
        code_tensor = primary.codec.encode_code(waveform)
    codes = [int(value) for value in code_tensor.detach().reshape(-1).cpu().tolist()]
    if not codes or min(codes) < 0 or max(codes) >= SPEECH_CODE_COUNT:
        raise RuntimeError("NeuCodec returned empty or out-of-range reference codes")
    for worker in workers:
        worker.reference_phonemes = reference_phonemes.strip()
        worker.reference_codes = codes
        worker.reference_code_token_ids = [worker.speech_token_min + code for code in codes]
    if config.release_codec_encoder:
        for worker in workers:
            release_codec_encoder(worker)
    LOGGER.info("Encoded %.2fs reference into %d shared NeuCodec tokens", duration, len(codes))
    return codes


def release_codec_encoder(worker: InferenceWorker) -> list[str]:
    import torch

    required_decoder_parts = ("generator", "fc_post_a")
    if not all(hasattr(worker.codec, name) for name in required_decoder_parts):
        LOGGER.warning("NeuCodec layout is unknown; keeping the full codec on %s", worker.device)
        return []
    encoder_parts = (
        "semantic_model",
        "feature_extractor",
        "SemanticEncoder_module",
        "CodecEnc",
        "codec_encoder",
        "fc_prior",
        "fc_sq_prior",
    )
    released: list[str] = []
    for name in encoder_parts:
        if hasattr(worker.codec, name):
            delattr(worker.codec, name)
            released.append(name)
    worker.codec_encoder_released = bool(released)
    gc.collect()
    if worker.device.type == "cuda":
        with torch.cuda.device(worker.device):
            torch.cuda.empty_cache()
    LOGGER.info("Worker %d released NeuCodec encoder parts: %s", worker.worker_id, released)
    return released


def close_workers(workers: list[InferenceWorker]) -> None:
    import torch

    for worker in workers:
        worker.model = None
        worker.codec = None
        worker.tokenizer = None
    workers.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
