from __future__ import annotations

import gc
import logging
import math
from pathlib import Path
from typing import Any

from .config import ModelConfig, Phase1Config, Phase3Config
from .data import _validate_special_tokens

LOGGER = logging.getLogger(__name__)


def torch_dtype(name: str):
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {name}") from exc


def load_tokenizer(model_name_or_path: str | Path, model_config: ModelConfig):
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"trust_remote_code": model_config.trust_remote_code}
    if model_config.model_revision and not Path(str(model_name_or_path)).exists():
        kwargs["revision"] = model_config.model_revision
    tokenizer = AutoTokenizer.from_pretrained(str(model_name_or_path), **kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    _validate_special_tokens(tokenizer)
    return tokenizer


def load_model(model_name_or_path: str | Path, model_config: ModelConfig, *, device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "dtype": torch_dtype(model_config.dtype),
        "low_cpu_mem_usage": True,
        "trust_remote_code": model_config.trust_remote_code,
        "attn_implementation": "sdpa",
    }
    if model_config.model_revision and not Path(str(model_name_or_path)).exists():
        kwargs["revision"] = model_config.model_revision
    model = AutoModelForCausalLM.from_pretrained(str(model_name_or_path), **kwargs)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for training but is not available")
        model.to(torch.device("cuda"))
    elif device == "cpu":
        model.to(torch.device("cpu"))
    else:
        raise ValueError(f"Unsupported device: {device}")
    return model


def apply_lora(model: Any, config: Phase1Config):
    from peft import LoraConfig, TaskType, get_peft_model

    available_suffixes = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing = [name for name in config.lora_target_modules if name not in available_suffixes]
    if missing:
        raise ValueError(f"LoRA target modules are missing from the base model: {missing}")
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_config)
    trainable, total = parameter_counts(peft_model)
    LOGGER.info(
        "LoRA trainable parameters: %s / %s (%.4f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )
    return peft_model


def configure_phase3_trainable_parameters(model: Any, config: Phase3Config) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    if config.unfreeze_mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
        trainable, total = parameter_counts(model)
        return {
            "mode": "full",
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_percent": 100 * trainable / total,
        }

    layers = _find_transformer_layers(model)
    number_to_unfreeze = max(1, math.ceil(len(layers) * config.upper_blocks_ratio))
    first_unfrozen_index = len(layers) - number_to_unfreeze
    for layer in layers[first_unfrozen_index:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    extra_modules: list[str] = []
    for name in ("lm_head", "model.norm", "transformer.ln_f", "model.final_layernorm"):
        module = _resolve_module(model, name)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad = True
            extra_modules.append(name)

    trainable, total = parameter_counts(model)
    if trainable == 0:
        raise RuntimeError("Phase 3 configured zero trainable parameters")
    report = {
        "mode": config.unfreeze_mode,
        "total_transformer_blocks": len(layers),
        "unfrozen_transformer_blocks": number_to_unfreeze,
        "first_unfrozen_block": first_unfrozen_index,
        "extra_modules": extra_modules,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": 100 * trainable / total,
    }
    LOGGER.info("Phase 3 unfreeze report: %s", report)
    return report


def _find_transformer_layers(model: Any):
    candidates = (
        "model.layers",
        "transformer.h",
        "model.decoder.layers",
        "layers",
    )
    for path in candidates:
        module = _resolve_module(model, path)
        if module is not None and hasattr(module, "__len__") and len(module) > 0:
            return module
    raise ValueError("Could not locate transformer blocks in the merged model")


def _resolve_module(root: Any, dotted_path: str):
    current = root
    for component in dotted_path.split("."):
        if not hasattr(current, component):
            return None
        current = getattr(current, component)
    return current


def parameter_counts(model: Any) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total


def assert_finite_parameters(model: Any) -> None:
    import torch

    for name, parameter in model.named_parameters():
        if parameter.numel() and not torch.isfinite(parameter.detach()).all().item():
            raise FloatingPointError(f"Non-finite parameter detected: {name}")


def save_full_model(model: Any, tokenizer: Any, output_dir: Path, max_shard_size: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)


def release_model(*objects: Any) -> None:
    import torch

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def gpu_report() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        return {"cuda_available": False}
    properties = torch.cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "device": torch.cuda.get_device_name(0),
        "total_memory_gib": round(properties.total_memory / 1024**3, 3),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }

