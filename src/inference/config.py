from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class InferenceConfig:
    model: str = "pnnbao-ump/VieNeu-TTS"
    model_revision: str | None = None
    adapter: str | None = None
    base_model: str | None = None
    codec_model: str = "neuphonic/neucodec"
    codec_revision: str | None = None
    trust_remote_code: bool = False

    devices: str = "auto"
    num_gpus: int = 1
    dtype: str = "auto"

    min_chars: int = 80
    target_chars: int = 128
    max_chars: int = 156
    batch_size: int = 128
    max_length_gap: int = 12
    max_runtime_batch_size: int | None = None

    max_context: int = 2048
    max_new_tokens: int = 600
    min_new_tokens: int = 50
    do_sample: bool = False
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.1
    cache_implementation: str = "dynamic"
    enable_compile: bool = False
    compile_full_batch_only: bool = True
    compile_dynamic: bool = True
    pad_to_multiple_of: int = 1
    max_retries: int = 2
    release_codec_encoder: bool = True
    show_progress: bool = True

    sentence_pause_ms: int = 200
    question_pause_ms: int = 200
    paragraph_pause_ms: int = 200
    leading_silence_ms: int = 300
    trailing_silence_ms: int = 300
    speed: float = 1.0
    keep_segments: bool = False
    seed: int = 3407

    reference_min_seconds: float = 3.0
    reference_max_seconds: float = 15.0

    def validate(self) -> None:
        if not str(self.model).strip():
            raise ValueError("model cannot be empty")
        if self.adapter and not (self.base_model or self.model):
            raise ValueError("base_model or model is required when adapter is used")
        if self.dtype not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if self.num_gpus <= 0:
            raise ValueError("num_gpus must be positive")
        if not 1 <= self.min_chars <= self.target_chars <= self.max_chars:
            raise ValueError("Expected 1 <= min_chars <= target_chars <= max_chars")
        if self.batch_size <= 0 or self.max_length_gap < 0:
            raise ValueError("batch_size must be positive and max_length_gap cannot be negative")
        if self.max_runtime_batch_size is not None and self.max_runtime_batch_size <= 0:
            raise ValueError("max_runtime_batch_size must be positive when specified")
        if self.max_context <= 0 or self.max_new_tokens <= 0 or self.min_new_tokens < 0:
            raise ValueError("generation token limits are invalid")
        if self.min_new_tokens >= self.max_new_tokens:
            raise ValueError("min_new_tokens must be smaller than max_new_tokens")
        if self.temperature <= 0 or not 0 < self.top_p <= 1 or self.top_k < 0:
            raise ValueError("sampling parameters are invalid")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if self.cache_implementation not in {"dynamic", "static"}:
            raise ValueError("cache_implementation must be dynamic or static")
        if self.pad_to_multiple_of <= 0 or self.max_retries < 0:
            raise ValueError("pad_to_multiple_of must be positive and max_retries cannot be negative")
        for name in (
            "sentence_pause_ms",
            "question_pause_ms",
            "paragraph_pause_ms",
            "leading_silence_ms",
            "trailing_silence_ms",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if not 0.8 <= self.speed <= 1.25:
            raise ValueError("speed must be in [0.8, 1.25]")
        if not 0 < self.reference_min_seconds <= self.reference_max_seconds:
            raise ValueError("reference duration limits are invalid")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "InferenceConfig":
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"Unknown inference config keys: {', '.join(unknown)}")
        config = cls(**raw)
        config.validate()
        return config

    @classmethod
    def from_yaml(cls, path: str | Path) -> "InferenceConfig":
        source = Path(path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Inference config not found: {source}")
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError("Inference config root must be a mapping")
        return cls.from_dict(raw)

    def apply_overrides(self, **overrides: Any) -> None:
        known = {field.name for field in fields(self)}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ValueError(f"Unknown inference overrides: {', '.join(unknown)}")
        for key, value in overrides.items():
            if value is not None:
                setattr(self, key, value)
        self.validate()
