from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class RunConfig:
    name: str = "ngoc-a10-v1"
    seed: int = 3407
    output_root: str = "./outputs"
    resume: bool = True
    cleanup_runtime_checkpoints: bool = True


@dataclass(slots=True)
class ModelConfig:
    base_model: str = "pnnbao-ump/VieNeu-TTS"
    model_revision: str | None = None
    codec_model: str = "neuphonic/neucodec"
    codec_revision: str | None = None
    dtype: str = "bfloat16"
    max_sequence_length: int = 2048
    trust_remote_code: bool = False


@dataclass(slots=True)
class DataConfig:
    dataset_root: str = "./data"
    audio_directory: str = "audio"
    train_metadata: str = "metadata.csv"
    eval_metadata: str = "metadata_eval.csv"
    codec_input_sample_rate: int = 16000
    fail_on_data_error: bool = True
    rebuild_encoded_cache: bool = False
    encoded_commit_interval: int = 250
    dataloader_num_workers: int = 4
    pad_to_multiple_of: int = 8


@dataclass(slots=True)
class TrainingPhaseConfig:
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    early_stopping_patience: int = 5
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = False


@dataclass(slots=True)
class Phase1Config(TrainingPhaseConfig):
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass(slots=True)
class Phase2Config:
    safe_merge: bool = True
    verify_merge: bool = True
    verification_tokens: int = 128
    max_abs_logit_difference: float = 0.5


@dataclass(slots=True)
class Phase3Config(TrainingPhaseConfig):
    unfreeze_mode: str = "upper_blocks_and_output_head"
    upper_blocks_ratio: float = 0.33


@dataclass(slots=True)
class ArtifactConfig:
    keep_phase1_final: bool = True
    keep_phase2_final: bool = True
    keep_phase3_final: bool = True
    max_shard_size: str = "5GB"


@dataclass(slots=True)
class PipelineConfig:
    run: RunConfig
    model: ModelConfig
    data: DataConfig
    phase1_lora: Phase1Config
    phase2_merge: Phase2Config
    phase3_finetune: Phase3Config
    artifacts: ArtifactConfig

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PipelineConfig":
        required = {"run", "model", "data", "phase1_lora", "phase2_merge", "phase3_finetune", "artifacts"}
        missing = sorted(required - raw.keys())
        unknown = sorted(raw.keys() - required)
        if missing:
            raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
        if unknown:
            raise ValueError(f"Unknown configuration sections: {', '.join(unknown)}")
        config = cls(
            run=RunConfig(**raw["run"]),
            model=ModelConfig(**raw["model"]),
            data=DataConfig(**raw["data"]),
            phase1_lora=Phase1Config(**raw["phase1_lora"]),
            phase2_merge=Phase2Config(**raw["phase2_merge"]),
            phase3_finetune=Phase3Config(**raw["phase3_finetune"]),
            artifacts=ArtifactConfig(**raw["artifacts"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if not self.run.name.strip():
            raise ValueError("run.name cannot be empty")
        if Path(self.run.name).name != self.run.name or self.run.name in {".", ".."}:
            raise ValueError("run.name must be a single safe path component")
        if self.model.max_sequence_length < 256:
            raise ValueError("model.max_sequence_length is unexpectedly small")
        if self.model.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError(f"Unsupported model dtype: {self.model.dtype}")
        if self.data.codec_input_sample_rate <= 0:
            raise ValueError("codec_input_sample_rate must be positive")
        if self.data.pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive")
        if self.data.encoded_commit_interval <= 0:
            raise ValueError("encoded_commit_interval must be positive")
        self._validate_training_phase("phase1_lora", self.phase1_lora)
        self._validate_training_phase("phase3_finetune", self.phase3_finetune)
        if self.phase1_lora.lora_rank <= 0 or self.phase1_lora.lora_alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not self.phase1_lora.lora_target_modules:
            raise ValueError("At least one LoRA target module is required")
        if self.phase2_merge.verification_tokens <= 0:
            raise ValueError("phase2_merge.verification_tokens must be positive")
        if self.phase2_merge.max_abs_logit_difference <= 0:
            raise ValueError("phase2_merge.max_abs_logit_difference must be positive")
        if not 0 < self.phase3_finetune.upper_blocks_ratio <= 1:
            raise ValueError("phase3_finetune.upper_blocks_ratio must be in (0, 1]")
        if self.phase3_finetune.unfreeze_mode not in {"upper_blocks_and_output_head", "full"}:
            raise ValueError("phase3_finetune.unfreeze_mode must be upper_blocks_and_output_head or full")
        if not self.artifacts.keep_phase3_final:
            raise ValueError("artifacts.keep_phase3_final must be true because it is the pipeline output")

    @staticmethod
    def _validate_training_phase(name: str, phase: TrainingPhaseConfig) -> None:
        if phase.learning_rate <= 0:
            raise ValueError(f"{name}.learning_rate must be positive")
        if phase.num_train_epochs <= 0:
            raise ValueError(f"{name}.num_train_epochs must be positive")
        if phase.per_device_train_batch_size <= 0 or phase.per_device_eval_batch_size <= 0:
            raise ValueError(f"{name} batch sizes must be positive")
        if phase.gradient_accumulation_steps <= 0:
            raise ValueError(f"{name}.gradient_accumulation_steps must be positive")
        if not 0 <= phase.warmup_ratio < 1:
            raise ValueError(f"{name}.warmup_ratio must be in [0, 1)")
        if phase.eval_steps <= 0 or phase.save_steps <= 0 or phase.logging_steps <= 0:
            raise ValueError(f"{name} logging/eval/save steps must be positive")
        if phase.save_steps % phase.eval_steps != 0:
            raise ValueError(
                f"{name}.save_steps must be a multiple of eval_steps when load_best_model_at_end is enabled"
            )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def signature(self) -> str:
        payload = json.dumps(self.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def output_root(self) -> Path:
        return Path(self.run.output_root).expanduser().resolve()

    @property
    def run_root(self) -> Path:
        return self.output_root / "runs" / self.run.name

    @property
    def dataset_root(self) -> Path:
        return Path(self.data.dataset_root).expanduser().resolve()

    def apply_overrides(
        self,
        *,
        run_name: str | None = None,
        dataset_root: str | None = None,
        output_root: str | None = None,
        phase1_learning_rate: float | None = None,
        phase1_epochs: float | None = None,
        phase3_learning_rate: float | None = None,
        phase3_epochs: float | None = None,
    ) -> None:
        if run_name is not None:
            self.run.name = run_name
        if dataset_root is not None:
            self.data.dataset_root = dataset_root
        if output_root is not None:
            self.run.output_root = output_root
        if phase1_learning_rate is not None:
            self.phase1_lora.learning_rate = phase1_learning_rate
        if phase1_epochs is not None:
            self.phase1_lora.num_train_epochs = phase1_epochs
        if phase3_learning_rate is not None:
            self.phase3_finetune.learning_rate = phase3_learning_rate
        if phase3_epochs is not None:
            self.phase3_finetune.num_train_epochs = phase3_epochs
        self.validate()


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return PipelineConfig.from_dict(raw)


def dump_config(config: PipelineConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config.as_dict(), handle, allow_unicode=True, sort_keys=False)
    temporary.replace(target)
