from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable

from .artifacts import atomic_write_json, phase_final_dir, safe_rmtree, utc_now
from .config import PipelineConfig, TrainingPhaseConfig
from .data import DynamicTTSCollator, EncodedTTSDataset
from .modeling import (
    apply_lora,
    assert_finite_parameters,
    configure_phase3_trainable_parameters,
    gpu_report,
    load_model,
    load_tokenizer,
    parameter_counts,
    save_full_model,
)

LOGGER = logging.getLogger(__name__)
CommitCallback = Callable[[], None]


def _training_arguments(
    *,
    phase: TrainingPhaseConfig,
    output_dir: Path,
    run_name: str,
    seed: int,
    dataloader_num_workers: int,
):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        num_train_epochs=phase.num_train_epochs,
        per_device_train_batch_size=phase.per_device_train_batch_size,
        per_device_eval_batch_size=phase.per_device_eval_batch_size,
        gradient_accumulation_steps=phase.gradient_accumulation_steps,
        learning_rate=phase.learning_rate,
        warmup_ratio=phase.warmup_ratio,
        weight_decay=phase.weight_decay,
        max_grad_norm=phase.max_grad_norm,
        lr_scheduler_type=phase.lr_scheduler_type,
        logging_strategy="steps",
        logging_steps=phase.logging_steps,
        eval_strategy="steps",
        eval_steps=phase.eval_steps,
        save_strategy="steps",
        save_steps=phase.save_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=phase.bf16,
        fp16=False,
        tf32=phase.tf32,
        gradient_checkpointing=phase.gradient_checkpointing,
        optim="adamw_torch_fused",
        report_to="none",
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=True,
        group_by_length=True,
        prediction_loss_only=True,
        save_safetensors=True,
        seed=seed,
        data_seed=seed,
    )


def _datasets(config: PipelineConfig, encoded_paths: dict[str, Path], tokenizer: Any):
    train_dataset = EncodedTTSDataset(
        encoded_paths["train"], tokenizer, config.model.max_sequence_length
    )
    eval_dataset = EncodedTTSDataset(
        encoded_paths["eval"], tokenizer, config.model.max_sequence_length
    )
    collator = DynamicTTSCollator(
        tokenizer.pad_token_id,
        pad_to_multiple_of=config.data.pad_to_multiple_of,
    )
    return train_dataset, eval_dataset, collator


def _latest_checkpoint(runtime_dir: Path, resume: bool) -> str | None:
    if not resume or not runtime_dir.is_dir():
        return None
    from transformers.trainer_utils import get_last_checkpoint

    return get_last_checkpoint(str(runtime_dir))


def _commit_trainer_callback(commit_callback: CommitCallback | None):
    from transformers import TrainerCallback

    class CommitOnSaveCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):  # noqa: ANN001, ANN003
            if commit_callback is not None:
                LOGGER.info("Committing runtime checkpoint at global step %s", state.global_step)
                commit_callback()
            return control

    return CommitOnSaveCallback()


def _early_stopping_callback(patience: int):
    from transformers import EarlyStoppingCallback

    return EarlyStoppingCallback(early_stopping_patience=patience)


def _json_safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metrics.items():
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


def _prepare_final_directory(final_dir: Path, run_root: Path) -> None:
    if final_dir.exists():
        safe_rmtree(final_dir, run_root)
    final_dir.mkdir(parents=True, exist_ok=True)


def train_phase1_lora(
    config: PipelineConfig,
    encoded_paths: dict[str, Path],
    commit_callback: CommitCallback | None = None,
) -> tuple[Path, dict[str, Any]]:
    from transformers import Trainer

    phase_name = "phase1_lora"
    runtime_dir = config.run_root / ".runtime" / phase_name
    final_dir = phase_final_dir(config.run_root, phase_name)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    _prepare_final_directory(final_dir, config.run_root)

    tokenizer = load_tokenizer(config.model.base_model, config.model)
    train_dataset, eval_dataset, collator = _datasets(config, encoded_paths, tokenizer)
    model = load_model(config.model.base_model, config.model)
    model.config.use_cache = False
    model = apply_lora(model, config.phase1_lora)

    args = _training_arguments(
        phase=config.phase1_lora,
        output_dir=runtime_dir,
        run_name=f"{config.run.name}-{phase_name}",
        seed=config.run.seed,
        dataloader_num_workers=config.data.dataloader_num_workers,
    )
    callbacks = [
        _early_stopping_callback(config.phase1_lora.early_stopping_patience),
        _commit_trainer_callback(commit_callback),
    ]
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
        processing_class=tokenizer,
    )
    checkpoint = _latest_checkpoint(runtime_dir, config.run.resume)
    LOGGER.info("Starting phase 1 LoRA; resume checkpoint: %s", checkpoint)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    eval_metrics = trainer.evaluate()
    _assert_finite_eval_loss(eval_metrics)
    assert_finite_parameters(model)

    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    trainable, total = parameter_counts(model)
    report = {
        "completed_at": utc_now(),
        "artifact_type": "peft_lora_adapter",
        "base_model": config.model.base_model,
        "base_model_revision": config.model.model_revision,
        "resumed_from": checkpoint,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "effective_batch_size": (
            config.phase1_lora.per_device_train_batch_size
            * config.phase1_lora.gradient_accumulation_steps
        ),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": 100 * trainable / total,
        "best_metric": trainer.state.best_metric,
        "best_global_step": _best_global_step(trainer.state.best_model_checkpoint),
        "train_metrics": _json_safe_metrics(train_result.metrics),
        "eval_metrics": _json_safe_metrics(eval_metrics),
        "gpu": gpu_report(),
    }
    atomic_write_json(final_dir / "phase_report.json", report)

    del trainer, model, train_dataset, eval_dataset, tokenizer
    _release_cuda()
    return final_dir, report


def merge_phase2(
    config: PipelineConfig,
    encoded_paths: dict[str, Path],
) -> tuple[Path, dict[str, Any]]:
    import torch
    from peft import PeftModel

    phase_name = "phase2_merge"
    adapter_dir = phase_final_dir(config.run_root, "phase1_lora")
    final_dir = phase_final_dir(config.run_root, phase_name)
    if not (adapter_dir / "COMPLETE.json").is_file():
        raise FileNotFoundError(f"Completed phase 1 adapter not found: {adapter_dir}")
    _prepare_final_directory(final_dir, config.run_root)

    tokenizer = load_tokenizer(config.model.base_model, config.model)
    base_model = load_model(config.model.base_model, config.model)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
    peft_model.eval()

    max_abs_logit_difference: float | None = None
    verification_shape: list[int] | None = None
    verification_input = None
    logits_before = None
    if config.phase2_merge.verify_merge:
        eval_dataset = EncodedTTSDataset(
            encoded_paths["eval"], tokenizer, config.model.max_sequence_length
        )
        verification_input = eval_dataset[0]["input_ids"][: config.phase2_merge.verification_tokens]
        verification_input = verification_input.unsqueeze(0).to("cuda")
        with torch.inference_mode():
            logits_before = peft_model(input_ids=verification_input).logits[:, -1, :].float().cpu()
        if not torch.isfinite(logits_before).all().item():
            raise FloatingPointError("Non-finite logits detected before LoRA merge")
        verification_shape = list(logits_before.shape)
        del eval_dataset

    LOGGER.info("Merging LoRA adapter into the full-precision base model")
    merged_model = peft_model.merge_and_unload(safe_merge=config.phase2_merge.safe_merge)
    merged_model.eval()
    assert_finite_parameters(merged_model)

    if verification_input is not None and logits_before is not None:
        with torch.inference_mode():
            logits_after = merged_model(input_ids=verification_input).logits[:, -1, :].float().cpu()
        max_abs_logit_difference = float((logits_before - logits_after).abs().max().item())
        if not torch.isfinite(logits_after).all().item():
            raise FloatingPointError("Non-finite logits detected after LoRA merge")
        if max_abs_logit_difference > config.phase2_merge.max_abs_logit_difference:
            raise RuntimeError(
                "LoRA merge verification exceeded the configured tolerance: "
                f"{max_abs_logit_difference:.6f} > "
                f"{config.phase2_merge.max_abs_logit_difference:.6f}"
            )
        del logits_after, logits_before, verification_input

    save_full_model(merged_model, tokenizer, final_dir, config.artifacts.max_shard_size)
    _, total = parameter_counts(merged_model)
    report = {
        "completed_at": utc_now(),
        "artifact_type": "merged_full_model",
        "base_model": config.model.base_model,
        "base_model_revision": config.model.model_revision,
        "adapter": str(adapter_dir),
        "dtype": config.model.dtype,
        "safe_merge": config.phase2_merge.safe_merge,
        "merge_verified": config.phase2_merge.verify_merge,
        "verification_logit_shape": verification_shape,
        "max_abs_logit_difference": max_abs_logit_difference,
        "max_abs_logit_difference_tolerance": config.phase2_merge.max_abs_logit_difference,
        "total_parameters": total,
        "gpu": gpu_report(),
    }
    atomic_write_json(final_dir / "phase_report.json", report)

    del merged_model, peft_model, base_model, tokenizer
    _release_cuda()
    return final_dir, report


def train_phase3_finetune(
    config: PipelineConfig,
    encoded_paths: dict[str, Path],
    commit_callback: CommitCallback | None = None,
) -> tuple[Path, dict[str, Any]]:
    from transformers import Trainer

    phase_name = "phase3_finetune"
    merged_dir = phase_final_dir(config.run_root, "phase2_merge")
    runtime_dir = config.run_root / ".runtime" / phase_name
    final_dir = phase_final_dir(config.run_root, phase_name)
    if not (merged_dir / "COMPLETE.json").is_file():
        raise FileNotFoundError(f"Completed merged model not found: {merged_dir}")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    _prepare_final_directory(final_dir, config.run_root)

    tokenizer = load_tokenizer(merged_dir, config.model)
    train_dataset, eval_dataset, collator = _datasets(config, encoded_paths, tokenizer)
    model = load_model(merged_dir, config.model)
    model.config.use_cache = False
    unfreeze_report = configure_phase3_trainable_parameters(model, config.phase3_finetune)
    if config.phase3_finetune.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    args = _training_arguments(
        phase=config.phase3_finetune,
        output_dir=runtime_dir,
        run_name=f"{config.run.name}-{phase_name}",
        seed=config.run.seed,
        dataloader_num_workers=config.data.dataloader_num_workers,
    )
    callbacks = [
        _early_stopping_callback(config.phase3_finetune.early_stopping_patience),
        _commit_trainer_callback(commit_callback),
    ]
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=callbacks,
        processing_class=tokenizer,
    )
    checkpoint = _latest_checkpoint(runtime_dir, config.run.resume)
    LOGGER.info("Starting phase 3 partial fine-tune; resume checkpoint: %s", checkpoint)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    eval_metrics = trainer.evaluate()
    _assert_finite_eval_loss(eval_metrics)

    model.config.use_cache = True
    assert_finite_parameters(model)
    save_full_model(model, tokenizer, final_dir, config.artifacts.max_shard_size)
    report = {
        "completed_at": utc_now(),
        "artifact_type": "partial_finetuned_full_model",
        "source_merged_model": str(merged_dir),
        "resumed_from": checkpoint,
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset),
        "effective_batch_size": (
            config.phase3_finetune.per_device_train_batch_size
            * config.phase3_finetune.gradient_accumulation_steps
        ),
        "unfreeze": unfreeze_report,
        "best_metric": trainer.state.best_metric,
        "best_global_step": _best_global_step(trainer.state.best_model_checkpoint),
        "train_metrics": _json_safe_metrics(train_result.metrics),
        "eval_metrics": _json_safe_metrics(eval_metrics),
        "gpu": gpu_report(),
    }
    atomic_write_json(final_dir / "phase_report.json", report)

    del trainer, model, train_dataset, eval_dataset, tokenizer
    _release_cuda()
    return final_dir, report


def _best_global_step(best_checkpoint: str | None) -> int | None:
    if not best_checkpoint:
        return None
    name = Path(best_checkpoint).name
    if not name.startswith("checkpoint-"):
        return None
    try:
        return int(name.removeprefix("checkpoint-"))
    except ValueError:
        return None


def _assert_finite_eval_loss(metrics: dict[str, Any]) -> None:
    value = metrics.get("eval_loss")
    if value is None or not math.isfinite(float(value)):
        raise FloatingPointError(f"Evaluation did not produce a finite eval_loss: {value!r}")


def _release_cuda() -> None:
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def read_phase_report(final_dir: Path) -> dict[str, Any]:
    with (final_dir / "phase_report.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)
