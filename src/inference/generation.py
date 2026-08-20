from __future__ import annotations

import gc
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Lock
from typing import Any, Callable

from .batching import assign_batches_to_workers, split_batches_for_execution
from .config import InferenceConfig
from .modeling import InferenceWorker
from .types import InferenceBatch, PreparedBatches

LOGGER = logging.getLogger(__name__)
_LOG_LOCK = Lock()
ProgressCallback = Callable[[list[dict[str, Any]]], None]


def _left_pad_prompts(worker: InferenceWorker, prompts: list[list[int]], pad_to_multiple_of: int):
    import torch

    max_length = max(len(ids) for ids in prompts)
    max_length = int(math.ceil(max_length / pad_to_multiple_of) * pad_to_multiple_of)
    pin_memory = worker.device.type == "cuda"
    input_ids = torch.full(
        (len(prompts), max_length),
        worker.pad_id,
        dtype=torch.long,
        pin_memory=pin_memory,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, ids in enumerate(prompts):
        row_ids = torch.as_tensor(ids, dtype=torch.long)
        input_ids[row, -row_ids.numel() :] = row_ids
        attention_mask[row, -row_ids.numel() :] = 1
    non_blocking = worker.device.type == "cuda"
    return (
        input_ids.to(worker.device, non_blocking=non_blocking),
        attention_mask.to(worker.device, non_blocking=non_blocking),
    )


def _decode_codes(worker: InferenceWorker, generated_ids: Any, speech_mask: Any):
    import numpy as np
    import torch

    codes = (generated_ids[speech_mask] - worker.speech_token_min).to(dtype=torch.long)
    if codes.numel() == 0:
        raise ValueError("Model generated no valid NeuCodec speech tokens")
    context = torch.cuda.device(worker.device) if worker.device.type == "cuda" else nullcontext()
    with context, torch.inference_mode():
        reconstructed = worker.codec.decode_code(codes[None, None, :])
    if isinstance(reconstructed, torch.Tensor):
        array = reconstructed[0, 0].detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        array = np.asarray(reconstructed[0, 0], dtype=np.float32)
    array = np.asarray(array, dtype=np.float32).reshape(-1)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("NeuCodec decoder returned empty or non-finite audio")
    return array


def _tail_repeat_ratio(values: Any, window: int = 32) -> float:
    data = values.detach().reshape(-1)
    if data.numel() < 2:
        return 0.0
    tail = data[-min(window, data.numel()) :]
    return float((tail[1:] == tail[:-1]).float().mean().item())


def _generation_kwargs(worker: InferenceWorker, config: InferenceConfig, max_new_tokens: int, batch_size: int):
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min(config.min_new_tokens, max(0, max_new_tokens - 1)),
        "eos_token_id": worker.speech_end_id,
        "pad_token_id": worker.pad_id,
        "do_sample": config.do_sample,
        "repetition_penalty": config.repetition_penalty,
        "use_cache": True,
        "cache_implementation": config.cache_implementation,
    }
    compile_batch = config.enable_compile and not worker.compile_disabled and (
        not config.compile_full_batch_only or batch_size == config.batch_size
    )
    kwargs["disable_compile"] = not compile_batch
    if compile_batch:
        from transformers import CompileConfig

        kwargs["compile_config"] = CompileConfig(
            dynamic=config.compile_dynamic,
            mode="reduce-overhead",
            fullgraph=False,
        )
    if config.do_sample:
        kwargs.update({"temperature": config.temperature, "top_p": config.top_p, "top_k": config.top_k})
    return kwargs, compile_batch


def _generate_batch(
    worker: InferenceWorker,
    batch: InferenceBatch,
    segment_directory: Path,
    config: InferenceConfig,
) -> list[dict[str, Any]]:
    import soundfile as sf
    import torch

    prompts = [worker.build_prompt(item["phonemes"]) for item in batch]
    input_ids, attention_mask = _left_pad_prompts(worker, prompts, config.pad_to_multiple_of)
    input_length = int(input_ids.shape[1])
    max_new_tokens = min(config.max_new_tokens, worker.context_limit - input_length)
    if max_new_tokens < 1:
        raise RuntimeError(
            f"Prompt length {input_length} exceeds the worker context limit {worker.context_limit}"
        )
    generation_kwargs, compile_batch = _generation_kwargs(worker, config, max_new_tokens, len(batch))
    context = torch.cuda.device(worker.device) if worker.device.type == "cuda" else nullcontext()
    if worker.device.type == "cuda":
        torch.cuda.synchronize(worker.device)
    started = time.perf_counter()
    with context, torch.inference_mode():
        try:
            outputs = worker.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        except Exception:
            if not compile_batch:
                raise
            LOGGER.warning("Compile failed on %s; retrying this batch in eager mode", worker.device)
            fallback = dict(generation_kwargs)
            fallback["disable_compile"] = True
            fallback.pop("compile_config", None)
            worker.compile_disabled = True
            outputs = worker.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **fallback,
            )
    if worker.device.type == "cuda":
        torch.cuda.synchronize(worker.device)
    elapsed = time.perf_counter() - started
    if len(outputs) != len(batch):
        raise RuntimeError(f"Model returned {len(outputs)} outputs for a batch of {len(batch)}")

    infos: list[dict[str, Any]] = []
    for row, output in enumerate(outputs):
        item = batch[row]
        generated = output[input_length:]
        end_positions = torch.nonzero(generated.eq(worker.speech_end_id), as_tuple=False).flatten()
        hit_end = end_positions.numel() > 0
        if hit_end:
            generated = generated[: int(end_positions[0].item())]
        speech_mask = (generated >= worker.speech_token_min) & (generated <= worker.speech_token_max)
        speech_tokens = int(speech_mask.sum().item())
        base = {
            "index": int(item["index"]),
            "uid": item["uid"],
            "text": item["text"],
            "source_audio_path": item.get("source_audio_path", ""),
            "device": str(worker.device),
            "generated_tokens": int(generated.numel()),
            "speech_tokens": speech_tokens,
            "hit_speech_end": bool(hit_end),
            "tail_repeat_ratio": _tail_repeat_ratio(generated[speech_mask]),
            "batch_inference_seconds": elapsed,
        }
        try:
            audio = _decode_codes(worker, generated, speech_mask)
            segment_path = segment_directory / f"segment_{int(item['index']):06d}.wav"
            sf.write(segment_path, audio, worker.sample_rate, subtype="FLOAT")
            infos.append(
                {
                    **base,
                    "status": "ok",
                    "segment_path": str(segment_path),
                    "audio_seconds": len(audio) / worker.sample_rate,
                }
            )
        except Exception as exc:  # item-level failure is retried later
            infos.append({**base, "status": "failed", "error": repr(exc), "audio_seconds": 0.0})
    return infos


def _is_out_of_memory(error: BaseException) -> bool:
    return "out of memory" in str(error).casefold() or error.__class__.__name__ == "OutOfMemoryError"


def _clear_after_oom(worker: InferenceWorker) -> None:
    import torch

    gc.collect()
    if worker.device.type == "cuda":
        with torch.cuda.device(worker.device):
            torch.cuda.empty_cache()


def _infer_with_oom_split(
    worker: InferenceWorker,
    batch: InferenceBatch,
    segment_directory: Path,
    config: InferenceConfig,
) -> list[dict[str, Any]]:
    try:
        return _generate_batch(worker, batch, segment_directory, config)
    except Exception as exc:
        if _is_out_of_memory(exc) and len(batch) > 1:
            _clear_after_oom(worker)
            middle = len(batch) // 2
            with _LOG_LOCK:
                LOGGER.warning(
                    "OOM on %s: splitting batch %d into %d + %d",
                    worker.device,
                    len(batch),
                    middle,
                    len(batch) - middle,
                )
            return [
                *_infer_with_oom_split(worker, batch[:middle], segment_directory, config),
                *_infer_with_oom_split(worker, batch[middle:], segment_directory, config),
            ]
        _clear_after_oom(worker)
        return [
            {
                "index": int(item["index"]),
                "uid": item["uid"],
                "text": item["text"],
                "source_audio_path": item.get("source_audio_path", ""),
                "device": str(worker.device),
                "status": "failed",
                "error": repr(exc),
                "audio_seconds": 0.0,
                "speech_tokens": 0,
                "generated_tokens": 0,
            }
            for item in batch
        ]


def _run_worker_queue(
    worker: InferenceWorker,
    queue: list[tuple[int, InferenceBatch]],
    segment_directory: Path,
    config: InferenceConfig,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for _, batch in queue:
        batch_results = _infer_with_oom_split(worker, batch, segment_directory, config)
        results.extend(batch_results)
        if progress_callback is not None:
            progress_callback(batch_results)
        else:
            with _LOG_LOCK:
                success = sum(info["status"] == "ok" for info in batch_results)
                LOGGER.info("%s completed %d/%d segments", worker.device, success, len(batch_results))
    return results


def _parallel_round(
    workers: list[InferenceWorker],
    batches: PreparedBatches,
    segment_directory: Path,
    config: InferenceConfig,
    progress_callback: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    assignments, loads = assign_batches_to_workers(batches, len(workers))
    if progress_callback is None:
        LOGGER.info("Inference batch cost by worker: %s", loads)
    if len(workers) == 1:
        return _run_worker_queue(
            workers[0],
            assignments[0],
            segment_directory,
            config,
            progress_callback,
        )
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(workers), thread_name_prefix="tts-gpu") as executor:
        futures = [
            executor.submit(
                _run_worker_queue,
                worker,
                assignments[index],
                segment_directory,
                config,
                progress_callback,
            )
            for index, worker in enumerate(workers)
            if assignments[index]
        ]
        for future in futures:
            results.extend(future.result())
    return results


def run_generation(
    workers: list[InferenceWorker],
    batches: PreparedBatches,
    segment_directory: str | Path,
    config: InferenceConfig,
) -> tuple[list[dict[str, Any]], float]:
    if not workers:
        raise RuntimeError("No inference workers are loaded")
    output = Path(segment_directory)
    output.mkdir(parents=True, exist_ok=True)
    runtime_limit = config.max_runtime_batch_size
    if runtime_limit is None and workers[0].device.type != "cuda":
        runtime_limit = 1
    execution_batches = split_batches_for_execution(
        batches,
        worker_count=len(workers),
        max_runtime_batch_size=runtime_limit,
    )
    original_items = {int(item["index"]): item for batch in batches for item in batch}
    progress = None
    progress_lock = Lock()
    completed_indices: set[int] = set()

    def update_progress(batch_results: list[dict[str, Any]]) -> None:
        if progress is None:
            return
        with progress_lock:
            newly_completed = {
                int(info["index"])
                for info in batch_results
                if info.get("status") == "ok" and int(info["index"]) not in completed_indices
            }
            completed_indices.update(newly_completed)
            if newly_completed:
                progress.update(len(newly_completed))
            failed = sum(info.get("status") != "ok" for info in batch_results)
            progress.set_postfix(
                ok=len(completed_indices),
                pending=len(original_items) - len(completed_indices),
                failed=failed,
                refresh=False,
            )

    if config.show_progress:
        from tqdm.auto import tqdm

        progress = tqdm(
            total=len(original_items),
            desc="TTS inference",
            unit="segment",
            dynamic_ncols=True,
            mininterval=0.5,
            leave=True,
        )
    started = time.perf_counter()
    progress_callback = update_progress if progress is not None else None
    try:
        results = _parallel_round(
            workers,
            execution_batches,
            output,
            config,
            progress_callback,
        )
        by_index = {int(info["index"]): info for info in results if info["status"] == "ok"}
        failures = {int(info["index"]): info for info in results if info["status"] != "ok"}

        for retry in range(1, config.max_retries + 1):
            pending = sorted(set(original_items) - set(by_index))
            if not pending:
                break
            LOGGER.warning("Retry %d/%d for %d failed segments", retry, config.max_retries, len(pending))
            retry_batches: PreparedBatches = [[original_items[index]] for index in pending]
            retry_results = _parallel_round(
                workers,
                retry_batches,
                output,
                config,
                progress_callback,
            )
            for info in retry_results:
                index = int(info["index"])
                if info["status"] == "ok":
                    info["retry"] = retry
                    by_index[index] = info
                    failures.pop(index, None)
                else:
                    failures[index] = info

        unresolved = sorted(set(original_items) - set(by_index))
        if unresolved:
            details = "; ".join(
                f"{index}: {failures.get(index, {}).get('error', 'unknown error')}"
                for index in unresolved[:8]
            )
            raise RuntimeError(
                f"Inference failed for {len(unresolved)} segment(s); final WAV was not created. {details}"
            )
        metadata = [by_index[index] for index in sorted(by_index)]
        return metadata, time.perf_counter() - started
    finally:
        if progress is not None:
            progress.close()
