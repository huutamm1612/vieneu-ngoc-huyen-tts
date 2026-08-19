from __future__ import annotations

from collections.abc import Iterable

from .types import InferenceBatch, InferenceItem, PreparedBatches


def phoneme_length(item: InferenceItem) -> int:
    return len(str(item["phonemes"]).strip())


def build_length_batches(
    items: Iterable[InferenceItem],
    *,
    batch_size: int = 128,
    max_length_gap: int = 12,
) -> PreparedBatches:
    if batch_size <= 0 or max_length_gap < 0:
        raise ValueError("batch_size must be positive and max_length_gap cannot be negative")
    sorted_items = sorted(items, key=lambda item: (phoneme_length(item), item["index"]))
    batches: PreparedBatches = []
    current: InferenceBatch = []
    for item in sorted_items:
        if not current:
            current = [item]
            continue
        gap = phoneme_length(item) - phoneme_length(current[0])
        if len(current) >= batch_size or gap > max_length_gap:
            batches.append(current)
            current = [item]
        else:
            current.append(item)
    if current:
        batches.append(current)
    return batches


def batch_cost(batch: InferenceBatch) -> int:
    if not batch:
        return 0
    return len(batch) * max(phoneme_length(item) for item in batch)


def split_batches_for_execution(
    batches: PreparedBatches,
    *,
    worker_count: int,
    max_runtime_batch_size: int | None,
) -> PreparedBatches:
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    execution: PreparedBatches = []
    for batch in batches:
        if not batch:
            continue
        limit = max_runtime_batch_size or len(batch)
        execution.extend(batch[start : start + limit] for start in range(0, len(batch), limit))

    while len(execution) < worker_count:
        candidates = [(len(batch), index) for index, batch in enumerate(execution) if len(batch) > 1]
        if not candidates:
            break
        _, index = max(candidates)
        batch = execution.pop(index)
        middle = len(batch) // 2
        execution[index:index] = [batch[:middle], batch[middle:]]
    return execution


def assign_batches_to_workers(
    batches: PreparedBatches,
    worker_count: int,
) -> tuple[list[list[tuple[int, InferenceBatch]]], list[int]]:
    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    assignments: list[list[tuple[int, InferenceBatch]]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    ordered = sorted(enumerate(batches), key=lambda pair: batch_cost(pair[1]), reverse=True)
    for batch_id, batch in ordered:
        worker_id = min(range(worker_count), key=lambda index: loads[index])
        assignments[worker_id].append((batch_id, batch))
        loads[worker_id] += batch_cost(batch)
    for queue in assignments:
        queue.sort(key=lambda pair: batch_cost(pair[1]), reverse=True)
    return assignments, loads


def validate_batches(batches: PreparedBatches) -> PreparedBatches:
    if not isinstance(batches, list) or not batches:
        raise ValueError("batches must be a non-empty list of non-empty batches")
    validated: PreparedBatches = []
    seen_indices: set[int] = set()
    required = {"index", "uid", "text", "phonemes"}
    for batch_number, batch in enumerate(batches):
        if not isinstance(batch, list) or not batch:
            raise ValueError(f"Batch {batch_number} is empty or is not a list")
        validated_batch: InferenceBatch = []
        for item in batch:
            if not isinstance(item, dict):
                raise ValueError(f"Batch {batch_number} contains a non-mapping item")
            missing = required - item.keys()
            if missing:
                raise ValueError(f"Inference item is missing fields: {sorted(missing)}")
            index = int(item["index"])
            if index in seen_indices:
                raise ValueError(f"Duplicate inference index: {index}")
            seen_indices.add(index)
            text = str(item["text"]).strip()
            phonemes = str(item["phonemes"]).strip()
            if not text or not phonemes:
                raise ValueError(f"Inference item {index} has empty text or phonemes")
            validated_batch.append(
                {
                    "index": index,
                    "uid": str(item["uid"]),
                    "text": text,
                    "phonemes": phonemes,
                    "source_audio_path": str(item.get("source_audio_path", "")),
                }
            )
        validated.append(validated_batch)
    return validated
