from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, TypedDict


class InferenceItem(TypedDict):
    index: int
    uid: str
    text: str
    phonemes: str
    source_audio_path: str


InferenceBatch = list[InferenceItem]
PreparedBatches = list[InferenceBatch]


@dataclass(slots=True)
class InferenceResult:
    output_path: str
    sample_rate: int
    segments: int
    duration_seconds: float
    inference_seconds: float
    rtf: float
    devices: list[str]
    metadata: list[dict[str, Any]]
    temporary_segments_kept: bool = False
    temporary_segment_directory: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
