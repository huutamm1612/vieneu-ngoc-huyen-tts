from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from .config import InferenceConfig


def _silence_samples(milliseconds: int, sample_rate: int) -> int:
    return round(milliseconds * sample_rate / 1000)


def _pause_after(text: str, config: InferenceConfig) -> int:
    stripped = text.rstrip()
    if stripped.endswith("?"):
        return config.question_pause_ms
    return config.sentence_pause_ms


def assemble_final_wav(
    metadata: list[dict[str, Any]],
    output_path: str | Path,
    config: InferenceConfig,
) -> tuple[int, float, list[dict[str, Any]]]:
    import numpy as np
    import soundfile as sf

    if not metadata:
        raise ValueError("No generated segments are available for WAV assembly")
    destination = Path(output_path).expanduser().resolve()
    if destination.suffix.casefold() != ".wav":
        raise ValueError("output_path must end with .wav")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{uuid.uuid4().hex}.partial.wav")
    ordered = sorted(metadata, key=lambda item: int(item["index"]))
    sample_rate: int | None = None
    timeline: list[dict[str, Any]] = []
    cursor = 0
    try:
        with sf.SoundFile(
            temporary,
            mode="w",
            samplerate=24_000,
            channels=1,
            subtype="PCM_16",
            format="WAV",
        ) as output:
            sample_rate = int(output.samplerate)
            leading = _silence_samples(config.leading_silence_ms, sample_rate)
            if leading:
                output.write(np.zeros(leading, dtype=np.float32))
                cursor += leading
            for position, info in enumerate(ordered):
                segment_path = Path(info["segment_path"])
                if not segment_path.is_file():
                    raise FileNotFoundError(f"Generated segment is missing: {segment_path}")
                audio, segment_rate = sf.read(segment_path, dtype="float32", always_2d=False)
                if int(segment_rate) != sample_rate:
                    raise ValueError(
                        f"Segment {segment_path} has sample rate {segment_rate}; expected {sample_rate}"
                    )
                audio = np.asarray(audio, dtype=np.float32)
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                audio = audio.reshape(-1)
                if config.speed != 1.0:
                    try:
                        import librosa
                    except ImportError as exc:
                        raise RuntimeError("librosa is required when speed is not 1.0") from exc
                    audio = librosa.effects.time_stretch(audio, rate=config.speed)
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
                audio = np.clip(audio, -1.0, 1.0)
                start_sample = cursor
                output.write(audio)
                cursor += len(audio)
                timeline_info = dict(info)
                timeline_info.update(
                    {
                        "start_seconds": start_sample / sample_rate,
                        "end_seconds": cursor / sample_rate,
                    }
                )
                timeline.append(timeline_info)
                if position + 1 < len(ordered):
                    pause = _silence_samples(_pause_after(str(info["text"]), config), sample_rate)
                    if pause:
                        output.write(np.zeros(pause, dtype=np.float32))
                        cursor += pause
            trailing = _silence_samples(config.trailing_silence_ms, sample_rate)
            if trailing:
                output.write(np.zeros(trailing, dtype=np.float32))
                cursor += trailing
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    assert sample_rate is not None
    return sample_rate, cursor / sample_rate, timeline
