from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path
from typing import Any, Protocol, Sequence

from .batching import build_length_batches
from .types import InferenceItem, PreparedBatches

ALLOWED_PUNCTUATION = set(".,!?")
QUOTE_CHARACTERS = set("\"'“”‘’`´«»‹›")
BRACKET_CHARACTERS = set("()[]{}（）［］｛｝")
SHORT_MERGE_PHONEMES = 35
LONG_SPLIT_CONNECTORS = (
    "nhưng",
    "tuy nhiên",
    "do đó",
    "vì vậy",
    "đồng thời",
    "còn",
    "nên",
    "rồi",
    "khi",
    "nếu",
    "vì",
    "để",
    "mà",
    "và",
    "hoặc",
)

ACRONYM_LETTERS = {
    "A": "ây",
    "B": "bi",
    "C": "xi",
    "D": "đi",
    "E": "i",
    "F": "ép",
    "G": "gi",
    "H": "âych",
    "I": "ai",
    "J": "giây",
    "K": "cây",
    "L": "eo",
    "M": "em",
    "N": "en",
    "O": "âu",
    "P": "pi",
    "Q": "kiu",
    "R": "a",
    "S": "ét",
    "T": "ti",
    "U": "diu",
    "V": "vi",
    "W": "đắp bồ diu",
    "X": "ích",
    "Y": "quai",
    "Z": "dét",
}
ACRONYM_PATTERN = re.compile(r"(?<!\w)(?:[A-Z]\.?){2,}(?!\w)")


class TextNormalizer(Protocol):
    def normalize(self, text: str) -> str: ...


class Phonemizer(Protocol):
    def run(self, text: str | list[str], *, punc_norm: bool = True) -> str | list[str]: ...


def decode_text_file(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "cp1258"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeError("Cannot decode the input TXT as UTF-8, UTF-16, or CP1258")


def read_text_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Input text file not found: {source}")
    return decode_text_file(source.read_bytes())


def expand_acronyms(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        return " ".join(ACRONYM_LETTERS[letter] for letter in re.findall(r"[A-Z]", match.group()))

    return ACRONYM_PATTERN.sub(replace, text)


def forbidden_characters(text: str, *, allow_digits: bool) -> list[str]:
    forbidden: set[str] = set()
    for char in text:
        if char.isspace():
            continue
        category = unicodedata.category(char)[0]
        if category in {"L", "M"} or (allow_digits and category == "N"):
            continue
        if char in ALLOWED_PUNCTUATION:
            continue
        forbidden.add(char)
    return sorted(forbidden)


def split_normalization_units(text: str) -> list[str]:
    cleaned = html.unescape(str(text))
    cleaned = expand_acronyms(cleaned)
    cleaned = unicodedata.normalize("NFC", cleaned).replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"[\u200b-\u200d\ufeff]", "", cleaned)
    units = re.split(r"\n+|(?<=[.!?;:…])\s+", cleaned)
    return [re.sub(r"\s+", " ", unit).strip() for unit in units if unit.strip()]


def prepare_normalization_unit(unit: str) -> str:
    unit = html.unescape(str(unit))
    unit = re.sub(r"</?[A-Za-z][^<>]*>", " ", unit)
    unit = "".join(char for char in unit if char not in QUOTE_CHARACTERS)
    unit = "".join(" " if char in BRACKET_CHARACTERS else char for char in unit)
    unit = re.sub(r"(?<=\d)\s*[-–—]\s*(?=\d)", " đến ", unit)
    unit = re.sub(r"(?<=\d)\s*%\s*", " phần trăm ", unit)
    unit = re.sub(r"^\s*[-–—]+\s*", "", unit)
    unit = re.sub(r"\s*[–—]+\s*", ". ", unit)
    unit = re.sub(r"(?<=\w)-(?=\w)", " ", unit)
    unit = re.sub(r"\s+-+\s+", ". ", unit)
    unit = unit.replace("&", " và ")
    unit = re.sub(r"[•●▪◦■□◆◇★☆]+", " ", unit)
    unit = re.sub(r"\s*[;；:：]+\s*", ". ", unit)
    unit = re.sub(r"(?:\.{2,}|…+)", ". ", unit)
    unit = re.sub(r"\s+", " ", unit).strip()
    forbidden = forbidden_characters(unit, allow_digits=True)
    if forbidden:
        raise ValueError(f"unsupported characters before number normalization: {forbidden}")
    return unit


def finalize_normalization_unit(unit: str) -> str:
    unit = unicodedata.normalize("NFC", str(unit)).lower()
    unit = unit.replace("…", ".")
    unit = re.sub(r"\s*[;；:：]+\s*", ". ", unit)
    unit = re.sub(r"\.{2,}", ".", unit)
    unit = re.sub(r",(?:\s*,)+", ",", unit)
    unit = re.sub(r"\s+([.,!?])", r"\1", unit)
    unit = re.sub(r"([.,!?])(?=\S)", r"\1 ", unit)
    unit = re.sub(r"\s+", " ", unit).strip(" ,")
    if re.search(r"\d", unit):
        raise ValueError(f"digits remain after number normalization: {unit}")
    forbidden = forbidden_characters(unit, allow_digits=False)
    if forbidden:
        raise ValueError(f"unsupported characters after normalization: {forbidden}")
    if unit.endswith(","):
        unit = unit.rstrip(" ,") + "."
    elif unit and not unit.endswith((".", "?", "!")):
        unit += "."
    return unit


def normalize_story(text: str, normalizer: TextNormalizer) -> str:
    units = split_normalization_units(text)
    if not units:
        raise ValueError("Input text is empty after normalization")
    normalized: list[str] = []
    for index, raw_unit in enumerate(units):
        try:
            unit = prepare_normalization_unit(raw_unit)
            unit = normalizer.normalize(unit)
            unit = finalize_normalization_unit(unit)
        except Exception as exc:
            raise ValueError(f"Cannot normalize text unit {index}: {raw_unit}") from exc
        if unit:
            normalized.append(unit)
    if not normalized:
        raise ValueError("Input text is empty after normalization")
    return "\n".join(normalized)


def finish_chunk(chunk: str) -> str:
    chunk = re.sub(r"\s+([.,!?])", r"\1", str(chunk))
    chunk = re.sub(r"([.,!?])(?=\S)", r"\1 ", chunk)
    chunk = re.sub(r"\s+", " ", chunk).strip()
    return chunk.rstrip(" ,.!?") + "." if chunk else ""


def join_units(left: str, right: str) -> str:
    left = re.sub(r"[.!?]+$", "", finish_chunk(left)).rstrip(" ,")
    right = finish_chunk(right).lstrip(" ,")
    return finish_chunk(f"{left}, {right}")


def split_punctuation_units(text: str) -> list[str]:
    pattern = r"[^.!?\n]+(?:[.!?]+|(?=\n)|$)"
    return [match.group().strip() for match in re.finditer(pattern, text) if match.group().strip()]


def split_long_unit(unit: str, *, target_chars: int, max_chars: int) -> list[str]:
    unit = re.sub(r"\s+", " ", str(unit)).strip()
    if len(unit) <= max_chars:
        return [unit]
    terminal = unit[-1] if unit.endswith((".", "?", "!")) else "."
    remaining = unit.rstrip(" ,.!?").strip()
    hard_limit = max_chars - 1
    if hard_limit < 1:
        raise ValueError("max_chars is too small")
    pieces: list[str] = []
    while len(remaining) + 1 > max_chars:
        piece_count = max(2, (len(remaining) + hard_limit - 1) // hard_limit)
        desired = min(hard_limit, max(1, round(len(remaining) / piece_count), target_chars - 1))
        lower = max(1, min(desired, round(desired * 0.55)))
        visible = remaining[: hard_limit + 1]
        comma_positions = [i for i, char in enumerate(visible) if char == "," and lower <= i <= hard_limit]
        connector_pattern = rf"\s+(?=(?:{'|'.join(map(re.escape, LONG_SPLIT_CONNECTORS))})\s)"
        connector_positions = [
            match.start()
            for match in re.finditer(connector_pattern, visible, flags=re.IGNORECASE)
            if lower <= match.start() <= hard_limit
        ]
        space_positions = [i for i, char in enumerate(visible) if char.isspace() and lower <= i <= hard_limit]
        positions = comma_positions or connector_positions or space_positions
        if not positions:
            positions = [i for i, char in enumerate(visible) if char.isspace() and i > 0]
        if not positions:
            raise ValueError(f"Cannot split a word below max_chars={max_chars}: {visible}")
        cut = min(positions, key=lambda position: (abs(position - desired), -position))
        pieces.append(remaining[:cut].rstrip(" ,.!?") + ".")
        remaining = remaining[cut + 1 :].lstrip(" ,.!?")
    if remaining:
        pieces.append(remaining.rstrip(" ,.!?") + terminal)
    return pieces


def _run_phonemizer(phonemizer: Phonemizer, texts: Sequence[str]) -> list[str]:
    if not texts:
        return []
    try:
        result = phonemizer.run(list(texts), punc_norm=True)
    except (TypeError, AttributeError):
        result = [phonemizer.run(text, punc_norm=True) for text in texts]
    if isinstance(result, str):
        if len(texts) != 1:
            result = [phonemizer.run(text, punc_norm=True) for text in texts]
        else:
            result = [result]
    phonemes = [str(value).strip() for value in result]
    if len(phonemes) != len(texts) or any(not value for value in phonemes):
        raise RuntimeError("Phonemizer returned an invalid number of non-empty outputs")
    return phonemes


def merge_short_units(
    units: Sequence[str],
    *,
    min_chars: int,
    target_chars: int,
    max_chars: int,
    phonemizer: Phonemizer,
) -> list[str]:
    texts = [finish_chunk(unit) for unit in units if str(unit).strip()]
    phonemes = _run_phonemizer(phonemizer, texts)
    rows = [{"text": text, "phoneme_length": len(phones)} for text, phones in zip(texts, phonemes)]
    merged: list[dict[str, Any]] = []
    index = 0
    while index < len(rows):
        current = rows[index]
        text_length = len(re.sub(r"[.!?]+$", "", current["text"]).strip())
        is_short = text_length < min_chars or current["phoneme_length"] < SHORT_MERGE_PHONEMES
        if not is_short or len(rows) == 1 or (not merged and index + 1 == len(rows)):
            merged.append(current)
            index += 1
            continue
        candidates: list[tuple[int, int, str, str]] = []
        if merged:
            combined = join_units(merged[-1]["text"], current["text"])
            if len(combined) <= max_chars:
                candidates.append((abs(len(combined) - target_chars), 1, "previous", combined))
        if index + 1 < len(rows):
            combined = join_units(current["text"], rows[index + 1]["text"])
            if len(combined) <= max_chars:
                candidates.append((abs(len(combined) - target_chars), 0, "next", combined))
        if not candidates:
            merged.append(current)
            index += 1
            continue
        _, _, direction, combined = min(candidates)
        combined_length = len(_run_phonemizer(phonemizer, [combined])[0])
        combined_row = {"text": combined, "phoneme_length": combined_length}
        if direction == "previous":
            merged[-1] = combined_row
        else:
            rows[index + 1] = combined_row
        index += 1
    return [row["text"] for row in merged]


def split_story(
    text: str,
    *,
    min_chars: int,
    target_chars: int,
    max_chars: int,
    phonemizer: Phonemizer,
) -> list[str]:
    if not 1 <= min_chars <= target_chars <= max_chars:
        raise ValueError("Expected 1 <= min_chars <= target_chars <= max_chars")
    units: list[str] = []
    for unit in split_punctuation_units(text):
        units.extend(split_long_unit(unit, target_chars=target_chars, max_chars=max_chars))
    chunks = merge_short_units(
        units,
        min_chars=min_chars,
        target_chars=target_chars,
        max_chars=max_chars,
        phonemizer=phonemizer,
    )
    if not chunks:
        raise ValueError("No inference chunks were produced")
    if any(len(chunk) > max_chars for chunk in chunks):
        raise RuntimeError("A generated chunk exceeds max_chars")
    return chunks


class StoryPreprocessor:
    def __init__(self, normalizer: TextNormalizer | None = None, phonemizer: Phonemizer | None = None) -> None:
        if normalizer is None or phonemizer is None:
            try:
                from sea_g2p import Normalizer, SEAPipeline
            except ImportError as exc:
                raise RuntimeError("Install the inference dependencies with: pip install -e '.[inference]'") from exc
            normalizer = normalizer or Normalizer(lang="vi")
            phonemizer = phonemizer or SEAPipeline(lang="vi")
        self.normalizer = normalizer
        self.phonemizer = phonemizer

    def phonemize(self, texts: Sequence[str]) -> list[str]:
        return _run_phonemizer(self.phonemizer, texts)

    def prepare(
        self,
        text: str,
        *,
        min_chars: int = 80,
        target_chars: int = 128,
        max_chars: int = 156,
        batch_size: int = 128,
        max_length_gap: int = 12,
        source_name: str = "text",
    ) -> tuple[PreparedBatches, str]:
        normalized = normalize_story(text, self.normalizer)
        chunks = split_story(
            normalized,
            min_chars=min_chars,
            target_chars=target_chars,
            max_chars=max_chars,
            phonemizer=self.phonemizer,
        )
        phonemes = self.phonemize(chunks)
        items: list[InferenceItem] = [
            {
                "index": index,
                "uid": f"{source_name}:{index}",
                "text": chunk,
                "phonemes": phones,
                "source_audio_path": "",
            }
            for index, (chunk, phones) in enumerate(zip(chunks, phonemes))
        ]
        return (
            build_length_batches(items, batch_size=batch_size, max_length_gap=max_length_gap),
            normalized,
        )
