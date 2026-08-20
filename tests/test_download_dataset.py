from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "data" / "download_dataset.py"
SPEC = importlib.util.spec_from_file_location("download_dataset", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
download_dataset = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = download_dataset
SPEC.loader.exec_module(download_dataset)


class FakeNormalizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def normalize(self, text: str) -> str:
        self.calls.append(text)
        return text.replace("12", "mười hai").lower()


class DownloadDatasetTests(unittest.TestCase):
    def test_target_filename_preserves_numeric_id(self) -> None:
        self.assertEqual(
            download_dataset.target_filename("audio/ngochuyen_00008.wav", "ngoc_base01", 5),
            "ngoc_base01_00008.wav",
        )

    def test_prepare_text_removes_quotes_and_replaces_semicolon(self) -> None:
        normalizer = FakeNormalizer()
        result = download_dataset.prepare_text("“Ngày 12; thử nghiệm.”", normalizer)
        self.assertEqual(result.after_punctuation, "Ngày 12. thử nghiệm.")
        self.assertEqual(result.normalized, "ngày mười hai. thử nghiệm.")
        self.assertEqual(result.reasons, ())
        self.assertEqual(normalizer.calls, ["Ngày 12. thử nghiệm."])

    def test_prepare_text_rejects_forbidden_before_normalizer(self) -> None:
        normalizer = FakeNormalizer()
        result = download_dataset.prepare_text("Ngày 23/4/2024.", normalizer)
        self.assertIsNone(result.normalized)
        self.assertEqual(result.forbidden_characters, ("/",))
        self.assertEqual(result.reasons, ("forbidden_character_before_number_normalization",))
        self.assertEqual(normalizer.calls, [])

    def test_write_new_or_identical_never_overwrites_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            self.assertTrue(download_dataset.write_new_or_identical(path, b"first"))
            self.assertFalse(download_dataset.write_new_or_identical(path, b"first"))
            with self.assertRaises(FileExistsError):
                download_dataset.write_new_or_identical(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")

    def test_audit_reports_extra_audio_without_deleting_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "audio"
            audio.mkdir()
            (audio / "train.wav").write_bytes(b"train")
            (audio / "eval.wav").write_bytes(b"eval")
            extra = audio / "sample-kept.wav"
            extra.write_bytes(b"keep")
            (root / "metadata.csv").write_text("train.wav|nội dung train.\n", encoding="utf-8")
            (root / "metadata_eval.csv").write_text("eval.wav|nội dung eval.\n", encoding="utf-8")

            report = download_dataset.audit_output(root)

            self.assertEqual(report["extra_wav_preserved"], 1)
            self.assertTrue(extra.is_file())
            self.assertEqual(extra.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
