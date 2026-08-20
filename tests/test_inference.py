from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from inference import InferenceConfig, prepare_batches
from inference.batching import (
    assign_batches_to_workers,
    build_length_batches,
    split_batches_for_execution,
    validate_batches,
)
from inference.modeling import CHAT_TEMPLATE, InferenceWorker
from inference.preprocessing import StoryPreprocessor
from inference.generation import run_generation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class IdentityNormalizer:
    def normalize(self, text: str) -> str:
        return text


class FakePhonemizer:
    def run(self, text, *, punc_norm=True):
        if isinstance(text, list):
            return [f"ph:{value}" for value in text]
        return f"ph:{text}"


def item(index: int, phoneme_size: int):
    return {
        "index": index,
        "uid": f"sample:{index}",
        "text": f"Câu {index}.",
        "phonemes": "p" * phoneme_size,
        "source_audio_path": "",
    }


class FakeTokenizer:
    pad_token_id = 0
    unk_token_id = -1

    def encode(self, text: str, add_special_tokens: bool = True):
        if text == CHAT_TEMPLATE:
            return [10, 11, 12, 13]
        return [200 + index for index, _ in enumerate(text.split())]


class InferenceConfigTests(unittest.TestCase):
    def test_repository_config_has_requested_defaults(self) -> None:
        config = InferenceConfig.from_yaml(PROJECT_ROOT / "configs" / "inference.yaml")
        self.assertEqual((config.min_chars, config.target_chars, config.max_chars), (80, 128, 156))
        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.max_length_gap, 12)
        self.assertFalse(config.do_sample)
        self.assertTrue(config.show_progress)

    def test_invalid_thresholds_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_chars"):
            InferenceConfig.from_dict({"min_chars": 160, "target_chars": 128, "max_chars": 156})


class BatchingTests(unittest.TestCase):
    def test_length_bucketing_and_worker_assignment_preserve_all_indices(self) -> None:
        batches = build_length_batches(
            [item(0, 30), item(1, 10), item(2, 12), item(3, 31)],
            batch_size=2,
            max_length_gap=3,
        )
        self.assertEqual([[row["index"] for row in batch] for batch in batches], [[1, 2], [0, 3]])
        execution = split_batches_for_execution(batches, worker_count=4, max_runtime_batch_size=1)
        assignments, _ = assign_batches_to_workers(execution, 4)
        found = sorted(row["index"] for queue in assignments for _, batch in queue for row in batch)
        self.assertEqual(found, [0, 1, 2, 3])

    def test_duplicate_indices_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_batches([[item(0, 10)], [item(0, 11)]])


class PreprocessingTests(unittest.TestCase):
    def test_prepare_batches_can_be_called_without_optional_ml_imports(self) -> None:
        processor = StoryPreprocessor(IdentityNormalizer(), FakePhonemizer())
        text = (
            "Đây là câu chuyện đầu tiên có độ dài vừa đủ để kiểm tra việc chia đoạn và giữ dấu câu. "
            "Đây là câu tiếp theo được nối vào nội dung để tạo thêm một đoạn đọc tự nhiên và rõ ràng."
        )
        batches = prepare_batches(
            text=text,
            min_chars=40,
            target_chars=60,
            max_chars=90,
            batch_size=8,
            max_length_gap=100,
            preprocessor=processor,
        )
        flattened = sorted((row for batch in batches for row in batch), key=lambda row: row["index"])
        self.assertGreaterEqual(len(flattened), 2)
        self.assertTrue(all(row["text"].endswith(".") for row in flattened))
        self.assertTrue(all(row["phonemes"].startswith("ph:") for row in flattened))

    def test_exactly_one_text_source_is_required(self) -> None:
        processor = StoryPreprocessor(IdentityNormalizer(), FakePhonemizer())
        with self.assertRaisesRegex(ValueError, "exactly one"):
            prepare_batches(text="x", input_path="x.txt", preprocessor=processor)


class PromptContractTests(unittest.TestCase):
    def test_prompt_contains_reference_codes_and_target_phonemes(self) -> None:
        tokenizer = FakeTokenizer()
        worker = InferenceWorker(
            worker_id=0,
            device="cpu",
            dtype="float32",
            model=None,
            tokenizer=tokenizer,
            codec=None,
            pad_id=0,
            text_replace_id=11,
            speech_replace_id=13,
            text_prompt_start_id=20,
            text_prompt_end_id=21,
            speech_start_id=22,
            speech_end_id=23,
            speech_token_min=1000,
            speech_token_max=66535,
            context_limit=2048,
            reference_phonemes="ref phones",
            reference_codes=[2, 4],
            reference_code_token_ids=[1002, 1004],
        )
        prompt = worker.build_prompt("target phones")
        self.assertEqual(prompt[-3:], [22, 1002, 1004])
        self.assertEqual(prompt[:2], [10, 20])
        self.assertIn(21, prompt)


class ProgressTests(unittest.TestCase):
    def test_progress_counts_successful_segments_once(self) -> None:
        batches = [[item(0, 10)], [item(1, 11)]]
        config = InferenceConfig(max_runtime_batch_size=1, max_retries=0, show_progress=True)
        worker = SimpleNamespace(device=SimpleNamespace(type="cuda"))
        progress = MagicMock()

        def fake_generate(_worker, batch, _directory, _config):
            return [
                {
                    "index": row["index"],
                    "status": "ok",
                    "text": row["text"],
                    "segment_path": f"segment_{row['index']}.wav",
                }
                for row in batch
            ]

        with tempfile.TemporaryDirectory() as directory:
            with patch("tqdm.auto.tqdm", return_value=progress) as tqdm_mock:
                with patch("inference.generation._generate_batch", side_effect=fake_generate):
                    metadata, _ = run_generation([worker], batches, directory, config)

        self.assertEqual([row["index"] for row in metadata], [0, 1])
        tqdm_mock.assert_called_once_with(
            total=2,
            desc="TTS inference",
            unit="segment",
            dynamic_ncols=True,
            mininterval=0.5,
            leave=True,
        )
        self.assertEqual(sum(call.args[0] for call in progress.update.call_args_list), 2)
        progress.close.assert_called_once_with()


class NotebookTests(unittest.TestCase):
    def test_kaggle_notebook_is_valid_and_has_no_literal_hf_token(self) -> None:
        path = PROJECT_ROOT / "notebooks" / "inference_kaggle.ipynb"
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["nbformat"], 4)
        source = "\n".join("".join(cell.get("source", [])) for cell in payload["cells"])
        self.assertNotRegex(source, r"hf_[A-Za-z0-9]{20,}")
        self.assertIn("ngochuyen_00769.wav", source)
        for cell in payload["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell.get("source", [])))


if __name__ == "__main__":
    unittest.main()
