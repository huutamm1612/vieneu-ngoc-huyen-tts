from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from train.artifacts import PipelineState, read_json, safe_rmtree
from train.config import load_config
from train.data import (
    _read_completed_filenames,
    audit_dataset,
    encoded_dataset_paths,
    format_training_sequence,
    read_pipe_metadata,
    require_encoded_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def test_metadata_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "audio"
            audio.mkdir()
            (audio / "train.wav").touch()
            (audio / "eval.wav").touch()
            (root / "metadata.csv").write_text("train.wav|Xin chào.\n", encoding="utf-8")
            (root / "metadata_eval.csv").write_text("eval.wav|Chào bạn.\n", encoding="utf-8")

            config = load_config(PROJECT_ROOT / "configs" / "pipeline_3phase.yaml")
            config.apply_overrides(dataset_root=str(root), output_root=str(root / "out"))
            rows = read_pipe_metadata(root / "metadata.csv")
            report = audit_dataset(config)

            self.assertEqual(rows[0].text, "Xin chào.")
            self.assertEqual(report["train_samples"], 1)
            self.assertEqual(report["eval_samples"], 1)
            self.assertEqual(report["missing_audio"], 0)
            self.assertEqual(report["cross_split_normalized_text_overlap"], 0)

    def test_cross_split_leakage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            audio = root / "audio"
            audio.mkdir()
            (audio / "same.wav").touch()
            (root / "metadata.csv").write_text("same.wav|Một.\n", encoding="utf-8")
            (root / "metadata_eval.csv").write_text("same.wav|Hai.\n", encoding="utf-8")
            config = load_config(PROJECT_ROOT / "configs" / "pipeline_3phase.yaml")
            config.apply_overrides(dataset_root=str(root), output_root=str(root / "out"))
            with self.assertRaisesRegex(ValueError, "leakage"):
                audit_dataset(config)

    def test_training_sequence_matches_vieneu_contract(self) -> None:
        sequence = format_training_sequence("xin chào.", [12, 34])
        self.assertEqual(
            sequence,
            "<|TEXT_PROMPT_START|>xin chào.<|TEXT_PROMPT_END|>"
            "<|SPEECH_GENERATION_START|><|speech_12|><|speech_34|>"
            "<|SPEECH_GENERATION_END|>",
        )

    def test_partial_cache_repairs_only_incomplete_last_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            partial = Path(temp_dir) / "train.partial.jsonl"
            partial.write_text(
                '{"filename":"a.wav"}\n{"filename":"b.wav"',
                encoding="utf-8",
            )
            completed = _read_completed_filenames(partial)
            self.assertEqual(completed, {"a.wav"})
            self.assertEqual(partial.read_text(encoding="utf-8"), '{"filename":"a.wav"}\n')

    def test_training_requires_a_completed_encoded_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_config(PROJECT_ROOT / "configs" / "pipeline_3phase.yaml")
            config.model.model_revision = "a" * 40
            config.model.codec_revision = "b" * 40
            config.apply_overrides(dataset_root=str(root), output_root=str(root / "out"))
            paths = encoded_dataset_paths(config)
            paths["root"].mkdir(parents=True)
            paths["train"].write_text("{}\n", encoding="utf-8")
            paths["eval"].write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "NeuCodec preparation stage"):
                require_encoded_dataset(config)

            paths["complete"].write_text("{}\n", encoding="utf-8")
            completed = require_encoded_dataset(config)
            self.assertEqual(completed["root"], paths["root"])


class ArtifactTests(unittest.TestCase):
    def test_pipeline_state_and_bounded_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state_path = root / "run" / "pipeline_state.json"
            state = PipelineState(state_path, "run-a", "signature-a")
            state.begin("phase1")
            state.complete("phase1", "/artifact", {"loss": 1.0})
            state.finish("/final")
            persisted = read_json(state_path)
            self.assertEqual(persisted["status"], "completed")

            disposable = root / "run" / ".runtime"
            disposable.mkdir(parents=True)
            safe_rmtree(disposable, root / "run")
            self.assertFalse(disposable.exists())
            with self.assertRaises(ValueError):
                safe_rmtree(root / "run", root / "run")


if __name__ == "__main__":
    unittest.main()
