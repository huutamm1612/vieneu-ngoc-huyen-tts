from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from train.artifacts import read_json
from train.config import load_config
from train.data import encoded_dataset_paths
from train.pipeline import prepare_pipeline_data, run_training_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PipelineSplitTests(unittest.TestCase):
    def test_preparation_commits_state_without_running_training(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = load_config(PROJECT_ROOT / "configs" / "pipeline_3phase.yaml")
            config.model.model_revision = "a" * 40
            config.model.codec_revision = "b" * 40
            config.apply_overrides(dataset_root=str(root), output_root=str(root / "out"))
            cache_root = config.output_root / "cache" / "encoded" / "test-cache"
            cache_root.mkdir(parents=True)
            (cache_root / "encoding_report.json").write_text(
                '{"splits":{"train":{"samples":1},"eval":{"samples":1}}}\n',
                encoding="utf-8",
            )
            commits: list[bool] = []

            with (
                patch(
                    "train.pipeline.prepare_encoded_dataset",
                    return_value={
                        "root": cache_root,
                        "train": cache_root / "train.jsonl",
                        "eval": cache_root / "eval.jsonl",
                    },
                ),
                patch("train.pipeline._environment_report", return_value={"gpu": {"cuda_available": False}}),
            ):
                result = prepare_pipeline_data(config, commit_callback=lambda: commits.append(True))

            state = read_json(config.run_root / "pipeline_state.json")
            self.assertEqual(result["status"], "prepared")
            self.assertEqual(state["phases"]["prepare_data"]["status"], "completed")
            self.assertNotIn("phase1_lora", state["phases"])
            self.assertTrue((config.run_root / "preparation_environment.json").is_file())
            self.assertGreaterEqual(len(commits), 2)

    def test_gpu_training_stage_uses_cache_without_calling_encoder(self) -> None:
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
            paths["complete"].write_text("{}\n", encoding="utf-8")
            (paths["root"] / "encoding_report.json").write_text("{}\n", encoding="utf-8")

            def phase_result(phase_name: str):
                final_dir = config.run_root / phase_name / "final"
                final_dir.mkdir(parents=True, exist_ok=True)
                return final_dir, {"phase": phase_name}

            with (
                patch(
                    "train.pipeline.prepare_encoded_dataset",
                    side_effect=AssertionError("GPU stage must not run NeuCodec"),
                ),
                patch(
                    "train.pipeline.train_phase1_lora",
                    side_effect=lambda *_args: phase_result("phase1_lora"),
                ),
                patch(
                    "train.pipeline.merge_phase2",
                    side_effect=lambda *_args: phase_result("phase2_merge"),
                ),
                patch(
                    "train.pipeline.train_phase3_finetune",
                    side_effect=lambda *_args: phase_result("phase3_finetune"),
                ),
                patch("train.pipeline._environment_report", return_value={"gpu": {"cuda_available": True}}),
            ):
                result = run_training_pipeline(config)

            self.assertEqual(result["status"], "completed")
            self.assertTrue((config.run_root / "environment.json").is_file())


if __name__ == "__main__":
    unittest.main()
