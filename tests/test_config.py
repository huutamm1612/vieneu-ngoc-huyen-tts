from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from train.config import PipelineConfig, load_config
from train.pipeline import _is_commit_sha


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_repository_config_loads(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "pipeline_3phase.yaml")
        self.assertEqual(config.model.base_model, "pnnbao-ump/VieNeu-TTS")
        self.assertEqual(config.phase1_lora.learning_rate, 1.0e-4)
        self.assertEqual(config.phase3_finetune.learning_rate, 1.0e-6)
        self.assertEqual(
            config.phase1_lora.per_device_train_batch_size
            * config.phase1_lora.gradient_accumulation_steps,
            32,
        )
        self.assertEqual(
            config.phase3_finetune.per_device_train_batch_size
            * config.phase3_finetune.gradient_accumulation_steps,
            32,
        )

    def test_signature_changes_with_training_override(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "pipeline_3phase.yaml")
        original = config.signature()
        config.apply_overrides(phase3_learning_rate=2.0e-6)
        self.assertNotEqual(original, config.signature())

    def test_unknown_section_is_rejected(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "pipeline_3phase.yaml")
        raw = config.as_dict()
        raw["unexpected"] = {}
        with self.assertRaisesRegex(ValueError, "Unknown configuration sections"):
            PipelineConfig.from_dict(raw)

    def test_unsafe_run_name_is_rejected(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "pipeline_3phase.yaml")
        with self.assertRaisesRegex(ValueError, "safe path component"):
            config.apply_overrides(run_name="../outside")

    def test_only_full_hub_commit_is_immutable(self) -> None:
        self.assertTrue(_is_commit_sha("a" * 40))
        self.assertFalse(_is_commit_sha("main"))
        self.assertFalse(_is_commit_sha(None))


if __name__ == "__main__":
    unittest.main()
