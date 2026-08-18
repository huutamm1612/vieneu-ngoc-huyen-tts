from __future__ import annotations

import unittest

from train.phases import _assert_finite_eval_loss, _best_global_step, _json_safe_metrics


class PhaseHelperTests(unittest.TestCase):
    def test_best_global_step(self) -> None:
        self.assertEqual(_best_global_step("/tmp/checkpoint-250"), 250)
        self.assertIsNone(_best_global_step(None))
        self.assertIsNone(_best_global_step("/tmp/model"))

    def test_metrics_are_json_safe(self) -> None:
        self.assertEqual(_json_safe_metrics({"loss": 1.2, "note": "ok"}), {"loss": 1.2, "note": "ok"})

    def test_non_finite_eval_loss_is_rejected(self) -> None:
        _assert_finite_eval_loss({"eval_loss": 1.0})
        with self.assertRaises(FloatingPointError):
            _assert_finite_eval_loss({"eval_loss": float("nan")})


if __name__ == "__main__":
    unittest.main()
