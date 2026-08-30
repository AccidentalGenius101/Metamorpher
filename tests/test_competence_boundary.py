from __future__ import annotations

import math
import unittest
from dataclasses import replace

import _support  # noqa: F401

try:
    from experiments.competence_boundary import (
        collision_task,
        run_benchmark,
        run_method,
    )
except ImportError:  # pragma: no cover - NumPy is an optional experiment extra.
    collision_task = None
    run_benchmark = None
    run_method = None


@unittest.skipIf(run_benchmark is None, "NumPy experiment dependency unavailable")
class CompetenceBoundaryTests(unittest.TestCase):
    def test_collision_reuses_inputs_and_inverts_labels(self) -> None:
        import numpy as np

        first = collision_task(np.random.default_rng(4), 32)
        normal_rng = np.random.default_rng(4)
        from experiments.continual_replay import make_task

        normal = make_task(normal_rng, "xor", 32)
        self.assertTrue(np.array_equal(first.features, normal.features))
        self.assertTrue(np.array_equal(first.labels, 1.0 - normal.labels))

    def test_run_is_deterministic(self) -> None:
        first = run_method(8, "boundary")
        second = run_method(8, "boundary")
        self.assertTrue(math.isnan(first.collision_accuracy))
        self.assertTrue(math.isnan(second.collision_accuracy))
        self.assertEqual(
            replace(first, collision_accuracy=0.0),
            replace(second, collision_accuracy=0.0),
        )

    def test_persistent_detector_finds_collision_without_false_alarm(self) -> None:
        results, summary = run_benchmark(seeds=12)
        boundary = summary["boundary"]
        self.assertGreaterEqual(boundary["detection_rate"], 0.70)
        self.assertEqual(boundary["false_boundary_batches"], 0.0)
        detected = [
            result for result in results
            if result.method == "boundary" and result.boundary_detected
        ]
        self.assertTrue(detected)
        self.assertTrue(
            all(math.isnan(result.collision_accuracy) for result in detected)
        )
        self.assertTrue(
            all(result.harmful_updates_after_detection == 0 for result in detected)
        )

    def test_oracle_is_explicitly_separate_from_tested_learner(self) -> None:
        _, summary = run_benchmark(seeds=8)
        self.assertEqual(summary["oracle"]["coverage"], 1.0)
        self.assertIsNotNone(summary["oracle"]["collision_accuracy"])
        self.assertLess(summary["boundary"]["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
