from __future__ import annotations

import unittest

import _support  # noqa: F401

try:
    from experiments.hidden_dimension_expansion import run_benchmark
except ImportError:  # pragma: no cover - NumPy is an optional experiment extra.
    run_benchmark = None


@unittest.skipIf(run_benchmark is None, "NumPy experiment dependency unavailable")
class HiddenDimensionExperimentTests(unittest.TestCase):
    def test_expansion_detects_helix_without_expanding_circle(self) -> None:
        _, summary = run_benchmark(seeds=10)
        self.assertGreaterEqual(summary["helix_expansion_detection_rate"], 0.9)
        self.assertLessEqual(summary["circle_false_expansion_rate"], 0.1)

    def test_expansion_improves_future_prediction_and_preserves_parent(self) -> None:
        _, summary = run_benchmark(seeds=10)
        self.assertLess(
            summary["helix_gated_future_mse"],
            summary["helix_fixed_future_mse"] * 0.10,
        )
        self.assertLess(
            summary["parent_return_mse"],
            summary["destructive_replacement_return_mse"] * 0.10,
        )


if __name__ == "__main__":
    unittest.main()
