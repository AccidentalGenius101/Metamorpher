from __future__ import annotations

import unittest

import _support  # noqa: F401

try:
    from experiments.continual_replay import ReplayBuffer, run_benchmark, run_method
except ImportError:  # pragma: no cover - NumPy is an optional experiment extra.
    ReplayBuffer = None
    run_benchmark = None
    run_method = None


@unittest.skipIf(run_benchmark is None, "NumPy experiment dependency unavailable")
class ContinualReplayTests(unittest.TestCase):
    def test_run_is_deterministic_for_seed_and_method(self) -> None:
        first = run_method(7, "prioritized")
        second = run_method(7, "prioritized")
        self.assertEqual(first, second)

    def test_replay_improves_retention_under_fixed_compute(self) -> None:
        _, summary = run_benchmark(seeds=8)
        no_replay = summary["none"]
        for method in ("random", "prioritized", "gated"):
            replay = summary[method]
            self.assertGreater(
                replay["final_average_accuracy"],
                no_replay["final_average_accuracy"] + 0.08,
            )
            self.assertLess(
                replay["mean_forgetting"],
                no_replay["mean_forgetting"] - 0.07,
            )

    def test_gating_uses_less_replay_than_always_replay(self) -> None:
        _, summary = run_benchmark(seeds=8)
        gated = summary["gated"]
        always = summary["prioritized"]
        self.assertGreater(gated["rejected_candidate_batches"], 0)
        self.assertGreater(gated["replay_examples"], 0)
        self.assertLess(gated["replay_examples"], always["replay_examples"] * 0.65)

    def test_reservoir_memory_remains_bounded(self) -> None:
        import numpy as np

        rng = np.random.default_rng(3)
        memory = ReplayBuffer(12, rng)
        features = rng.normal(size=(100, 2))
        labels = rng.integers(0, 2, size=100).astype(float)
        memory.add(features, labels, np.ones(100))
        self.assertEqual(memory.size, 12)
        self.assertEqual(memory.seen, 100)


if __name__ == "__main__":
    unittest.main()
