from __future__ import annotations

import unittest

import _support  # noqa: F401

try:
    from experiments.decision_gated_expansion import (
        control_cell,
        run_benchmark,
        run_control_world,
    )
except ImportError:  # pragma: no cover - NumPy is an optional experiment extra.
    control_cell = None
    run_benchmark = None
    run_control_world = None


@unittest.skipIf(run_benchmark is None, "NumPy experiment dependency unavailable")
class DecisionGatedExpansionTests(unittest.TestCase):
    def test_control_equivalence_uses_existing_common_safe_intersection(self) -> None:
        self.assertEqual(
            control_cell("descriptive_only").common_safe_actions(),
            frozenset({"coast"}),
        )
        self.assertEqual(
            control_cell("control_relevant").common_safe_actions(),
            frozenset(),
        )

    def test_descriptive_expansion_does_not_force_measurement(self) -> None:
        result = run_control_world(3, "descriptive_only", "decision_gated")
        self.assertTrue(result.epistemic_expansion_promoted)
        self.assertFalse(result.control_refinement_warranted)
        self.assertEqual(result.coordinate_measurements, 0)
        self.assertEqual(result.control_regret, 0.0)

    def test_control_relevant_expansion_requires_refinement(self) -> None:
        gated = run_control_world(3, "control_relevant", "decision_gated")
        coarse = run_control_world(3, "control_relevant", "never_refine")
        self.assertTrue(gated.control_refinement_warranted)
        self.assertEqual(gated.coordinate_measurements, gated.decisions)
        self.assertEqual(gated.control_regret, 0.0)
        self.assertGreater(coarse.control_regret, 0.0)
        self.assertGreater(coarse.unsafe_action_disagreements, 0)

    def test_decision_gate_cannot_use_rejected_coordinate(self) -> None:
        result = run_control_world(
            3,
            "control_relevant",
            "decision_gated",
            improvement_ratio=0.0,
        )
        self.assertFalse(result.epistemic_expansion_promoted)
        self.assertFalse(result.control_refinement_warranted)
        self.assertEqual(result.coordinate_measurements, 0)
        self.assertGreater(result.control_regret, 0.0)

    def test_decision_gate_matches_oracle_control_with_fewer_measurements(self) -> None:
        _, summary = run_benchmark(seeds=8)
        descriptive = summary["descriptive_only"]
        relevant = summary["control_relevant"]
        self.assertEqual(
            descriptive["decision_gated"]["mean_control_regret"],
            descriptive["always_refine"]["mean_control_regret"],
        )
        self.assertEqual(
            descriptive["decision_gated"]["mean_coordinate_measurements"],
            0.0,
        )
        self.assertEqual(
            relevant["decision_gated"]["mean_control_regret"],
            relevant["always_refine"]["mean_control_regret"],
        )


if __name__ == "__main__":
    unittest.main()
