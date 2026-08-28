from __future__ import annotations

import unittest

import _support  # noqa: F401

from metamorpher.carving import FailureCarver
from metamorpher.model import ClassStatus


class FailureCarvingTests(unittest.TestCase):
    def test_provisional_generalization_becomes_supported_without_contradiction(self) -> None:
        carver = FailureCarver("cell", min_branch_support=3)
        self.assertEqual(carver.observe("e1", "works").status, ClassStatus.PROVISIONAL)
        self.assertEqual(carver.observe("e2", "works").status, ClassStatus.PROVISIONAL)
        self.assertEqual(carver.observe("e3", "works").status, ClassStatus.SUPPORTED)

    def test_contradiction_creates_unresolved_class_without_explanation(self) -> None:
        carver = FailureCarver("cell", min_branch_support=2)
        carver.observe("e1", "left")
        result = carver.observe("e2", "right")
        self.assertEqual(result.status, ClassStatus.UNRESOLVED)
        self.assertIsNone(result.separator_name)
        self.assertEqual(result.branches, ())

    def test_duplicate_evidence_is_idempotent(self) -> None:
        carver = FailureCarver("cell", min_branch_support=2)
        carver.observe("same", "left")
        duplicate = carver.observe("same", "left")
        self.assertEqual(duplicate.support, 1)
        with self.assertRaises(ValueError):
            carver.observe("same", "right")

    def test_missing_separator_assignment_preserves_unresolved_parent(self) -> None:
        carver = FailureCarver("cell", min_branch_support=2)
        carver.observe("e1", "left")
        carver.observe("e2", "right")
        result = carver.try_carve("sensor", {"e1": "A"})
        self.assertEqual(result.status, ClassStatus.UNRESOLVED)
        self.assertIsNone(result.separator_name)

    def test_stable_supported_separator_carves(self) -> None:
        carver = FailureCarver("cell", min_branch_support=3, stability_threshold=0.8)
        assignments = {}
        for index in range(3):
            evidence_id = f"a{index}"
            carver.observe(evidence_id, "left")
            assignments[evidence_id] = "A"
        for index in range(3):
            evidence_id = f"b{index}"
            carver.observe(evidence_id, "right")
            assignments[evidence_id] = "B"
        result = carver.try_carve("observed_sensor", assignments)
        self.assertEqual(result.status, ClassStatus.CARVED)
        self.assertEqual(result.separator_name, "observed_sensor")
        self.assertEqual(len(result.branches), 2)
        self.assertEqual({branch.dominant_outcome for branch in result.branches}, {"left", "right"})

    def test_unique_identifiers_do_not_create_singleton_carving(self) -> None:
        carver = FailureCarver("cell", min_branch_support=2)
        assignments = {}
        for index, outcome in enumerate(("left", "right", "left", "right")):
            evidence_id = f"e{index}"
            carver.observe(evidence_id, outcome)
            assignments[evidence_id] = f"unique-{index}"
        result = carver.try_carve("case_id", assignments)
        self.assertEqual(result.status, ClassStatus.UNRESOLVED)
        self.assertIsNone(result.separator_name)

    def test_unstable_separator_does_not_force_partition(self) -> None:
        carver = FailureCarver("cell", min_branch_support=2, stability_threshold=0.8)
        outcomes = {
            "a1": "left",
            "a2": "right",
            "b1": "left",
            "b2": "right",
        }
        assignments = {key: key[0].upper() for key in outcomes}
        for evidence_id, outcome in outcomes.items():
            carver.observe(evidence_id, outcome)
        result = carver.try_carve("noisy_feature", assignments)
        self.assertEqual(result.status, ClassStatus.UNRESOLVED)
        self.assertEqual(result.branches, ())


if __name__ == "__main__":
    unittest.main()
