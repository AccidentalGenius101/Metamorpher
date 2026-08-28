from __future__ import annotations

import unittest

import _support  # noqa: F401

from metamorpher.model import ClassStatus, DomainTag
from metamorpher.version_space import Hypothesis, UnresolvedCell, VersionSpaceManager


def hypotheses() -> dict[str, Hypothesis]:
    return {
        "gasket": Hypothesis(
            "gasket",
            frozenset({"inspect", "stop"}),
            predictions={"soot_pattern": "joint"},
        ),
        "crack": Hypothesis(
            "crack",
            frozenset({"inspect", "replace_manifold", "stop"}),
            predictions={"soot_pattern": "crack"},
        ),
    }


class VersionSpaceTests(unittest.TestCase):
    def test_common_safe_action_is_intersection(self) -> None:
        cell = UnresolvedCell("leak", hypotheses())
        self.assertEqual(cell.common_safe_actions(), frozenset({"inspect", "stop"}))

    def test_observation_narrows_without_inventing_hypothesis(self) -> None:
        cell = UnresolvedCell("leak", hypotheses(), min_resolution_support=1)
        cell.observe("soot_pattern", "joint", "e1")
        self.assertEqual(cell.surviving_ids, ("gasket",))
        self.assertEqual(cell.status, ClassStatus.SUPPORTED)

    def test_unrepresented_test_does_not_oracle_narrow(self) -> None:
        cell = UnresolvedCell("leak", hypotheses(), min_resolution_support=1)
        cell.observe("future_magic_sensor", "gasket", "e1")
        self.assertEqual(cell.surviving_ids, ("crack", "gasket"))
        self.assertEqual(cell.status, ClassStatus.UNRESOLVED)

    def test_duplicate_evidence_cannot_satisfy_resolution_threshold(self) -> None:
        cell = UnresolvedCell("leak", hypotheses(), min_resolution_support=2)
        cell.observe("soot_pattern", "joint", "same")
        self.assertNotEqual(cell.status, ClassStatus.SUPPORTED)
        cell.observe("soot_pattern", "joint", "same")
        self.assertNotEqual(
            cell.status,
            ClassStatus.SUPPORTED,
            "one observation replayed twice is not two supporting observations",
        )

    def test_novel_contradiction_downgrades_supported_cell(self) -> None:
        cell = UnresolvedCell("leak", hypotheses(), min_resolution_support=1)
        cell.observe("soot_pattern", "joint", "e1")
        self.assertEqual(cell.status, ClassStatus.SUPPORTED)
        cell.observe("soot_pattern", "impossible-new-pattern", "e2")
        self.assertNotEqual(
            cell.status,
            ClassStatus.SUPPORTED,
            "an observation outside every surviving hypothesis must reopen uncertainty",
        )

    def test_manager_only_updates_active_cell(self) -> None:
        manager = VersionSpaceManager()
        first = UnresolvedCell("first", hypotheses())
        second = UnresolvedCell("second", hypotheses())
        manager.add(first, activate=True)
        manager.add(second)
        manager.observe("soot_pattern", "joint", "e1")
        self.assertEqual(first.surviving_ids, ("gasket",))
        self.assertEqual(second.surviving_ids, ("crack", "gasket"))

    def test_hypothesis_domain_is_preserved(self) -> None:
        domain = DomainTag.from_mapping("sierra", {"engine": "6.0"})
        item = Hypothesis("h", frozenset({"inspect"}), domain=domain, provenance=("failure-7",))
        self.assertEqual(item.domain, domain)
        self.assertEqual(item.provenance, ("failure-7",))


if __name__ == "__main__":
    unittest.main()
