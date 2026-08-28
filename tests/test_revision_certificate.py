from __future__ import annotations

import unittest

import _support  # noqa: F401

from metamorpher.carving import ConstraintRevision, apply_constraint_revisions
from metamorpher.certificate import (
    assert_fresh,
    decision_token,
    supporting_evidence_ids,
)
from metamorpher.evidence import EvidenceLedger
from metamorpher.graph import TypedActionGraph
from metamorpher.model import (
    ActionNode,
    Constraint,
    ConstraintKind,
    Decision,
    DecisionStatus,
    DomainTag,
    InvalidGraphError,
    Observation,
    ObservationStatus,
    StaleDecisionError,
)


def base_graph() -> TypedActionGraph:
    graph = TypedActionGraph()
    graph.add_node(ActionNode("inspect", "Inspect fastener"))
    graph.add_node(ActionNode("remove", "Remove manifold", irreversible=True, reversible=False))
    return graph


def decision_for(graph, ledger, domain) -> Decision:
    token = decision_token(
        sequence=1,
        graph_epoch=graph.epoch,
        evidence_revision=ledger.revision,
        domain=domain,
        status=DecisionStatus.SUPPORTED_UNDER_MODEL.value,
        action_id="remove",
        probe_id=None,
        frontier=("remove",),
    )
    return Decision(
        DecisionStatus.SUPPORTED_UNDER_MODEL,
        "remove",
        None,
        "supported",
        ("remove",),
        graph.epoch,
        ledger.revision,
        domain,
        token,
    )


class CertificateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = base_graph()
        self.ledger = EvidenceLedger()
        self.domain = DomainTag.from_mapping("sierra", {"engine": "6.0"})

    def test_fresh_decision_passes(self) -> None:
        assert_fresh(decision_for(self.graph, self.ledger, self.domain), self.graph, self.ledger)

    def test_evidence_revision_makes_decision_stale(self) -> None:
        decision = decision_for(self.graph, self.ledger, self.domain)
        self.ledger.append(Observation("new", "bolt_present", False, source="inspection"))
        with self.assertRaises(StaleDecisionError):
            assert_fresh(decision, self.graph, self.ledger)

    def test_graph_revision_makes_decision_stale(self) -> None:
        decision = decision_for(self.graph, self.ledger, self.domain)
        transaction = self.graph.transaction()
        with transaction:
            transaction.add_constraint(
                Constraint(
                    "inspect_first",
                    ConstraintKind.HARD_PREREQUISITE,
                    ("inspect",),
                    "remove",
                )
            )
            transaction.commit()
        with self.assertRaises(StaleDecisionError):
            assert_fresh(decision, self.graph, self.ledger)

    def test_decision_token_is_snapshot_bound_and_deterministic(self) -> None:
        first = decision_for(self.graph, self.ledger, self.domain).token
        second = decision_for(self.graph, self.ledger, self.domain).token
        self.assertEqual(first, second)
        self.ledger.append(Observation("new", "bolt_present", True, source="inspection"))
        third = decision_for(self.graph, self.ledger, self.domain).token
        self.assertNotEqual(first, third)

    def test_supporting_evidence_comes_from_ledger(self) -> None:
        self.ledger.append(Observation("seen", "engine_cold", True, source="sensor"))
        self.graph.add_constraint(
            Constraint(
                "cold",
                ConstraintKind.GUARD,
                (),
                "remove",
                fact_key="engine_cold",
                expected_value=True,
                provenance=("policy-document", "seen"),
            )
        )
        self.assertEqual(supporting_evidence_ids(self.graph, self.ledger, "remove"), ("seen",))


class ConstraintRevisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = base_graph()
        self.observed = Observation("soot", "soot_at_bolt", True, source="camera")

    def revision(self, evidence_ids=("soot",)) -> ConstraintRevision:
        return ConstraintRevision(
            Constraint(
                "inspect_first",
                ConstraintKind.HARD_PREREQUISITE,
                ("inspect",),
                "remove",
            ),
            tuple(evidence_ids),
            "Soot indicates the joint must be inspected before removal.",
        )

    def test_observed_evidence_can_atomically_add_constraint(self) -> None:
        result = apply_constraint_revisions(self.graph, (self.revision(),), (self.observed,))
        self.assertIsNotNone(result)
        self.assertIn("inspect_first", self.graph.constraints)
        self.assertEqual(self.graph.epoch, 1)
        self.assertIn("soot", self.graph.constraints["inspect_first"].provenance)

    def test_censored_evidence_cannot_carve(self) -> None:
        censored = Observation(
            "soot",
            "soot_at_bolt",
            None,
            status=ObservationStatus.CENSORED,
            source="not_audited",
            censoring_reason="camera_not_used",
        )
        with self.assertRaises(ValueError):
            apply_constraint_revisions(self.graph, (self.revision(),), (censored,))
        self.assertNotIn("inspect_first", self.graph.constraints)
        self.assertEqual(self.graph.epoch, 0)

    def test_unknown_evidence_id_rejects_without_partial_revision(self) -> None:
        with self.assertRaises(ValueError):
            apply_constraint_revisions(self.graph, (self.revision(("missing",)),), (self.observed,))
        self.assertEqual(self.graph.constraints, {})
        self.assertEqual(self.graph.epoch, 0)

    def test_cycle_batch_is_atomic(self) -> None:
        revision_one = self.revision()
        revision_two = ConstraintRevision(
            Constraint("remove_to_inspect", ConstraintKind.HARD_PREREQUISITE, ("remove",), "inspect"),
            ("soot",),
            "Adversarial reverse edge.",
        )
        with self.assertRaises(InvalidGraphError):
            apply_constraint_revisions(self.graph, (revision_one, revision_two), (self.observed,))
        self.assertEqual(self.graph.constraints, {})
        self.assertEqual(self.graph.epoch, 0)


if __name__ == "__main__":
    unittest.main()
