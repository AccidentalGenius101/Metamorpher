from __future__ import annotations

import unittest

import _support  # noqa: F401

from metamorpher.backends import NumPyBackend, ReferenceBackend
from metamorpher.backends.base import AccelerationBackend, BatchedScoreResult
from metamorpher.batch import GraphBatchCompiler
from metamorpher.evidence import EvidenceLedger
from metamorpher.graph import TypedActionGraph
from metamorpher.model import (
    ActionNode,
    ActionStatus,
    Constraint,
    ConstraintKind,
    ControllerState,
    Observation,
)
from metamorpher.policy import HeuristicLookaheadPolicy


def graph_fixture() -> TypedActionGraph:
    graph = TypedActionGraph()
    graph.add_node(ActionNode("A", "A", decision_value=1.0))
    graph.add_node(ActionNode("B", "B", decision_value=4.0))
    graph.add_node(ActionNode("C", "C", decision_value=2.0))
    graph.add_constraint(
        Constraint("a_before_b", ConstraintKind.HARD_PREREQUISITE, ("A",), "B")
    )
    graph.add_constraint(
        Constraint(
            "c_guard",
            ConstraintKind.GUARD,
            (),
            "C",
            fact_key="c_ready",
            expected_value=True,
        )
    )
    graph.validate()
    return graph


class BatchCompilerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = graph_fixture()
        self.compiler = GraphBatchCompiler(self.graph)
        self.first_state = ControllerState()
        self.first_ledger = EvidenceLedger()
        self.second_state = ControllerState({"A": ActionStatus.COMPLETED})
        self.second_ledger = EvidenceLedger()
        self.second_ledger.append(Observation("ready", "c_ready", True, source="sensor"))

    def test_reference_batch_matches_symbolic_policy(self) -> None:
        states = (self.first_state, self.second_state)
        ledgers = (self.first_ledger, self.second_ledger)
        result = self.compiler.run(states, ledgers, ReferenceBackend())
        expected = []
        policy = HeuristicLookaheadPolicy()
        for state, ledger in zip(states, ledgers):
            frontier = self.graph.frontier(state, ledger)
            expected.append(policy.select(self.graph, state, frontier.certified))
        self.assertEqual(result.selected_action_ids, tuple(expected))
        self.assertEqual(result.refinement[0], ("C",))
        self.assertEqual(result.refinement[1], ())

    @unittest.skipUnless(NumPyBackend().is_available(), "NumPy optional backend unavailable")
    def test_numpy_batch_matches_reference_batch(self) -> None:
        states = (self.first_state, self.second_state)
        ledgers = (self.first_ledger, self.second_ledger)
        reference = self.compiler.run(states, ledgers, ReferenceBackend())
        accelerated = self.compiler.run(states, ledgers, NumPyBackend())
        self.assertEqual(accelerated.action_ids, reference.action_ids)
        self.assertEqual(accelerated.selected_action_ids, reference.selected_action_ids)
        self.assertEqual(accelerated.refinement, reference.refinement)


class ForgedBackend(AccelerationBackend):
    """Adversarial accelerator used to verify the control-plane firewall."""

    name = "forged"

    def __init__(self, selected):
        self.forged_selected = selected

    def hard_frontier_mask(self, pending, completed, hard_prerequisites):
        return [[True for _ in row] for row in pending]

    def fused_scores(self, frontier, features, weights, bias=None):
        return [[999.0 for _ in row] for row in frontier]

    def select(self, scores):
        return self.forged_selected

    def frontier_and_score(
        self,
        pending,
        completed,
        hard_prerequisites,
        features,
        weights,
        bias=None,
    ):
        action_count = len(pending[0])
        return BatchedScoreResult(
            frontier=[[True] * action_count for _ in pending],
            scores=[[999.0] * action_count for _ in pending],
            selected=self.forged_selected,
        )


class BatchBackendFirewallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.compiler = GraphBatchCompiler(graph_fixture())
        self.state = ControllerState()
        self.ledger = EvidenceLedger()

    def test_backend_cannot_select_symbolically_blocked_action(self) -> None:
        # Sorted actions are A, B, C. At this state B has an unfinished hard
        # prerequisite and C has an unknown guard; only A is symbolically legal.
        with self.assertRaises(ValueError):
            self.compiler.run((self.state,), (self.ledger,), ForgedBackend([1]))

    def test_backend_out_of_range_index_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            self.compiler.run((self.state,), (self.ledger,), ForgedBackend([999]))

    def test_backend_must_return_one_selection_per_batch_item(self) -> None:
        with self.assertRaises(ValueError):
            self.compiler.run((self.state,), (self.ledger,), ForgedBackend([]))


if __name__ == "__main__":
    unittest.main()
