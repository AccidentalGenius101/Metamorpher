from __future__ import annotations

import math
import unittest

from _support import graph_snapshot

from metamorpher.evidence import EvidenceLedger
from metamorpher.graph import TypedActionGraph
from metamorpher.model import (
    ActionKind,
    ActionNode,
    ActionStatus,
    ClaimStatus,
    Constraint,
    ConstraintKind,
    ControllerState,
    InvalidGraphError,
    Observation,
)
from metamorpher.policy import HeuristicLookaheadPolicy


def node(node_id: str, **values) -> ActionNode:
    return ActionNode(node_id, node_id, **values)


def hard(edge_id: str, sources: tuple[str, ...], target: str) -> Constraint:
    return Constraint(edge_id, ConstraintKind.HARD_PREREQUISITE, sources, target)


class TypedFrontierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = EvidenceLedger()

    def diamond(self) -> TypedActionGraph:
        graph = TypedActionGraph()
        for action_id in "ABCD":
            graph.add_node(node(action_id))
        graph.add_constraint(hard("a_b", ("A",), "B"))
        graph.add_constraint(hard("a_c", ("A",), "C"))
        graph.add_constraint(hard("bc_d", ("B", "C"), "D"))
        graph.validate()
        return graph

    def assert_frontier(self, graph, completed, expected_certified, expected_blocked=()):
        state = ControllerState({x: ActionStatus.COMPLETED for x in completed})
        result = graph.frontier(state, self.ledger)
        self.assertEqual(set(result.certified), set(expected_certified))
        self.assertEqual(set(result.blocked), set(expected_blocked))
        return result

    def test_frontier_requires_all_hard_parents(self) -> None:
        graph = self.diamond()
        self.assert_frontier(graph, (), {"A"}, {"B", "C", "D"})
        self.assert_frontier(graph, {"A"}, {"B", "C"}, {"D"})
        self.assert_frontier(graph, {"A", "B"}, {"C"}, {"D"})
        self.assert_frontier(graph, {"A", "B", "C"}, {"D"})
        self.assert_frontier(graph, {"A", "B", "C", "D"}, set())

    def test_hard_prerequisite_is_and_not_or(self) -> None:
        graph = self.diamond()
        result = self.assert_frontier(graph, {"B"}, {"A"}, {"C", "D"})
        self.assertIn("bc_d", result.violated_constraints["D"])

    def test_unknown_guard_routes_to_refinement(self) -> None:
        graph = TypedActionGraph()
        graph.add_node(node("probe", kind=ActionKind.OBSERVE))
        graph.add_node(node("repair", kind=ActionKind.REPAIR))
        graph.add_constraint(
            Constraint(
                "cold_guard",
                ConstraintKind.GUARD,
                (),
                "repair",
                fact_key="engine_cold",
                expected_value=True,
                probe_action_id="probe",
            )
        )
        state = ControllerState()
        first = graph.frontier(state, self.ledger)
        self.assertEqual(first.certified, ("probe",))
        self.assertEqual(first.refinement, ("repair",))
        self.assertEqual(first.unknown_constraints["repair"], ("cold_guard",))

        self.ledger.append(Observation("cold", "engine_cold", True, source="sensor"))
        second = graph.frontier(state, self.ledger)
        self.assertIn("repair", second.certified)
        self.assertNotIn("repair", second.refinement)

    def test_violated_guard_blocks_action(self) -> None:
        graph = TypedActionGraph()
        graph.add_node(node("repair"))
        graph.add_constraint(
            Constraint(
                "cold_guard",
                ConstraintKind.GUARD,
                (),
                "repair",
                fact_key="engine_cold",
                expected_value=True,
            )
        )
        self.ledger.append(Observation("hot", "engine_cold", False, source="sensor"))
        result = graph.frontier(ControllerState(), self.ledger)
        self.assertEqual(result.blocked, ("repair",))

    def test_soft_epistemic_relation_ranks_but_does_not_gate(self) -> None:
        graph = TypedActionGraph()
        graph.add_node(node("inspect", kind=ActionKind.OBSERVE))
        graph.add_node(node("remove", irreversible=True, reversible=False))
        graph.add_constraint(
            Constraint(
                "inspect_first",
                ConstraintKind.SOFT_EPISTEMIC,
                ("inspect",),
                "remove",
                confidence=3.0,
            )
        )
        state = ControllerState()
        frontier = graph.frontier(state, self.ledger)
        self.assertEqual(set(frontier.certified), {"inspect", "remove"})
        self.assertEqual(HeuristicLookaheadPolicy().select(graph, state, frontier.certified), "inspect")

        state.action_status["inspect"] = ActionStatus.DISABLED
        available = graph.frontier(state, self.ledger).certified
        self.assertEqual(available, ("remove",))
        self.assertEqual(HeuristicLookaheadPolicy().select(graph, state, available), "remove")

    def test_rejected_constraint_no_longer_blocks(self) -> None:
        graph = TypedActionGraph()
        graph.add_node(node("A"))
        graph.add_node(node("B"))
        graph.add_constraint(
            Constraint(
                "a_b",
                ConstraintKind.HARD_PREREQUISITE,
                ("A",),
                "B",
                status=ClaimStatus.REJECTED,
            )
        )
        result = graph.frontier(ControllerState(), self.ledger)
        self.assertEqual(set(result.certified), {"A", "B"})

    def test_cycle_transaction_is_atomic(self) -> None:
        graph = self.diamond()
        before_nodes, before_constraints, before_epoch = graph_snapshot(graph)
        transaction = graph.transaction()
        with self.assertRaises(InvalidGraphError), transaction as working:
            working.add_constraint(hard("d_a", ("D",), "A"))
            transaction.commit()
        self.assertEqual(graph.nodes, before_nodes)
        self.assertEqual(graph.constraints, before_constraints)
        self.assertEqual(
            graph.epoch,
            before_epoch,
            "a rejected staged edit must not consume a live graph epoch",
        )
        graph.validate()

    def test_uncommitted_transaction_rolls_back(self) -> None:
        graph = self.diamond()
        before_nodes, before_constraints, before_epoch = graph_snapshot(graph)
        with graph.transaction() as working:
            working.add_node(node("E"))
        self.assertEqual(graph.nodes, before_nodes)
        self.assertEqual(graph.constraints, before_constraints)
        self.assertEqual(graph.epoch, before_epoch)


class PolicyTests(unittest.TestCase):
    def test_policy_never_selects_outside_candidates(self) -> None:
        graph = TypedActionGraph()
        graph.add_node(node("legal", decision_value=1.0))
        graph.add_node(node("illegal", decision_value=1e100))
        selected = HeuristicLookaheadPolicy().select(graph, ControllerState(), ("legal",))
        self.assertEqual(selected, "legal")

    def test_nonfinite_scores_fail_closed(self) -> None:
        graph = TypedActionGraph()
        graph.add_node(node("nan", decision_value=math.nan))
        graph.add_node(node("inf", decision_value=math.inf))
        with self.assertRaises(ValueError):
            HeuristicLookaheadPolicy().select(graph, ControllerState(), ("nan", "inf"))

    def test_tie_break_is_deterministic(self) -> None:
        graph = TypedActionGraph()
        graph.add_node(node("z", decision_value=1.0))
        graph.add_node(node("a", decision_value=1.0))
        policy = HeuristicLookaheadPolicy()
        self.assertEqual(policy.select(graph, ControllerState(), ("z", "a")), "a")
        self.assertEqual(policy.select(graph, ControllerState(), ("a", "z")), "a")


if __name__ == "__main__":
    unittest.main()
