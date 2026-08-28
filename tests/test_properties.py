from __future__ import annotations

import random
import unittest

import _support  # noqa: F401

from metamorpher.evidence import EvidenceLedger
from metamorpher.graph import TypedActionGraph
from metamorpher.model import (
    ActionNode,
    ActionStatus,
    Constraint,
    ConstraintKind,
    ControllerState,
)
from metamorpher.policy import HeuristicLookaheadPolicy


class DeterministicGraphPropertyTests(unittest.TestCase):
    SEED = 20260828

    def build_random_graph(self, rng: random.Random, size: int):
        graph = TypedActionGraph()
        parents: dict[str, tuple[str, ...]] = {}
        for index in range(size):
            graph.add_node(
                ActionNode(
                    f"n{index}",
                    f"node {index}",
                    decision_value=rng.uniform(-3.0, 3.0),
                    information_value=rng.uniform(0.0, 2.0),
                    cost=rng.uniform(0.0, 1.0),
                )
            )
        for target in range(1, size):
            options = [f"n{i}" for i in range(target) if rng.random() < 0.22]
            selected = tuple(sorted(options))
            parents[f"n{target}"] = selected
            if selected:
                graph.add_constraint(
                    Constraint(
                        f"p{target}",
                        ConstraintKind.HARD_PREREQUISITE,
                        selected,
                        f"n{target}",
                    )
                )
        parents.setdefault("n0", ())
        graph.validate()
        return graph, parents

    def reference_frontier(self, size, parents, completed):
        return {
            f"n{i}"
            for i in range(size)
            if f"n{i}" not in completed and all(parent in completed for parent in parents.get(f"n{i}", ()))
        }

    def test_random_dags_match_reference_frontier(self) -> None:
        rng = random.Random(self.SEED)
        ledger = EvidenceLedger()
        for _ in range(100):
            size = rng.randint(2, 30)
            graph, parents = self.build_random_graph(rng, size)
            completed = {f"n{i}" for i in range(size) if rng.random() < 0.45}
            state = ControllerState({item: ActionStatus.COMPLETED for item in completed})
            actual = set(graph.frontier(state, ledger).certified)
            expected = self.reference_frontier(size, parents, completed)
            self.assertEqual(actual, expected)
            if actual:
                selected = HeuristicLookaheadPolicy().select(graph, state, tuple(actual))
                self.assertIn(selected, expected)

    def test_same_seed_produces_same_graph_and_frontier(self) -> None:
        first_rng = random.Random(self.SEED)
        second_rng = random.Random(self.SEED)
        first_graph, first_parents = self.build_random_graph(first_rng, 20)
        second_graph, second_parents = self.build_random_graph(second_rng, 20)
        self.assertEqual(first_parents, second_parents)
        self.assertEqual(first_graph.constraints, second_graph.constraints)
        self.assertEqual(
            first_graph.frontier(ControllerState(), EvidenceLedger()),
            second_graph.frontier(ControllerState(), EvidenceLedger()),
        )


if __name__ == "__main__":
    unittest.main()
