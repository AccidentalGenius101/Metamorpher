from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .graph import TypedActionGraph
from .model import ActionNode, ActionStatus, ControllerState


class DecisionPolicy(Protocol):
    def select(self, graph: TypedActionGraph, state: ControllerState, candidates: Sequence[str]) -> str: ...


@dataclass(slots=True)
class HeuristicLookaheadPolicy:
    """Transparent heuristic; not claimed to be exact Bayes-risk planning."""

    information_weight: float = 1.0
    decision_weight: float = 1.0
    cost_weight: float = 1.0
    harm_weight: float = 2.0
    delay_weight: float = 0.2
    soft_weight: float = 1.0
    unlock_weight: float = 0.15

    def score(self, graph: TypedActionGraph, state: ControllerState, node: ActionNode) -> float:
        unlock = sum(
            graph.nodes[x].decision_value
            for x in graph.descendants(node.id)
            if state.status_of(x) == ActionStatus.PENDING
        )
        value = (
            self.information_weight * node.information_value
            + self.decision_weight * node.decision_value
            + self.soft_weight * graph.soft_bonus(node.id, state)
            + self.unlock_weight * unlock
            - self.cost_weight * node.cost
            - self.harm_weight * node.harm
            - self.delay_weight * node.delay
        )
        return float(value)

    def select(self, graph: TypedActionGraph, state: ControllerState, candidates: Sequence[str]) -> str:
        if not candidates:
            raise ValueError("cannot select from an empty frontier")
        scored: list[tuple[float, str]] = []
        for node_id in candidates:
            score = self.score(graph, state, graph.nodes[node_id])
            if not math.isfinite(score):
                continue
            scored.append((score, node_id))
        if not scored:
            raise ValueError("all candidate scores were non-finite")
        # Deterministic: maximum value, then lexicographically smallest ID.
        best = max(x[0] for x in scored)
        return min(node_id for score, node_id in scored if score == best)
