"""Compile auditable controller state into accelerated numerical batches.

This is the narrow bridge between the symbolic control plane and the optional
NumPy/CUDA/Triton data plane.  Evidence and typed constraints are resolved in
Python first; a backend receives only booleans and numeric action features.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from .backends.base import AccelerationBackend, BatchedScoreResult, ReferenceBackend
from .evidence import EvidenceLedger
from .graph import TypedActionGraph
from .model import (
    ActionStatus,
    ClaimStatus,
    ClaimTier,
    ConstraintKind,
    ControllerState,
    TruthValue,
)
from .policy import HeuristicLookaheadPolicy


@dataclass(frozen=True, slots=True)
class CompiledBatch:
    action_ids: tuple[str, ...]
    pending: list[list[bool]]
    completed: list[list[bool]]
    hard_prerequisites: list[list[list[bool]]]
    features: list[list[list[float]]]
    weights: tuple[float, ...]
    refinement: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class BatchDecisionResult:
    action_ids: tuple[str, ...]
    selected_action_ids: tuple[str | None, ...]
    refinement: tuple[tuple[str, ...], ...]
    backend_result: BatchedScoreResult


class GraphBatchCompiler:
    """Compile many states of one graph for frontier/scoring acceleration."""

    def __init__(
        self,
        graph: TypedActionGraph,
        policy: HeuristicLookaheadPolicy | None = None,
    ) -> None:
        self.graph = graph
        self.policy = policy or HeuristicLookaheadPolicy()

    def compile(
        self,
        states: Sequence[ControllerState],
        ledgers: Sequence[EvidenceLedger] | None = None,
    ) -> CompiledBatch:
        if not states:
            raise ValueError("states must contain at least one controller state")
        evidence = list(ledgers or (EvidenceLedger() for _ in states))
        if len(evidence) != len(states):
            raise ValueError("states and ledgers must have the same length")

        action_ids = tuple(sorted(self.graph.nodes))
        index = {action_id: i for i, action_id in enumerate(action_ids)}
        n = len(action_ids)
        if n == 0:
            raise ValueError("cannot compile an empty graph")

        pending_batch: list[list[bool]] = []
        completed_batch: list[list[bool]] = []
        prerequisites_batch: list[list[list[bool]]] = []
        features_batch: list[list[list[float]]] = []
        refinement_batch: list[tuple[str, ...]] = []

        for state, ledger in zip(states, evidence):
            pending = [state.status_of(x) == ActionStatus.PENDING for x in action_ids]
            completed = [state.completed(x) for x in action_ids]
            prerequisites = [[False] * n for _ in range(n)]
            refinement: set[str] = set()

            for constraint in self.graph.constraints.values():
                target = index[constraint.target]
                if constraint.kind == ConstraintKind.SOFT_EPISTEMIC:
                    continue
                encodable = (
                    constraint.kind
                    in {ConstraintKind.HARD_PREREQUISITE, ConstraintKind.SAFETY}
                    and constraint.status == ClaimStatus.CONFIRMED
                    and constraint.tier != ClaimTier.CANDIDATE
                )
                if encodable:
                    for source in constraint.sources:
                        prerequisites[target][index[source]] = True
                    continue

                resolved = self.graph.evaluate_constraint(constraint, state, ledger)
                if resolved != TruthValue.SATISFIED:
                    pending[target] = False
                if resolved == TruthValue.UNKNOWN:
                    refinement.add(constraint.target)

            feature_rows: list[list[float]] = []
            for action_id in action_ids:
                node = self.graph.nodes[action_id]
                unlock = sum(
                    self.graph.nodes[descendant].decision_value
                    for descendant in self.graph.descendants(action_id)
                    if state.status_of(descendant) == ActionStatus.PENDING
                )
                feature_rows.append(
                    [
                        float(node.information_value),
                        float(node.decision_value),
                        float(node.cost),
                        float(node.harm),
                        float(node.delay),
                        float(self.graph.soft_bonus(action_id, state)),
                        float(unlock),
                    ]
                )

            pending_batch.append(pending)
            completed_batch.append(completed)
            prerequisites_batch.append(prerequisites)
            features_batch.append(feature_rows)
            refinement_batch.append(tuple(sorted(refinement)))

        p = self.policy
        weights = (
            float(p.information_weight),
            float(p.decision_weight),
            -float(p.cost_weight),
            -float(p.harm_weight),
            -float(p.delay_weight),
            float(p.soft_weight),
            float(p.unlock_weight),
        )
        return CompiledBatch(
            action_ids,
            pending_batch,
            completed_batch,
            prerequisites_batch,
            features_batch,
            weights,
            tuple(refinement_batch),
        )

    def run(
        self,
        states: Sequence[ControllerState],
        ledgers: Sequence[EvidenceLedger] | None = None,
        backend: AccelerationBackend | None = None,
    ) -> BatchDecisionResult:
        compiled = self.compile(states, ledgers)
        engine = backend or ReferenceBackend()
        result = engine.frontier_and_score(
            compiled.pending,
            compiled.completed,
            compiled.hard_prerequisites,
            compiled.features,
            compiled.weights,
        )
        frontier = self._host_matrix(
            result.frontier,
            "backend frontier",
            len(states),
            len(compiled.action_ids),
        )
        scores = self._host_matrix(
            result.scores,
            "backend scores",
            len(states),
            len(compiled.action_ids),
        )
        raw_selected = self._host(result.selected)
        if not isinstance(raw_selected, (list, tuple)):
            raise TypeError("backend selected must have shape [batch]")
        if len(raw_selected) != len(states):
            raise ValueError("backend selected length must equal the batch size")

        normalized: list[int] = []
        action_count = len(compiled.action_ids)
        for batch_index, raw_index in enumerate(raw_selected):
            if isinstance(raw_index, bool) or not isinstance(raw_index, Integral):
                raise TypeError("backend selected values must be integer indices")
            selected_index = int(raw_index)
            if selected_index < -1 or selected_index >= action_count:
                raise ValueError(
                    f"backend selected index out of range: {selected_index}"
                )
            if selected_index == -1:
                normalized.append(-1)
                continue

            frontier_value = frontier[batch_index][selected_index]
            if frontier_value not in (False, True, 0, 1):
                raise ValueError("backend frontier must contain boolean/0/1 values")
            score = scores[batch_index][selected_index]
            if (
                isinstance(score, bool)
                or not isinstance(score, Real)
                or not math.isfinite(float(score))
            ):
                raise ValueError("backend selected an action with a non-finite score")
            if not bool(frontier_value):
                raise ValueError("backend selected an action outside its frontier")

            # Do not trust the accelerated mask as execution authority.  Check
            # the chosen cell against the already-resolved symbolic inputs in
            # O(actions) per case.  This retains batching speed without
            # recomputing the entire O(actions²) frontier on CPU.
            if (
                not compiled.pending[batch_index][selected_index]
                or compiled.completed[batch_index][selected_index]
                or any(
                    required and not compiled.completed[batch_index][source]
                    for source, required in enumerate(
                        compiled.hard_prerequisites[batch_index][selected_index]
                    )
                )
            ):
                raise ValueError(
                    "backend selected an action blocked by symbolic prerequisites"
                )
            normalized.append(selected_index)

        selected = tuple(
            None if index < 0 else compiled.action_ids[index]
            for index in normalized
        )
        return BatchDecisionResult(
            compiled.action_ids,
            selected,
            compiled.refinement,
            result,
        )

    @staticmethod
    def _host(value: Any) -> Any:
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        return value

    @classmethod
    def _host_matrix(
        cls,
        value: Any,
        name: str,
        rows: int,
        columns: int,
    ) -> list[list[Any]]:
        raw = cls._host(value)
        if not isinstance(raw, (list, tuple)) or len(raw) != rows:
            raise ValueError(f"{name} must have shape [batch, action]")
        matrix: list[list[Any]] = []
        for row in raw:
            if not isinstance(row, (list, tuple)) or len(row) != columns:
                raise ValueError(f"{name} must have shape [batch, action]")
            matrix.append(list(row))
        return matrix


__all__ = ["BatchDecisionResult", "CompiledBatch", "GraphBatchCompiler"]
