from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Self

from .evidence import EvidenceLedger
from .model import (
    ActionNode,
    ActionStatus,
    ClaimStatus,
    ClaimTier,
    Constraint,
    ConstraintKind,
    ControllerState,
    InvalidGraphError,
    TruthValue,
)


@dataclass(frozen=True, slots=True)
class FrontierResult:
    certified: tuple[str, ...]
    refinement: tuple[str, ...]
    blocked: tuple[str, ...]
    unknown_constraints: dict[str, tuple[str, ...]] = field(default_factory=dict)
    violated_constraints: dict[str, tuple[str, ...]] = field(default_factory=dict)


class TypedActionGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, ActionNode] = {}
        self.constraints: dict[str, Constraint] = {}
        self.epoch = 0

    def add_node(self, node: ActionNode) -> None:
        if node.id in self.nodes:
            raise InvalidGraphError(f"duplicate node: {node.id}")
        self.nodes[node.id] = node

    def add_constraint(self, constraint: Constraint) -> None:
        if constraint.id in self.constraints:
            raise InvalidGraphError(f"duplicate constraint: {constraint.id}")
        if constraint.target not in self.nodes:
            raise InvalidGraphError(f"unknown target: {constraint.target}")
        missing = [x for x in constraint.sources if x not in self.nodes]
        if missing:
            raise InvalidGraphError(f"unknown sources: {missing}")
        self.constraints[constraint.id] = constraint

    def replace_constraint(self, constraint: Constraint) -> None:
        """Replace a revisable claim while preserving immutable safety policy."""
        previous = self.constraints.get(constraint.id)
        if previous is None:
            raise InvalidGraphError(f"unknown constraint: {constraint.id}")
        if previous.externally_governed:
            raise InvalidGraphError(f"externally governed constraint is immutable: {constraint.id}")
        if constraint.id != previous.id:
            raise InvalidGraphError("replacement constraint ID must not change")
        if constraint.target not in self.nodes:
            raise InvalidGraphError(f"unknown target: {constraint.target}")
        missing = [x for x in constraint.sources if x not in self.nodes]
        if missing:
            raise InvalidGraphError(f"unknown sources: {missing}")
        self.constraints[constraint.id] = constraint

    def remove_constraint(self, constraint_id: str) -> None:
        previous = self.constraints.get(constraint_id)
        if previous is None:
            return
        if previous.externally_governed:
            raise InvalidGraphError(f"externally governed constraint is immutable: {constraint_id}")
        del self.constraints[constraint_id]

    def constraints_for(self, target: str) -> tuple[Constraint, ...]:
        return tuple(x for x in self.constraints.values() if x.target == target)

    def evaluate_constraint(self, c: Constraint, state: ControllerState, evidence: EvidenceLedger) -> TruthValue:
        if c.kind == ConstraintKind.SOFT_EPISTEMIC:
            return TruthValue.SATISFIED
        if c.status == ClaimStatus.REJECTED:
            return TruthValue.SATISFIED
        if c.tier == ClaimTier.CANDIDATE:
            # Candidate structure is quarantined: it cannot silently become a
            # hard gate.  If its proposed sources are already resolved it is
            # harmless; otherwise the correct result is refinement, not false.
            return (
                TruthValue.SATISFIED
                if c.sources and all(state.completed(x) for x in c.sources)
                else TruthValue.UNKNOWN
            )
        if c.kind == ConstraintKind.GUARD:
            if c.fact_key is None:
                return TruthValue.UNKNOWN
            return evidence.evaluate(c.fact_key, c.expected_value).state
        if c.kind == ConstraintKind.MUTEX:
            return TruthValue.VIOLATED if any(state.completed(x) for x in c.sources) else TruthValue.SATISFIED
        if c.status == ClaimStatus.UNKNOWN and not all(state.completed(x) for x in c.sources):
            return TruthValue.UNKNOWN
        if c.kind == ConstraintKind.ALTERNATIVE:
            return TruthValue.SATISFIED if any(state.completed(x) for x in c.sources) else TruthValue.VIOLATED
        return TruthValue.SATISFIED if all(state.completed(x) for x in c.sources) else TruthValue.VIOLATED

    def frontier(self, state: ControllerState, evidence: EvidenceLedger) -> FrontierResult:
        certified: list[str] = []
        refinement: list[str] = []
        blocked: list[str] = []
        unknown: dict[str, tuple[str, ...]] = {}
        violated: dict[str, tuple[str, ...]] = {}
        for node_id in sorted(self.nodes):
            if state.status_of(node_id) != ActionStatus.PENDING:
                continue
            cs = self.constraints_for(node_id)
            unknown_ids: list[str] = []
            violated_ids: list[str] = []
            for c in cs:
                result = self.evaluate_constraint(c, state, evidence)
                if result == TruthValue.UNKNOWN:
                    unknown_ids.append(c.id)
                elif result == TruthValue.VIOLATED:
                    violated_ids.append(c.id)
            if violated_ids:
                blocked.append(node_id)
                violated[node_id] = tuple(violated_ids)
            elif unknown_ids:
                refinement.append(node_id)
                unknown[node_id] = tuple(unknown_ids)
            else:
                certified.append(node_id)
        return FrontierResult(tuple(certified), tuple(refinement), tuple(blocked), unknown, violated)

    def soft_bonus(self, action_id: str, state: ControllerState) -> float:
        bonus = 0.0
        for c in self.constraints.values():
            if c.kind != ConstraintKind.SOFT_EPISTEMIC or c.status == ClaimStatus.REJECTED:
                continue
            if action_id in c.sources and state.status_of(c.target) == ActionStatus.PENDING:
                bonus += max(0.0, c.confidence)
        return bonus

    def descendants(self, action_id: str) -> set[str]:
        children: dict[str, set[str]] = {x: set() for x in self.nodes}
        for c in self.constraints.values():
            if c.status == ClaimStatus.REJECTED or c.kind not in {
                ConstraintKind.HARD_PREREQUISITE, ConstraintKind.SAFETY,
                ConstraintKind.ALTERNATIVE, ConstraintKind.SOFT_EPISTEMIC,
            }:
                continue
            for source in c.sources:
                children[source].add(c.target)
        seen: set[str] = set()
        stack = list(children.get(action_id, ()))
        while stack:
            item = stack.pop()
            if item in seen:
                continue
            seen.add(item)
            stack.extend(children.get(item, ()))
        return seen

    def validate(self) -> None:
        # Every non-rejected structural relation participates in cycle checks.
        children: dict[str, set[str]] = {x: set() for x in self.nodes}
        indegree = {x: 0 for x in self.nodes}
        for c in self.constraints.values():
            if c.status == ClaimStatus.REJECTED or c.kind in {ConstraintKind.GUARD, ConstraintKind.MUTEX}:
                continue
            for source in c.sources:
                if c.target not in children[source]:
                    children[source].add(c.target)
                    indegree[c.target] += 1
        ready = sorted(x for x, n in indegree.items() if n == 0)
        visited = 0
        while ready:
            node = ready.pop(0)
            visited += 1
            for child in sorted(children[node]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort()
        if visited != len(self.nodes):
            raise InvalidGraphError("constraint revision would create a cycle")

    def transaction(self) -> GraphTransaction:
        return GraphTransaction(self)


class GraphTransaction:
    def __init__(self, graph: TypedActionGraph) -> None:
        self.graph = graph
        # Revisions are staged on a private graph.  The live graph is never
        # exposed to a half-applied edit, even if validation raises.
        self.staged = TypedActionGraph()
        self.staged.nodes = copy.deepcopy(graph.nodes)
        self.staged.constraints = copy.deepcopy(graph.constraints)
        self.staged.epoch = graph.epoch
        self._committed = False

    def __enter__(self) -> Self:
        return self

    def __getattr__(self, name: str):
        # Delegate graph editing and inspection methods to the private copy.
        return getattr(self.staged, name)

    def commit(self) -> None:
        if self._committed:
            raise InvalidGraphError("transaction already closed")
        self.staged.validate()
        self.graph.nodes = self.staged.nodes
        self.graph.constraints = self.staged.constraints
        self.graph.epoch += 1
        self._committed = True

    def rollback(self) -> None:
        # A staged transaction has not touched the live graph, so abandoning it
        # is an exact no-op.  Rollback of a *committed controller snapshot* is a
        # separate operation and must advance the live epoch.
        self._committed = True

    def __exit__(self, exc_type, exc, tb) -> bool:
        # An uncommitted or failed edit is simply discarded.
        return False
