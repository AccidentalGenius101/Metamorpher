"""Composable learning boundaries around the Metamorpher control kernel.

The objects in this module do not make perception or structure learning
trustworthy by declaration.  They make their authority explicit: perception
produces provenance-bearing observations, proposers create quarantined
candidate structure, and only the controller may apply evidence-grounded
revisions while completing a committed action.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .adapters import ActionExecutor, ExecutionResult
from .carving import ConstraintRevision
from .controller import MetamorpherController, ObservationReceipt
from .model import (
    ActionNode,
    ActionStatus,
    ClaimTier,
    Constraint,
    Decision,
    DecisionStatus,
    DomainTag,
    InvalidGraphError,
    Observation,
)
from .version_space import Hypothesis, UnresolvedCell


@dataclass(frozen=True, slots=True)
class Perception:
    """A lossy, attributable interpretation of raw external input."""

    observations: tuple[Observation, ...]
    representation: Mapping[str, Any]
    source: str


class Perceiver(Protocol):
    """Converts raw input into observations without exposing hidden truth."""

    def perceive(
        self,
        raw_input: Any,
        domain: DomainTag,
        *,
        action_token: str | None = None,
    ) -> Perception: ...


@dataclass(frozen=True, slots=True)
class CandidateStructure:
    """Untrusted structure proposed for quarantine in the candidate graph."""

    nodes: tuple[ActionNode, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    rationale: str = ""


class Proposer(Protocol):
    """Uses observations and memory to imagine possible structure."""

    def propose(
        self,
        context: Mapping[str, Any],
        observations: Sequence[Observation],
        domain: DomainTag,
    ) -> CandidateStructure: ...


@dataclass(frozen=True, slots=True)
class RefinementProposal:
    """A represented probe intended to separate surviving hypotheses."""

    probe: ActionNode
    constraints: tuple[Constraint, ...] = ()
    targets: tuple[str, ...] = ()
    rationale: str = ""


class Discriminator(Protocol):
    """Proposes an informative probe for an unresolved equivalence class."""

    def discriminate(
        self,
        cell: UnresolvedCell,
        context: Mapping[str, Any],
        domain: DomainTag,
    ) -> RefinementProposal | None: ...


class StructureLearner(Protocol):
    """Proposes evidence-grounded graph edits after an external outcome."""

    def revise(
        self,
        controller: MetamorpherController,
        decision: Decision,
        result: ExecutionResult,
    ) -> Sequence[ConstraintRevision]: ...


@dataclass(frozen=True, slots=True)
class StructuralCapsule:
    """Domain-bounded reusable structure with explicit provenance."""

    id: str
    domain: DomainTag
    constraints: tuple[Constraint, ...] = ()
    hypotheses: tuple[Hypothesis, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    parent_ids: tuple[str, ...] = ()


class CapsuleStore(Protocol):
    def put(self, capsule: StructuralCapsule) -> None: ...

    def get(self, capsule_id: str) -> StructuralCapsule | None: ...

    def candidates(self, domain: DomainTag) -> Sequence[StructuralCapsule]: ...


class InMemoryCapsuleStore:
    """Small reference store; persistence providers can implement CapsuleStore."""

    def __init__(self) -> None:
        self._capsules: dict[str, StructuralCapsule] = {}

    def put(self, capsule: StructuralCapsule) -> None:
        if not capsule.id:
            raise ValueError("capsule ID cannot be empty")
        self._capsules[capsule.id] = copy.deepcopy(capsule)

    def get(self, capsule_id: str) -> StructuralCapsule | None:
        capsule = self._capsules.get(capsule_id)
        return copy.deepcopy(capsule) if capsule is not None else None

    def candidates(self, domain: DomainTag) -> tuple[StructuralCapsule, ...]:
        return tuple(
            copy.deepcopy(capsule)
            for _, capsule in sorted(self._capsules.items())
            if capsule.domain == domain
        )


@dataclass(frozen=True, slots=True)
class CognitiveStep:
    decision: Decision
    result: ExecutionResult | None
    receipt: ObservationReceipt | None


class CognitiveLoop:
    """Minimal outer loop that composes learning without bypassing control."""

    def __init__(
        self,
        controller: MetamorpherController,
        *,
        executor: ActionExecutor,
        learner: StructureLearner | None = None,
        capsules: CapsuleStore | None = None,
    ) -> None:
        self.controller = controller
        self.executor = executor
        self.learner = learner
        self.capsules = capsules or InMemoryCapsuleStore()

    def install_candidates(self, proposal: CandidateStructure) -> int:
        """Atomically install novel nodes and quarantined candidate constraints."""

        if not proposal.rationale.strip():
            raise ValueError("candidate structure requires a rationale")
        existing_nodes = set(self.controller.graph.nodes)
        proposed_node_ids = [node.id for node in proposal.nodes]
        if len(proposed_node_ids) != len(set(proposed_node_ids)):
            raise InvalidGraphError("candidate proposal contains duplicate nodes")
        collisions = sorted(existing_nodes.intersection(proposed_node_ids))
        if collisions:
            raise InvalidGraphError(f"candidate nodes already exist: {collisions}")
        for constraint in proposal.constraints:
            if constraint.tier != ClaimTier.CANDIDATE:
                raise InvalidGraphError(
                    "proposed constraints must enter with candidate tier: "
                    + constraint.id
                )
            if constraint.id in self.controller.graph.constraints:
                raise InvalidGraphError(
                    f"candidate constraint already exists: {constraint.id}"
                )

        tx = self.controller.graph.transaction()
        with tx as staged:
            for node in proposal.nodes:
                staged.add_node(copy.deepcopy(node))
            for constraint in proposal.constraints:
                staged.add_constraint(copy.deepcopy(constraint))
            tx.commit()

        if proposal.hypotheses:
            cell_id = f"candidate-cell-{self.controller.graph.epoch}"
            self.controller.version_space.add(
                UnresolvedCell(
                    cell_id,
                    {item.id: copy.deepcopy(item) for item in proposal.hypotheses},
                )
            )
        return self.controller.graph.epoch

    def step(self, domain: DomainTag | None = None) -> CognitiveStep:
        """Decide, commit, execute externally, then atomically learn/observe."""

        decision = self.controller.next(domain)
        if decision.status == DecisionStatus.ABSTAIN:
            return CognitiveStep(decision, None, None)

        node = self.controller.commit(decision)
        result = self.executor.execute(decision)
        if result.action_id != node.id:
            raise ValueError(
                f"executor returned {result.action_id!r} for committed action {node.id!r}"
            )
        observations = tuple(result.observations)
        if not observations:
            raise ValueError("executor must return at least one observation")
        normalized = tuple(
            observation
            if observation.action_token is not None
            else Observation(
                id=observation.id,
                key=observation.key,
                value=observation.value,
                status=observation.status,
                source=observation.source,
                reliability=observation.reliability,
                domain=observation.domain,
                action_token=decision.token,
                timestamp=observation.timestamp,
                censoring_reason=observation.censoring_reason,
                independent_audit=observation.independent_audit,
            )
            for observation in observations
        )
        normalized_result = ExecutionResult(
            action_id=result.action_id,
            succeeded=result.succeeded,
            observations=normalized,
            message=result.message,
            external_reference=result.external_reference,
        )
        revisions = (
            tuple(self.learner.revise(self.controller, decision, normalized_result))
            if self.learner is not None
            else ()
        )
        receipt = self.controller.observe_many(
            normalized,
            token=decision.token,
            action_status=(
                ActionStatus.COMPLETED if result.succeeded else ActionStatus.FAILED
            ),
            revisions=revisions,
        )
        return CognitiveStep(decision, normalized_result, receipt)
