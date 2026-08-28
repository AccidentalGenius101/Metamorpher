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

from .adapters import ActionExecutor, ExecutionResult, ObservationSource
from .audit import AuditPolicy
from .carving import ConstraintRevision
from .controller import MetamorpherController, ObservationReceipt
from .model import (
    ActionNode,
    ActionStatus,
    ClaimTier,
    ClassStatus,
    Constraint,
    Decision,
    DecisionStatus,
    DomainTag,
    InvalidGraphError,
    Observation,
    ObservationStatus,
)
from .version_space import Hypothesis, UnresolvedCell


def _bind_observation(
    item: Observation,
    domain: DomainTag,
    *,
    token: str | None = None,
    independent_audit: bool = False,
    source: str | None = None,
) -> Observation:
    return Observation(
        item.id,
        item.key,
        item.value,
        item.status,
        item.source or source or "unknown",
        item.reliability,
        item.domain or domain,
        item.action_token or token,
        item.timestamp,
        item.censoring_reason,
        independent_audit,
    )


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
    audit_receipt: ObservationReceipt | None = None


@dataclass(frozen=True, slots=True)
class CognitiveIngestion:
    perception: Perception
    proposal: CandidateStructure | None
    graph_epoch: int


class CognitiveLoop:
    """Minimal outer loop that composes learning without bypassing control."""

    def __init__(
        self,
        controller: MetamorpherController,
        *,
        executor: ActionExecutor,
        learner: StructureLearner | None = None,
        capsules: CapsuleStore | None = None,
        perceiver: Perceiver | None = None,
        proposer: Proposer | None = None,
        discriminator: Discriminator | None = None,
        audit_policy: AuditPolicy | None = None,
        auditor: ObservationSource | None = None,
    ) -> None:
        self.controller = controller
        self.executor = executor
        self.learner = learner
        self.capsules = capsules or InMemoryCapsuleStore()
        self.perceiver = perceiver
        self.proposer = proposer
        self.discriminator = discriminator
        self.audit_policy = audit_policy
        self.auditor = auditor
        self._case_index = 0

    def ingest(
        self,
        raw_input: Any,
        domain: DomainTag,
        context: Mapping[str, Any] | None = None,
    ) -> CognitiveIngestion:
        """Perceive raw input, record evidence, and quarantine proposed structure."""

        if self.perceiver is None:
            raise RuntimeError("cognitive ingestion requires a perceiver")
        perception = self.perceiver.perceive(raw_input, domain)
        observations = tuple(
            _bind_observation(
                item, domain, independent_audit=True, source=perception.source
            )
            for item in perception.observations
        )
        self.controller.observe_many(observations)
        proposal = None
        if self.proposer is not None:
            proposal_context = {
                **dict(context or {}),
                "representation": dict(perception.representation),
                "capsules": self.capsules.candidates(domain),
                "memory": self.controller.memory.for_domain(domain),
            }
            proposal = self.proposer.propose(proposal_context, observations, domain)
            self.install_candidates(proposal)
        return CognitiveIngestion(perception, proposal, self.controller.graph.epoch)

    def capture_capsule(
        self,
        capsule_id: str,
        domain: DomainTag,
        *,
        constraint_ids: Sequence[str] = (),
        evidence_ids: Sequence[str] = (),
        parent_ids: Sequence[str] = (),
    ) -> StructuralCapsule:
        """Persist only active, non-candidate, evidence-backed structure."""

        cell = self.controller.version_space.active_for(domain)
        constraints = tuple(
            self.controller.graph.constraints[item] for item in constraint_ids
        )
        if any(item.tier == ClaimTier.CANDIDATE for item in constraints):
            raise ValueError("candidate constraints cannot enter a capsule")
        evidence = {item.id: item for item in self.controller.evidence.events}
        if set(evidence_ids) - evidence.keys():
            raise ValueError("capsule references unknown evidence")
        if any(evidence[item].domain != domain for item in evidence_ids):
            raise ValueError("capsule evidence does not match its domain")
        if cell is None or not evidence_ids:
            raise ValueError("capsule requires active hypotheses and evidence")
        capsule = StructuralCapsule(
            capsule_id,
            domain,
            constraints,
            tuple(cell.hypotheses.values()),
            tuple(sorted(set(evidence_ids))),
            tuple(parent_ids),
        )
        self.capsules.put(capsule)
        self.controller.trace.append("capsule_captured", capsule_id=capsule_id)
        return capsule

    def request_refinement(
        self,
        domain: DomainTag,
        context: Mapping[str, Any] | None = None,
    ) -> RefinementProposal | None:
        """Quarantine a discriminator's probe and corresponding safe hypotheses."""

        if self.discriminator is None:
            raise RuntimeError("refinement requires a discriminator")
        cell = self.controller.version_space.active_for(domain)
        if cell is None:
            return None
        refinement = self.discriminator.discriminate(cell, context or {}, domain)
        if refinement is None:
            return None
        unknown_targets = sorted(set(refinement.targets) - cell.hypotheses.keys())
        if unknown_targets:
            raise InvalidGraphError(
                f"refinement names unknown hypotheses: {unknown_targets}"
            )
        hypotheses = tuple(
            Hypothesis(
                item.id,
                item.safe_actions.union({refinement.probe.id}),
                item.predictions,
                item.domain,
                item.provenance,
            )
            for item in cell.hypotheses.values()
        )
        self.install_candidates(
            CandidateStructure(
                nodes=(refinement.probe,),
                constraints=refinement.constraints,
                hypotheses=hypotheses,
                rationale=refinement.rationale,
            )
        )
        return refinement

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

        hypothesis_ids = [item.id for item in proposal.hypotheses]
        if len(hypothesis_ids) != len(set(hypothesis_ids)):
            raise InvalidGraphError("candidate proposal contains duplicate hypotheses")
        available_actions = existing_nodes.union(proposed_node_ids)
        for hypothesis in proposal.hypotheses:
            unknown_actions = sorted(hypothesis.safe_actions - available_actions)
            if unknown_actions:
                raise InvalidGraphError(
                    f"candidate hypothesis {hypothesis.id!r} names unknown actions: "
                    f"{unknown_actions}"
                )
            if hypothesis.domain is None:
                raise InvalidGraphError(
                    f"candidate hypothesis {hypothesis.id!r} requires a domain"
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

    def promote_candidate_cell(
        self,
        cell_id: str,
        evidence_ids: Sequence[str],
        reason: str,
    ) -> UnresolvedCell:
        """Activate quarantined hypotheses only with observed domain evidence."""

        if not reason.strip():
            raise ValueError("candidate promotion requires a reason")
        cell = self.controller.version_space.cells.get(cell_id)
        if cell is None or not cell_id.startswith("candidate-cell-"):
            raise KeyError(f"unknown candidate cell: {cell_id}")
        if not evidence_ids:
            raise ValueError("candidate promotion requires evidence")
        records = {
            item.id: item
            for item in self.controller.evidence.events
            if item.status in {ObservationStatus.OBSERVED, ObservationStatus.INFERRED}
        }
        missing = tuple(sorted(set(evidence_ids) - records.keys()))
        if missing:
            raise ValueError(f"unknown or unusable promotion evidence: {missing}")
        domains = {hypothesis.domain for hypothesis in cell.hypotheses.values()}
        evidence_domains = {records[item].domain for item in evidence_ids}
        if len(domains) != 1 or evidence_domains != domains:
            raise ValueError("promotion evidence does not match candidate cell domain")
        cell.status = ClassStatus.SUPPORTED
        cell.evidence_ids.update(evidence_ids)
        self.controller.version_space.activate(cell_id)
        self.controller.trace.append(
            "candidate_cell_promoted",
            cell_id=cell_id,
            evidence_ids=tuple(sorted(set(evidence_ids))),
            reason=reason,
        )
        return cell

    def step(self, domain: DomainTag | None = None) -> CognitiveStep:
        """Decide, commit, execute externally, then atomically learn/observe."""

        decision = self.controller.next(domain)
        if decision.status == DecisionStatus.ABSTAIN:
            return CognitiveStep(decision, None, None)

        node = self.controller.commit(decision)
        try:
            result = self.executor.execute(decision)
        except Exception as exc:
            self.controller.fail_committed(
                f"executor raised {type(exc).__name__}: {exc}"
            )
            raise
        if result.action_id != node.id:
            self.controller.fail_committed(
                f"executor attributed result to {result.action_id!r}, expected {node.id!r}"
            )
            raise ValueError(
                f"executor returned {result.action_id!r} for committed action {node.id!r}"
            )
        observations = tuple(result.observations)
        if not observations:
            self.controller.fail_committed("executor returned no observations")
            raise ValueError("executor must return at least one observation")
        normalized = tuple(
            _bind_observation(
                observation,
                decision.domain,
                token=decision.token,
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
        resulting_status = (
            ActionStatus.COMPLETED if result.succeeded else ActionStatus.FAILED
        )
        try:
            revisions = (
                tuple(self.learner.revise(self.controller, decision, normalized_result))
                if self.learner is not None
                else ()
            )
        except Exception:
            self.controller.observe_many(
                normalized,
                token=decision.token,
                action_status=resulting_status,
            )
            raise
        try:
            receipt = self.controller.observe_many(
                normalized,
                token=decision.token,
                action_status=resulting_status,
                revisions=revisions,
            )
        except Exception as primary:
            # Invalid proposed revisions must not strand a real external
            # outcome. Retry the observation without structural authority.
            if revisions:
                try:
                    receipt = self.controller.observe_many(
                        normalized,
                        token=decision.token,
                        action_status=resulting_status,
                    )
                except Exception:
                    self.controller.fail_committed(
                        f"observation ingestion raised {type(primary).__name__}: {primary}"
                    )
                    raise primary
            else:
                self.controller.fail_committed(
                    f"observation ingestion raised {type(primary).__name__}: {primary}"
                )
                raise
        self._case_index += 1
        audit_receipt = None
        if (
            self.audit_policy is not None
            and self.auditor is not None
            and self.audit_policy.consume(self._case_index)
        ):
            audits = tuple(
                _bind_observation(item, decision.domain, independent_audit=True)
                for item in self.auditor.collect(node, decision.domain)
            )
            if audits:
                audit_receipt = self.controller.observe_many(audits)
        return CognitiveStep(decision, normalized_result, receipt, audit_receipt)
