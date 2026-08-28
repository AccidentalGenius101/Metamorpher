"""The Metamorpher runtime control plane.

The controller separates proposal/value ranking from execution authority.  A
policy may rank only actions that the current typed graph certifies and that are
safe under every surviving hypothesis.  Structural uncertainty is represented
as refinement or abstention; it is never converted into permission by a score.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from threading import RLock

from .carving import (
    AdaptiveLearningLoop,
    AdaptiveLearningRouter,
    ConstraintRevision,
    RevisionResult,
    apply_prepared_revisions,
    prepare_constraint_revisions,
)
from .certificate import (
    assert_fresh,
    decision_token,
    supporting_evidence_ids,
)
from .evidence import EvidenceLedger
from .graph import FrontierResult, TypedActionGraph
from .memory import DomainMemory
from .model import (
    ActionKind,
    ActionNode,
    ActionStatus,
    Constraint,
    ControllerState,
    Decision,
    DecisionStatus,
    DomainTag,
    MetamorpherError,
    Observation,
    ObservationStatus,
    StaleDecisionError,
    UnsafeExecutionError,
)
from .policy import DecisionPolicy, HeuristicLookaheadPolicy
from .trace import EventTrace
from .version_space import VersionSpaceManager


@dataclass(frozen=True, slots=True)
class ObservationReceipt:
    evidence_revision: int
    graph_epoch: int
    action_id: str | None
    action_status: ActionStatus | None
    revision: RevisionResult | None = None
    observation_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ControllerCheckpoint:
    id: str
    label: str | None
    graph_epoch: int
    evidence_revision: int
    irreversible_effects: tuple[str, ...]


@dataclass(slots=True)
class _CheckpointState:
    public: ControllerCheckpoint
    graph_nodes: dict[str, ActionNode]
    graph_constraints: dict[str, Constraint]
    state: ControllerState
    version_space: VersionSpaceManager
    memory: DomainMemory
    adaptive_learning: AdaptiveLearningLoop | AdaptiveLearningRouter | None


@dataclass(frozen=True, slots=True)
class _CommittedDirective:
    decision: Decision
    action_id: str


class MetamorpherController:
    """Coordinate graph gating, value selection, evidence, and revision.

    Runtime protocol::

        decision = controller.next(domain)
        node = controller.commit(decision)
        # The caller executes ``node`` outside this library.
        receipt = controller.observe(observation, token=decision.token)

    An independent audit may be recorded without a token only when its
    ``Observation.independent_audit`` flag is true.  The library accepts no
    hidden-state, oracle-frontier, or post-hoc validity field.
    """

    _REFINEMENT_KINDS = frozenset(
        {
            ActionKind.OBSERVE,
            ActionKind.TEST,
            ActionKind.AUDIT,
            ActionKind.ESCALATE,
        }
    )

    def __init__(
        self,
        graph: TypedActionGraph,
        *,
        state: ControllerState | None = None,
        evidence: EvidenceLedger | None = None,
        version_space: VersionSpaceManager | None = None,
        memory: DomainMemory | None = None,
        policy: DecisionPolicy | None = None,
        trace: EventTrace | None = None,
        default_domain: DomainTag | None = None,
        adaptive_learning: AdaptiveLearningLoop | AdaptiveLearningRouter | None = None,
    ) -> None:
        graph.validate()
        self.graph = graph
        self.state = state or ControllerState()
        self.evidence = evidence or EvidenceLedger()
        self.version_space = version_space or VersionSpaceManager()
        self.memory = memory or DomainMemory()
        self.policy = policy or HeuristicLookaheadPolicy()
        self.trace = trace or EventTrace()
        self.default_domain = default_domain or DomainTag("default")
        self.adaptive_learning = adaptive_learning
        self._sequence = 0
        self._checkpoint_sequence = 0
        self._checkpoints: dict[str, _CheckpointState] = {}
        self._observation_journal: list[tuple[int, tuple[Observation, ...]]] = []
        self._issued: Decision | None = None
        self._committed: _CommittedDirective | None = None
        self._lock = RLock()

        unknown_statuses = sorted(set(self.state.action_status) - self.graph.nodes.keys())
        if unknown_statuses:
            raise ValueError(f"state contains unknown action IDs: {unknown_statuses}")
        self.trace.append(
            "controller_initialized",
            graph_epoch=self.graph.epoch,
            evidence_revision=self.evidence.revision,
            action_count=len(self.graph.nodes),
            constraint_count=len(self.graph.constraints),
        )

    @property
    def pending_decision(self) -> Decision | None:
        return self._issued

    @property
    def committed_decision(self) -> Decision | None:
        return self._committed.decision if self._committed else None

    def frontier(self) -> FrontierResult:
        """Expose the graph's complete tri-state frontier snapshot."""

        with self._lock:
            return self.graph.frontier(self.state, self.evidence)

    def checkpoint(self, label: str | None = None) -> ControllerCheckpoint:
        """Snapshot derived controller state without copying the evidence log.

        Evidence is an append-only historical record and intentionally remains
        outside rollback.  Checkpoints capture only revisable/derived state.
        """

        with self._lock:
            if self._committed is not None:
                raise MetamorpherError(
                    "cannot checkpoint while an external action is committed"
                )
            self._checkpoint_sequence += 1
            seed = (
                f"{self._checkpoint_sequence}|{self.graph.epoch}|"
                f"{self.evidence.revision}|{label or ''}"
            )
            checkpoint_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()
            public = ControllerCheckpoint(
                id=checkpoint_id,
                label=label,
                graph_epoch=self.graph.epoch,
                evidence_revision=self.evidence.revision,
                irreversible_effects=tuple(self.state.irreversible_effects),
            )
            self._checkpoints[checkpoint_id] = _CheckpointState(
                public=public,
                graph_nodes=copy.deepcopy(self.graph.nodes),
                graph_constraints=copy.deepcopy(self.graph.constraints),
                state=copy.deepcopy(self.state),
                version_space=copy.deepcopy(self.version_space),
                memory=copy.deepcopy(self.memory),
                adaptive_learning=copy.deepcopy(self.adaptive_learning),
            )
            self.trace.append("checkpoint_created", checkpoint=public)
            return public

    def rollback(
        self,
        checkpoint: ControllerCheckpoint | str,
    ) -> ControllerCheckpoint:
        """Restore derived state while retaining evidence and physical reality.

        If any irreversible effect occurred after the checkpoint, rollback is
        refused rather than representing that the world was undone.  A
        successful rollback always advances the graph epoch and invalidates all
        issued or committed directives.
        """

        with self._lock:
            checkpoint_id = checkpoint.id if isinstance(checkpoint, ControllerCheckpoint) else checkpoint
            saved = self._checkpoints.get(checkpoint_id)
            if saved is None:
                raise KeyError(f"unknown checkpoint: {checkpoint_id}")
            if isinstance(checkpoint, ControllerCheckpoint) and checkpoint != saved.public:
                raise StaleDecisionError("checkpoint payload does not match stored snapshot")

            if self._committed is not None:
                committed_node = self.graph.nodes[self._committed.action_id]
                if committed_node.irreversible or not committed_node.reversible:
                    self.trace.append(
                        "rollback_refused",
                        checkpoint_id=checkpoint_id,
                        committed_irreversible_action=committed_node.id,
                        graph_epoch=self.graph.epoch,
                        evidence_revision=self.evidence.revision,
                    )
                    raise UnsafeExecutionError(
                        "cannot rollback while an irreversible action may be in flight: "
                        + committed_node.id
                    )

            before_effects = set(saved.public.irreversible_effects)
            current_effects = set(self.state.irreversible_effects)
            new_effects = tuple(sorted(current_effects - before_effects))
            if new_effects:
                self.trace.append(
                    "rollback_refused",
                    checkpoint_id=checkpoint_id,
                    irreversible_effects=new_effects,
                    graph_epoch=self.graph.epoch,
                    evidence_revision=self.evidence.revision,
                )
                raise UnsafeExecutionError(
                    "rollback would erase irreversible effects: "
                    + ", ".join(new_effects)
                )

            old_epoch = self.graph.epoch
            restored_nodes = copy.deepcopy(saved.graph_nodes)
            restored_constraints = copy.deepcopy(saved.graph_constraints)
            probe = TypedActionGraph()
            probe.nodes = copy.deepcopy(restored_nodes)
            probe.constraints = copy.deepcopy(restored_constraints)
            probe.validate()

            self.graph.nodes = restored_nodes
            self.graph.constraints = restored_constraints
            self.graph.epoch = max(old_epoch, saved.public.graph_epoch) + 1
            self.state = copy.deepcopy(saved.state)
            self.version_space = copy.deepcopy(saved.version_space)
            self.memory = copy.deepcopy(saved.memory)
            self.adaptive_learning = copy.deepcopy(saved.adaptive_learning)
            replayed_batches = 0
            for revision, observation_batch in self._observation_journal:
                if revision <= saved.public.evidence_revision:
                    continue
                staged_learning = copy.deepcopy(self.adaptive_learning)
                learned = (
                    staged_learning.ingest(observation_batch)
                    if staged_learning is not None
                    else None
                )
                for observation in observation_batch:
                    if observation.status in {
                        ObservationStatus.OBSERVED,
                        ObservationStatus.INFERRED,
                    }:
                        self.version_space.observe(
                            observation.key,
                            observation.value,
                            observation.id,
                            observation.domain,
                        )
                if learned is not None:
                    _, learned_cell = learned
                    self.adaptive_learning = staged_learning
                    self.version_space.upsert(learned_cell, activate=True)
                replayed_batches += 1
            self._issued = None
            self._committed = None
            self.trace.append(
                "rollback_completed",
                checkpoint_id=checkpoint_id,
                old_graph_epoch=old_epoch,
                graph_epoch=self.graph.epoch,
                checkpoint_evidence_revision=saved.public.evidence_revision,
                retained_evidence_revision=self.evidence.revision,
                replayed_observation_batches=replayed_batches,
            )
            return saved.public

    def _version_snapshot(
        self,
        domain: DomainTag | None = None,
    ) -> tuple[frozenset[str], tuple[str, ...]]:
        active = self.version_space.active_for(domain)
        if active is None:
            return frozenset(self.graph.nodes), ()
        applicable = active.hypotheses_for(domain)
        return (
            active.common_safe_actions(domain).intersection(self.graph.nodes),
            tuple(sorted(applicable)),
        )

    def _required_probes(
        self,
        frontier: FrontierResult,
    ) -> dict[str, tuple[str, ...]]:
        by_probe: dict[str, list[str]] = {}
        for constraint_ids in frontier.unknown_constraints.values():
            for constraint_id in constraint_ids:
                constraint = self.graph.constraints[constraint_id]
                probe_id = constraint.probe_action_id
                if probe_id is None or probe_id not in self.graph.nodes:
                    continue
                by_probe.setdefault(probe_id, []).append(constraint_id)
        return {
            probe_id: tuple(sorted(set(constraint_ids)))
            for probe_id, constraint_ids in by_probe.items()
        }

    def _is_refinement_node(
        self,
        action_id: str,
        required_probes: dict[str, tuple[str, ...]],
    ) -> bool:
        node = self.graph.nodes[action_id]
        return bool(
            action_id in required_probes
            or node.probe_for
            or node.kind in self._REFINEMENT_KINDS
        )

    def _make_decision(
        self,
        *,
        status: DecisionStatus,
        domain: DomainTag,
        frontier: tuple[str, ...],
        common_safe: frozenset[str],
        represented_hypotheses: tuple[str, ...],
        reason: str,
        action_id: str | None = None,
        probe_id: str | None = None,
        blockers: Iterable[str] = (),
        unresolved: Iterable[str] = (),
    ) -> Decision:
        self._sequence += 1
        token = decision_token(
            sequence=self._sequence,
            graph_epoch=self.graph.epoch,
            evidence_revision=self.evidence.revision,
            domain=domain,
            status=status.value,
            action_id=action_id,
            probe_id=probe_id,
            frontier=frontier,
        )
        selected = action_id or probe_id
        support = (
            supporting_evidence_ids(self.graph, self.evidence, selected)
            if selected is not None
            else ()
        )
        return Decision(
            status=status,
            action_id=action_id,
            probe_id=probe_id,
            reason=reason,
            frontier=frontier,
            graph_epoch=self.graph.epoch,
            evidence_revision=self.evidence.revision,
            domain=domain,
            token=token,
            blockers=tuple(sorted(set(blockers))),
            unresolved_assumptions=tuple(sorted(set(unresolved))),
            supporting_evidence_ids=support,
            represented_hypotheses=represented_hypotheses,
            common_safe_actions=tuple(sorted(common_safe)),
        )

    def _abstain(
        self,
        *,
        domain: DomainTag,
        frontier: FrontierResult,
        common_safe: frozenset[str],
        represented_hypotheses: tuple[str, ...],
        reason: str,
        extra_blockers: Iterable[str] = (),
    ) -> Decision:
        violated = {
            constraint_id
            for ids in frontier.violated_constraints.values()
            for constraint_id in ids
        }
        unknown = {
            constraint_id
            for ids in frontier.unknown_constraints.values()
            for constraint_id in ids
        }
        return self._make_decision(
            status=DecisionStatus.ABSTAIN,
            domain=domain,
            frontier=(),
            common_safe=common_safe,
            represented_hypotheses=represented_hypotheses,
            reason=reason,
            blockers=(*violated, *extra_blockers),
            unresolved=(*unknown, *represented_hypotheses),
        )

    def next(self, domain: DomainTag | None = None) -> Decision:
        """Issue one revision-bound execute/refine/abstain decision."""

        with self._lock:
            if self._committed is not None:
                raise MetamorpherError(
                    "cannot issue another decision while an action is committed"
                )
            if self._issued is not None:
                self.trace.append(
                    "decision_superseded",
                    token=self._issued.token,
                    graph_epoch=self.graph.epoch,
                    evidence_revision=self.evidence.revision,
                )

            selected_domain = domain or self.default_domain
            graph_frontier = self.graph.frontier(self.state, self.evidence)
            common_safe, hypotheses = self._version_snapshot(selected_domain)
            safe_frontier = tuple(
                sorted(set(graph_frontier.certified).intersection(common_safe))
            )
            required_probes = self._required_probes(graph_frontier)

            if not safe_frontier:
                if graph_frontier.certified and not common_safe:
                    reason = (
                        "surviving hypotheses have no common-safe executable action"
                    )
                elif graph_frontier.certified:
                    reason = (
                        "certified actions are not safe under every surviving hypothesis"
                    )
                elif graph_frontier.refinement:
                    reason = (
                        "the frontier requires refinement but no required probe is "
                        "currently certified and common-safe"
                    )
                else:
                    reason = "all unfinished actions are blocked by hard constraints"
                decision = self._abstain(
                    domain=selected_domain,
                    frontier=graph_frontier,
                    common_safe=common_safe,
                    represented_hypotheses=hypotheses,
                    reason=reason,
                )
            else:
                try:
                    selected = self.policy.select(
                        self.graph,
                        self.state,
                        safe_frontier,
                    )
                except (KeyError, ValueError) as exc:
                    decision = self._abstain(
                        domain=selected_domain,
                        frontier=graph_frontier,
                        common_safe=common_safe,
                        represented_hypotheses=hypotheses,
                        reason=f"policy could not rank the safe frontier: {exc}",
                    )
                else:
                    if selected not in safe_frontier:
                        decision = self._abstain(
                            domain=selected_domain,
                            frontier=graph_frontier,
                            common_safe=common_safe,
                            represented_hypotheses=hypotheses,
                            reason="policy proposed an action outside the safe frontier",
                            extra_blockers=(str(selected),),
                        )
                    elif self._is_refinement_node(selected, required_probes):
                        constraints = required_probes.get(selected, ())
                        unresolved = (*constraints, *self.graph.nodes[selected].probe_for)
                        decision = self._make_decision(
                            status=DecisionStatus.REFINEMENT_REQUIRED,
                            domain=selected_domain,
                            frontier=safe_frontier,
                            common_safe=common_safe,
                            represented_hypotheses=hypotheses,
                            reason="selected a certified refinement/observation action",
                            probe_id=selected,
                            blockers=constraints,
                            unresolved=unresolved,
                        )
                    else:
                        decision = self._make_decision(
                            status=DecisionStatus.SUPPORTED_UNDER_MODEL,
                            domain=selected_domain,
                            frontier=safe_frontier,
                            common_safe=common_safe,
                            represented_hypotheses=hypotheses,
                            reason=(
                                "selected from the certified frontier and safe under "
                                "every surviving hypothesis"
                            ),
                            action_id=selected,
                        )

            self._issued = decision
            self.trace.append(
                "decision_issued",
                decision=decision,
                tri_state_frontier={
                    "certified": graph_frontier.certified,
                    "refinement": graph_frontier.refinement,
                    "blocked": graph_frontier.blocked,
                },
            )
            return decision

    def _resolve_issued(self, decision_or_token: Decision | str) -> Decision:
        if self._issued is None:
            raise StaleDecisionError("no decision is currently issued")
        token = (
            decision_or_token.token
            if isinstance(decision_or_token, Decision)
            else decision_or_token
        )
        if token != self._issued.token:
            raise StaleDecisionError("decision token is stale or was never issued")
        if isinstance(decision_or_token, Decision) and decision_or_token != self._issued:
            raise StaleDecisionError("decision payload does not match the issued token")
        return self._issued

    def _revalidate_permission(self, decision: Decision) -> str:
        assert_fresh(decision, self.graph, self.evidence)
        if decision.status == DecisionStatus.ABSTAIN:
            raise UnsafeExecutionError("an abstention cannot be committed")
        selected = decision.action_id or decision.probe_id
        if selected is None or selected not in self.graph.nodes:
            raise UnsafeExecutionError("decision does not name an executable node")

        graph_frontier = self.graph.frontier(self.state, self.evidence)
        common_safe, hypotheses = self._version_snapshot(decision.domain)
        safe_frontier = tuple(
            sorted(set(graph_frontier.certified).intersection(common_safe))
        )
        if safe_frontier != decision.frontier:
            raise StaleDecisionError("certified/common-safe frontier changed")
        if hypotheses != decision.represented_hypotheses:
            raise StaleDecisionError("represented version space changed")
        if tuple(sorted(common_safe)) != decision.common_safe_actions:
            raise StaleDecisionError("common-safe action set changed")
        if selected not in safe_frontier:
            raise UnsafeExecutionError(
                f"action is not certified and common-safe: {selected}"
            )

        required_probes = self._required_probes(graph_frontier)
        is_refinement = self._is_refinement_node(selected, required_probes)
        if decision.status == DecisionStatus.REFINEMENT_REQUIRED:
            if decision.action_id is not None or not is_refinement:
                raise StaleDecisionError("refinement probe is no longer required")
        elif (
            decision.status == DecisionStatus.SUPPORTED_UNDER_MODEL
            and (decision.probe_id is not None or is_refinement)
        ):
            raise StaleDecisionError("decision kind no longer matches selected node")
        return selected

    def commit(self, decision_or_token: Decision | str) -> ActionNode:
        """Authorize the exact issued directive for external execution."""

        with self._lock:
            if self._committed is not None:
                raise StaleDecisionError("a decision is already committed")
            decision = self._resolve_issued(decision_or_token)
            selected = self._revalidate_permission(decision)
            self._committed = _CommittedDirective(decision, selected)
            self.trace.append(
                "decision_committed",
                token=decision.token,
                action_id=selected,
                status=decision.status,
                graph_epoch=self.graph.epoch,
                evidence_revision=self.evidence.revision,
            )
            return self.graph.nodes[selected]

    def _check_observation_batch(
        self,
        observations: tuple[Observation, ...],
    ) -> None:
        if not observations:
            raise ValueError("an observation batch cannot be empty")
        ids = tuple(item.id for item in observations)
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate observation IDs within batch")
        existing = {item.id for item in self.evidence.events}
        duplicates = tuple(sorted(existing.intersection(ids)))
        if duplicates:
            raise ValueError(f"duplicate observation IDs: {duplicates}")
        for observation in observations:
            if not 0.0 <= observation.reliability <= 1.0:
                raise ValueError("reliability must be in [0, 1]")

    def observe(
        self,
        observation: Observation,
        *,
        token: str | None = None,
        action_status: ActionStatus = ActionStatus.COMPLETED,
        revisions: Iterable[ConstraintRevision] = (),
    ) -> ObservationReceipt:
        """Record one observation; a convenience wrapper for ``observe_many``."""

        return self.observe_many(
            (observation,),
            token=token,
            action_status=action_status,
            revisions=revisions,
        )

    def observe_many(
        self,
        observations: Iterable[Observation],
        *,
        token: str | None = None,
        action_status: ActionStatus = ActionStatus.COMPLETED,
        revisions: Iterable[ConstraintRevision] = (),
    ) -> ObservationReceipt:
        """Atomically record all observations produced by one committed action.

        Constraint candidates are validated against the prospective evidence and
        a detached graph before either the evidence ledger or live graph changes.
        Censored/conflicted evidence remains in the ledger but cannot eliminate
        hypotheses or support structural revisions.
        """

        with self._lock:
            observation_batch = tuple(observations)
            revision_batch = tuple(revisions)
            self._check_observation_batch(observation_batch)

            embedded_tokens = {
                item.action_token
                for item in observation_batch
                if item.action_token is not None
            }
            if token is not None and any(item != token for item in embedded_tokens):
                raise StaleDecisionError(
                    "an observation action token does not match supplied token"
                )
            if len(embedded_tokens) > 1:
                raise StaleDecisionError(
                    "an observation batch contains multiple action tokens"
                )
            supplied_token = token or (
                next(iter(embedded_tokens)) if embedded_tokens else None
            )
            action_id: str | None = None
            resulting_status: ActionStatus | None = None

            if supplied_token is None:
                if not all(item.independent_audit for item in observation_batch):
                    raise StaleDecisionError(
                        "every untokened observation must be an independent audit"
                    )
                if self._committed is not None:
                    raise StaleDecisionError(
                        "cannot interleave an audit with a committed action"
                    )
                if self._issued is not None:
                    self.trace.append(
                        "decision_invalidated_by_audit",
                        token=self._issued.token,
                        observation_ids=tuple(
                            item.id for item in observation_batch
                        ),
                    )
                    self._issued = None
            else:
                if self._committed is None:
                    raise StaleDecisionError(
                        "observation token has no committed decision"
                    )
                committed = self._committed
                if supplied_token != committed.decision.token:
                    raise StaleDecisionError("observation token is stale")
                assert_fresh(committed.decision, self.graph, self.evidence)
                wrong_domains = tuple(
                    item.id
                    for item in observation_batch
                    if item.domain is not None
                    and item.domain != committed.decision.domain
                )
                if wrong_domains:
                    raise StaleDecisionError(
                        "observation domain does not match committed decision: "
                        f"{wrong_domains}"
                    )
                if action_status not in {ActionStatus.COMPLETED, ActionStatus.FAILED}:
                    raise ValueError(
                        "a committed action observation must complete or fail it"
                    )
                if self.state.status_of(committed.action_id) != ActionStatus.PENDING:
                    raise StaleDecisionError("committed action state changed")
                action_id = committed.action_id
                resulting_status = action_status

            prospective_observations = (*self.evidence.events, *observation_batch)
            prepared = prepare_constraint_revisions(
                self.graph,
                revision_batch,
                prospective_observations,
            )
            staged_learning = copy.deepcopy(self.adaptive_learning)
            learned = (
                staged_learning.ingest(observation_batch)
                if staged_learning is not None
                else None
            )

            for observation in observation_batch:
                if not self.evidence.append(observation):
                    # Batch IDs were prevalidated; reaching this branch means
                    # the ledger was mutated outside the controller.
                    raise StaleDecisionError(
                        f"evidence ledger changed during append: {observation.id}"
                    )
            revision_result = apply_prepared_revisions(self.graph, prepared)

            for observation in observation_batch:
                if observation.status in {
                    ObservationStatus.OBSERVED,
                    ObservationStatus.INFERRED,
                }:
                    self.version_space.observe(
                        observation.key,
                        observation.value,
                        observation.id,
                        observation.domain,
                    )

            # Publish the learned revision only after the previous active cell
            # consumed this batch. Otherwise the training case would
            # immediately narrow the new global class to its own branch.
            if learned is not None:
                learning_result, learned_cell = learned
                self.adaptive_learning = staged_learning
                self.version_space.upsert(learned_cell, activate=True)
                self.trace.append(
                    "adaptive_class_updated",
                    cell_id=learned_cell.id,
                    status=learning_result.status,
                    support=learning_result.support,
                    separator_name=learning_result.separator_name,
                )

            if action_id is not None and resulting_status is not None:
                self.state.action_status[action_id] = resulting_status
                node = self.graph.nodes[action_id]
                if resulting_status == ActionStatus.COMPLETED and node.irreversible:
                    self.state.irreversible_effects.append(action_id)

            self.trace.append(
                "observation_batch_accepted",
                observations=observation_batch,
                action_id=action_id,
                action_status=resulting_status,
                graph_epoch=self.graph.epoch,
                evidence_revision=self.evidence.revision,
            )
            if revision_result is not None:
                self.trace.append(
                    "candidate_graph_revised",
                    graph_epoch=revision_result.graph_epoch,
                    constraint_ids=revision_result.constraint_ids,
                    evidence_ids=revision_result.evidence_ids,
                    reasons=revision_result.reasons,
                )
            active = self.version_space.active
            if active is not None:
                self.trace.append(
                    "version_space_updated",
                    cell_id=active.id,
                    status=active.status,
                    surviving_hypotheses=active.surviving_ids,
                    common_safe_actions=tuple(sorted(active.common_safe_actions())),
                    evidence_ids=tuple(item.id for item in observation_batch),
                )

            self._observation_journal.append(
                (self.evidence.revision, observation_batch)
            )

            if action_id is not None:
                self._committed = None
                self._issued = None

            return ObservationReceipt(
                evidence_revision=self.evidence.revision,
                graph_epoch=self.graph.epoch,
                action_id=action_id,
                action_status=resulting_status,
                revision=revision_result,
                observation_ids=tuple(item.id for item in observation_batch),
            )
