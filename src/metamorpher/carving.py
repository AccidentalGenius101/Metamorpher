"""Evidence-grounded, transactional revisions of the candidate graph."""

from __future__ import annotations

import copy
from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, replace

from .graph import TypedActionGraph
from .model import (
    ClaimTier,
    ClassStatus,
    Constraint,
    InvalidGraphError,
    Observation,
    ObservationStatus,
    UnsafeExecutionError,
)


@dataclass(frozen=True, slots=True)
class OutcomeSupport:
    outcome: Hashable
    count: int


@dataclass(frozen=True, slots=True)
class CarvedBranch:
    separator_value: Hashable
    support: int
    outcomes: tuple[OutcomeSupport, ...]
    dominant_outcome: Hashable
    purity: float


@dataclass(frozen=True, slots=True)
class CarvingResult:
    cell_id: str
    status: ClassStatus
    support: int
    outcomes: tuple[OutcomeSupport, ...]
    separator_name: str | None
    branches: tuple[CarvedBranch, ...]
    reason: str


class FailureCarver:
    """Preserve contradictions, carving only an identifiable supplied split.

    The carver deliberately does not search for or name hidden causes.  A caller
    may supply an observed separator and its value for every accumulated record.
    Until each branch has enough support and a stable outcome, the parent remains
    one unresolved equivalence class.
    """

    def __init__(
        self,
        cell_id: str,
        *,
        min_branch_support: int = 10,
        stability_threshold: float = 0.85,
    ) -> None:
        if not cell_id:
            raise ValueError("cell_id cannot be empty")
        if min_branch_support < 1:
            raise ValueError("min_branch_support must be positive")
        if not 0.5 < stability_threshold <= 1.0:
            raise ValueError("stability_threshold must be in (0.5, 1]")
        self.cell_id = cell_id
        self.min_branch_support = min_branch_support
        self.stability_threshold = stability_threshold
        self._records: dict[str, Hashable] = {}
        self._status = ClassStatus.PROVISIONAL
        self._separator_name: str | None = None
        self._branches: tuple[CarvedBranch, ...] = ()

    @property
    def records(self) -> Mapping[str, Hashable]:
        return dict(self._records)

    @staticmethod
    def _supports(values: Iterable[Hashable]) -> tuple[OutcomeSupport, ...]:
        counts = Counter(values)
        return tuple(
            OutcomeSupport(outcome, count)
            for outcome, count in sorted(counts.items(), key=lambda item: repr(item[0]))
        )

    def result(self, reason: str = "current equivalence-class state") -> CarvingResult:
        return CarvingResult(
            cell_id=self.cell_id,
            status=self._status,
            support=len(self._records),
            outcomes=self._supports(self._records.values()),
            separator_name=self._separator_name,
            branches=self._branches,
            reason=reason,
        )

    def observe(self, evidence_id: str, outcome: Hashable) -> CarvingResult:
        """Accumulate an outcome without inventing a separating explanation."""

        if not evidence_id:
            raise ValueError("evidence_id cannot be empty")
        try:
            hash(outcome)
        except TypeError as exc:
            raise TypeError("outcomes must be hashable") from exc
        if evidence_id in self._records:
            if self._records[evidence_id] != outcome:
                raise ValueError(
                    f"evidence ID {evidence_id!r} was assigned conflicting outcomes"
                )
            return self.result("duplicate evidence ignored")

        self._records[evidence_id] = outcome
        distinct = len(set(self._records.values()))
        self._separator_name = None
        self._branches = ()
        if distinct > 1:
            # Contradiction is representable immediately, even before enough
            # evidence exists to justify any split.
            self._status = ClassStatus.UNRESOLVED
            reason = "incompatible outcomes preserved in one unresolved cell"
        elif len(self._records) >= self.min_branch_support:
            self._status = ClassStatus.SUPPORTED
            reason = "provisional parent rule has support without contradiction"
        else:
            self._status = ClassStatus.PROVISIONAL
            reason = "insufficient support; retaining the provisional parent"
        return self.result(reason)

    def try_carve(
        self,
        separator_name: str,
        assignments: Mapping[str, Hashable],
    ) -> CarvingResult:
        """Carve only when a supplied observed separator is stably supported."""

        if not separator_name.strip():
            raise ValueError("separator_name cannot be empty")
        missing = tuple(sorted(set(self._records) - assignments.keys()))
        if missing:
            self._status = ClassStatus.UNRESOLVED
            return self.result(
                "separator remains unidentified for evidence IDs: "
                + ", ".join(missing)
            )
        if len(set(self._records.values())) < 2:
            return self.result("no contradictory outcomes require a partition")

        # A new carving proposal is evaluated conservatively from the unresolved
        # parent.  A rejected proposal must not leave stale branch metadata.
        self._status = ClassStatus.UNRESOLVED
        self._separator_name = None
        self._branches = ()

        grouped: dict[Hashable, list[Hashable]] = defaultdict(list)
        for evidence_id, outcome in self._records.items():
            separator_value = assignments[evidence_id]
            try:
                hash(separator_value)
            except TypeError as exc:
                raise TypeError("separator values must be hashable") from exc
            grouped[separator_value].append(outcome)
        if len(grouped) < 2:
            self._status = ClassStatus.UNRESOLVED
            return self.result("supplied separator does not create multiple branches")

        branches: list[CarvedBranch] = []
        insufficient: list[str] = []
        unstable: list[str] = []
        for separator_value, outcomes in sorted(
            grouped.items(), key=lambda item: repr(item[0])
        ):
            counts = Counter(outcomes)
            support = len(outcomes)
            dominant_outcome, dominant_count = max(
                counts.items(), key=lambda item: (item[1], repr(item[0]))
            )
            purity = dominant_count / support
            if support < self.min_branch_support:
                insufficient.append(repr(separator_value))
            if purity < self.stability_threshold:
                unstable.append(repr(separator_value))
            branches.append(
                CarvedBranch(
                    separator_value=separator_value,
                    support=support,
                    outcomes=self._supports(outcomes),
                    dominant_outcome=dominant_outcome,
                    purity=purity,
                )
            )

        if insufficient or unstable:
            self._status = ClassStatus.UNRESOLVED
            details: list[str] = []
            if insufficient:
                details.append("insufficient branch support=" + ", ".join(insufficient))
            if unstable:
                details.append("unstable branches=" + ", ".join(unstable))
            return self.result("; ".join(details))
        if len({branch.dominant_outcome for branch in branches}) < 2:
            self._status = ClassStatus.UNRESOLVED
            return self.result(
                "separator branches do not predict incompatible outcomes"
            )

        self._status = ClassStatus.CARVED
        self._separator_name = separator_name.strip()
        self._branches = tuple(branches)
        return self.result(
            "supplied separator met per-branch support and stability requirements"
        )


@dataclass(frozen=True, slots=True)
class ConstraintRevision:
    """A proposed addition or replacement of one candidate constraint.

    ``evidence_ids`` must refer to observed or explicitly inferred evidence.
    Censored/conflicted records cannot justify a structural commitment.  A
    replacement preserves prior provenance and may not modify an externally
    governed constraint.
    """

    constraint: Constraint
    evidence_ids: tuple[str, ...]
    reason: str
    replace_existing: bool = False


@dataclass(frozen=True, slots=True)
class RevisionResult:
    graph_epoch: int
    constraint_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    reasons: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class PreparedRevisionBatch:
    base_graph_epoch: int
    constraints: tuple[Constraint, ...]
    constraint_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    reasons: tuple[tuple[str, str], ...]


def _validate_revision(
    graph: TypedActionGraph,
    revision: ConstraintRevision,
    observations: Mapping[str, Observation],
) -> Constraint:
    if not revision.reason.strip():
        raise ValueError("a graph revision requires a non-empty reason")
    if not revision.evidence_ids:
        raise ValueError("a graph revision requires evidence provenance")
    missing = sorted(set(revision.evidence_ids) - observations.keys())
    if missing:
        raise ValueError(f"unknown revision evidence IDs: {missing}")
    unsupported = sorted(
        evidence_id
        for evidence_id in revision.evidence_ids
        if observations[evidence_id].status
        in {ObservationStatus.CENSORED, ObservationStatus.CONFLICTED}
    )
    if unsupported:
        raise ValueError(
            "censored or conflicted observations cannot support a graph "
            f"revision: {unsupported}"
        )
    candidate = copy.deepcopy(revision.constraint)
    if not 0.0 <= candidate.confidence <= 1.0:
        raise ValueError("constraint confidence must be in [0, 1]")
    if (
        candidate.probe_action_id is not None
        and candidate.probe_action_id not in graph.nodes
    ):
        raise InvalidGraphError(
            f"unknown probe action: {candidate.probe_action_id}"
        )

    existing = graph.constraints.get(candidate.id)
    if existing is not None:
        if not revision.replace_existing:
            raise InvalidGraphError(
                f"constraint already exists; replacement not authorized: {candidate.id}"
            )
        if existing.externally_governed:
            raise UnsafeExecutionError(
                f"controller cannot revise externally governed constraint: {candidate.id}"
            )
        inherited = existing.provenance
        inherited_support = existing.supporting_evidence_ids
        inherited_contradictions = existing.contradicting_evidence_ids
    else:
        if revision.replace_existing:
            raise InvalidGraphError(
                f"cannot replace missing constraint: {candidate.id}"
            )
        if candidate.externally_governed:
            raise UnsafeExecutionError(
                "controller evidence cannot create an externally governed constraint"
            )
        inherited = ()
        inherited_support = ()
        inherited_contradictions = ()

    if candidate.tier == ClaimTier.EXTERNAL_POLICY:
        raise UnsafeExecutionError(
            "controller evidence cannot promote a claim to external policy"
        )

    provenance = tuple(
        dict.fromkeys((*inherited, *candidate.provenance, *revision.evidence_ids))
    )
    supporting = tuple(
        dict.fromkeys(
            (
                *inherited_support,
                *candidate.supporting_evidence_ids,
                *revision.evidence_ids,
            )
        )
    )
    contradictions = tuple(
        dict.fromkeys(
            (*inherited_contradictions, *candidate.contradicting_evidence_ids)
        )
    )
    return replace(
        candidate,
        provenance=provenance,
        supporting_evidence_ids=supporting,
        contradicting_evidence_ids=contradictions,
        graph_version=graph.epoch + 1,
    )


def _install(graph: TypedActionGraph, constraints: Iterable[Constraint]) -> None:
    for constraint in constraints:
        if constraint.id in graph.constraints:
            del graph.constraints[constraint.id]
        graph.add_constraint(copy.deepcopy(constraint))


def prepare_constraint_revisions(
    graph: TypedActionGraph,
    revisions: Iterable[ConstraintRevision],
    observations: Iterable[Observation],
) -> PreparedRevisionBatch | None:
    """Validate a batch without mutating the live graph."""

    batch = tuple(revisions)
    if not batch:
        return None
    observation_batch = tuple(observations)
    observation_map = {item.id: item for item in observation_batch}
    if len(observation_map) != len(observation_batch):
        # This normally cannot happen through EvidenceLedger, but keeping the
        # revision helper correct in isolation prevents ambiguous provenance.
        raise ValueError("duplicate observation IDs in revision evidence")

    ids = [revision.constraint.id for revision in batch]
    if len(ids) != len(set(ids)):
        raise ValueError("a revision batch may modify each constraint only once")
    prepared = tuple(
        _validate_revision(graph, revision, observation_map)
        for revision in batch
    )

    # Preflight against a detached copy so a bad proposal never touches the
    # live object (or consumes one of its epochs).
    detached = copy.deepcopy(graph)
    _install(detached, prepared)
    detached.validate()

    evidence_ids = tuple(
        sorted({item for revision in batch for item in revision.evidence_ids})
    )
    return PreparedRevisionBatch(
        base_graph_epoch=graph.epoch,
        constraints=prepared,
        constraint_ids=tuple(sorted(ids)),
        evidence_ids=evidence_ids,
        reasons=tuple(
            sorted(
                (revision.constraint.id, revision.reason.strip())
                for revision in batch
            )
        ),
    )


def apply_prepared_revisions(
    graph: TypedActionGraph,
    prepared: PreparedRevisionBatch | None,
) -> RevisionResult | None:
    """Install one prevalidated batch in a single graph transaction."""

    if prepared is None:
        return None
    if graph.epoch != prepared.base_graph_epoch:
        raise InvalidGraphError(
            "candidate revision is stale: "
            f"prepared={prepared.base_graph_epoch}, current={graph.epoch}"
        )

    transaction = graph.transaction()
    with transaction as staged:
        _install(staged, prepared.constraints)
        transaction.commit()

    return RevisionResult(
        graph.epoch,
        prepared.constraint_ids,
        prepared.evidence_ids,
        prepared.reasons,
    )


def apply_constraint_revisions(
    graph: TypedActionGraph,
    revisions: Iterable[ConstraintRevision],
    observations: Iterable[Observation],
) -> RevisionResult | None:
    """Validate and atomically apply a batch of candidate constraints."""

    prepared = prepare_constraint_revisions(graph, revisions, observations)
    return apply_prepared_revisions(graph, prepared)
