from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TruthValue(str, Enum):
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


class ActionKind(str, Enum):
    OBSERVE = "observe"
    TEST = "test"
    ACT = "act"
    REPAIR = "repair"
    AUDIT = "audit"
    ESCALATE = "escalate"


class ActionStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    DISABLED = "disabled"


class ConstraintKind(str, Enum):
    HARD_PREREQUISITE = "hard_prerequisite"
    SAFETY = "safety"
    GUARD = "guard"
    ALTERNATIVE = "alternative"
    MUTEX = "mutex"
    SOFT_EPISTEMIC = "soft_epistemic"


class ClaimStatus(str, Enum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class ClaimTier(str, Enum):
    EXTERNAL_POLICY = "external_policy"
    SUPPORTED = "supported"
    CANDIDATE = "candidate"


class ObservationStatus(str, Enum):
    OBSERVED = "observed"
    INFERRED = "inferred"
    CENSORED = "censored"
    CONFLICTED = "conflicted"


class DecisionStatus(str, Enum):
    SUPPORTED_UNDER_MODEL = "supported_under_model"
    REFINEMENT_REQUIRED = "refinement_required"
    ABSTAIN = "abstain"


class ClassStatus(str, Enum):
    PROVISIONAL = "provisional"
    SUPPORTED = "supported"
    UNRESOLVED = "unresolved"
    CARVED = "carved"


@dataclass(frozen=True, slots=True)
class DomainTag:
    name: str
    attributes: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_mapping(cls, name: str, values: Mapping[str, Any] | None = None) -> DomainTag:
        attrs = tuple(sorted((str(k), str(v)) for k, v in (values or {}).items()))
        return cls(name=name, attributes=attrs)


@dataclass(slots=True)
class ActionNode:
    id: str
    label: str
    kind: ActionKind = ActionKind.ACT
    cost: float = 0.0
    harm: float = 0.0
    delay: float = 0.0
    information_value: float = 0.0
    decision_value: float = 0.0
    reversible: bool = True
    irreversible: bool = False
    probe_for: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Constraint:
    id: str
    kind: ConstraintKind
    sources: tuple[str, ...]
    target: str
    status: ClaimStatus = ClaimStatus.CONFIRMED
    tier: ClaimTier = ClaimTier.SUPPORTED
    fact_key: str | None = None
    expected_value: Any = True
    probe_action_id: str | None = None
    confidence: float = 1.0
    provenance: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    contradicting_evidence_ids: tuple[str, ...] = ()
    domain: DomainTag | None = None
    externally_governed: bool = False
    graph_version: int = 0
    last_verified: float | None = None


@dataclass(slots=True)
class ControllerState:
    action_status: dict[str, ActionStatus] = field(default_factory=dict)
    irreversible_effects: list[str] = field(default_factory=list)

    def status_of(self, action_id: str) -> ActionStatus:
        return self.action_status.get(action_id, ActionStatus.PENDING)

    def completed(self, action_id: str) -> bool:
        return self.status_of(action_id) == ActionStatus.COMPLETED


@dataclass(frozen=True, slots=True)
class Observation:
    id: str
    key: str
    value: Any = None
    status: ObservationStatus = ObservationStatus.OBSERVED
    source: str = "unknown"
    reliability: float = 1.0
    domain: DomainTag | None = None
    action_token: str | None = None
    timestamp: float | None = None
    censoring_reason: str | None = None
    independent_audit: bool = False


@dataclass(frozen=True, slots=True)
class Decision:
    status: DecisionStatus
    action_id: str | None
    probe_id: str | None
    reason: str
    frontier: tuple[str, ...]
    graph_epoch: int
    evidence_revision: int
    domain: DomainTag
    token: str
    blockers: tuple[str, ...] = ()
    unresolved_assumptions: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    represented_hypotheses: tuple[str, ...] = ()
    common_safe_actions: tuple[str, ...] = ()
    version_space_digest: str = ""


class MetamorpherError(RuntimeError):
    pass


class InvalidGraphError(MetamorpherError):
    pass


class StaleDecisionError(MetamorpherError):
    pass


class UnsafeExecutionError(MetamorpherError):
    pass
