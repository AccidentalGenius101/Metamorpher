"""Evidence-gated expansion of an incomplete representational vocabulary.

This module sits outside ordinary graph revision.  A ``ConstraintRevision``
changes relations among already represented objects; an ``ExpansionCapsule``
proposes a new distinction while recording which old constraints must survive,
which residuals motivated the proposal, and which observation could
discriminate it.  Proposal detail never counts as evidential support.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from .model import DomainTag, Observation, ObservationStatus


class RepresentationStatus(str, Enum):
    """Model-boundary status, separate from the truth of individual claims."""

    ADEQUATE_IN_SCOPE = "adequate_in_scope"
    PREDICTIVE_BUT_INCOMPLETE = "predictive_but_incomplete"
    UNRESOLVED_IN_VOCABULARY = "unresolved_in_vocabulary"
    EXPANSION_PROPOSED = "expansion_proposed"


class ExpansionStatus(str, Enum):
    QUARANTINED = "quarantined"
    SUPPORTED = "supported"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ResidualSignature:
    """One attributable prediction failure that existing carving did not absorb."""

    prediction_id: str
    expected: Any
    observed: Any
    evidence_id: str
    context: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.prediction_id.strip():
            raise ValueError("residual prediction_id cannot be empty")
        if not self.evidence_id.strip():
            raise ValueError("residual evidence_id cannot be empty")
        keys = [key for key, _ in self.context]
        if len(keys) != len(set(keys)):
            raise ValueError("residual context keys must be unique")


@dataclass(frozen=True, slots=True)
class DiscriminatingPrediction:
    """A prospective observation that separates an expansion from its parent."""

    id: str
    observation_key: str
    expected_value: Any
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.observation_key.strip():
            raise ValueError("discriminating prediction ID and key cannot be empty")


@dataclass(frozen=True, slots=True)
class ProjectionMapping:
    """How a parent remains a valid, possibly lossy, view of an expansion."""

    source_representation_id: str
    target_representation_id: str
    preserved_constraint_ids: tuple[str, ...]
    lost_constraint_ids: tuple[str, ...] = ()
    coordinate_map: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.source_representation_id or not self.target_representation_id:
            raise ValueError("projection endpoints cannot be empty")
        if self.source_representation_id == self.target_representation_id:
            raise ValueError("a projection must connect distinct representations")
        preserved = set(self.preserved_constraint_ids)
        lost = set(self.lost_constraint_ids)
        if preserved & lost:
            raise ValueError("a constraint cannot be both preserved and lost")
        sources = [source for source, _ in self.coordinate_map]
        if len(sources) != len(set(sources)):
            raise ValueError("projection coordinate sources must be unique")


@dataclass(frozen=True, slots=True)
class RepresentationBoundary:
    """Explicit demonstrated scope and unresolved residuals of one representation."""

    representation_id: str
    domain: DomainTag
    status: RepresentationStatus
    preserved_constraint_ids: tuple[str, ...] = ()
    demonstrated_contexts: tuple[tuple[tuple[str, Any], ...], ...] = ()
    residuals: tuple[ResidualSignature, ...] = ()
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.representation_id.strip():
            raise ValueError("representation_id cannot be empty")
        if self.status == RepresentationStatus.ADEQUATE_IN_SCOPE and self.residuals:
            raise ValueError("an adequate-in-scope boundary cannot retain residuals")


@dataclass(frozen=True, slots=True)
class ExpansionCapsule:
    """A quarantined proposal to add representational freedom.

    ``supporting_evidence_ids`` is intentionally empty on a new proposal.
    Residuals motivate attention; only observations matching the prospective
    discriminating predictions can support promotion.
    """

    id: str
    domain: DomainTag
    parent_representation_id: str
    proposed_representation_id: str
    projection: ProjectionMapping
    residuals: tuple[ResidualSignature, ...]
    predictions: tuple[DiscriminatingPrediction, ...]
    proposed_distinctions: tuple[str, ...] = ()
    rationale: str = ""
    status: ExpansionStatus = ExpansionStatus.QUARANTINED
    supporting_evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("expansion capsule ID cannot be empty")
        if self.parent_representation_id != self.projection.source_representation_id:
            raise ValueError("projection source must be the parent representation")
        if self.proposed_representation_id != self.projection.target_representation_id:
            raise ValueError("projection target must be the proposed representation")
        if not self.residuals:
            raise ValueError("an expansion requires at least one structured residual")
        if not self.predictions:
            raise ValueError("an expansion requires a discriminating prediction")
        ids = [prediction.id for prediction in self.predictions]
        if len(ids) != len(set(ids)):
            raise ValueError("discriminating prediction IDs must be unique")


@dataclass(frozen=True, slots=True)
class LocalEvidencePacket:
    """Non-voting evidence sent from a locally privileged learner."""

    id: str
    domain: DomainTag
    context: tuple[tuple[str, Any], ...]
    violated_prediction_id: str
    evidence_ids: tuple[str, ...]
    residuals: tuple[ResidualSignature, ...]
    proposed_distinction: str = ""
    reproduction_prediction_id: str = ""
    reliability: float = 1.0
    expected_under_parent: bool = False

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.violated_prediction_id.strip():
            raise ValueError("packet and violated prediction IDs cannot be empty")
        if not 0.0 <= self.reliability <= 1.0:
            raise ValueError("packet reliability must be in [0, 1]")
        if not self.evidence_ids or not self.residuals:
            raise ValueError("a local packet requires evidence and residuals")

    @property
    def attention_priority(self) -> float:
        """Route surprising credible evidence; do not count agreeing nodes."""

        novelty = 0.0 if self.expected_under_parent else 1.0
        reproducibility = 1.0 if self.reproduction_prediction_id else 0.5
        return self.reliability * novelty * reproducibility


@dataclass(slots=True)
class ExpansionRegistry:
    """Quarantine and promote expansion capsules using external evidence."""

    capsules: dict[str, ExpansionCapsule] = field(default_factory=dict)

    def propose(self, capsule: ExpansionCapsule) -> None:
        if capsule.id in self.capsules:
            raise ValueError(f"duplicate expansion capsule: {capsule.id}")
        if capsule.status != ExpansionStatus.QUARANTINED:
            raise ValueError("a new expansion must enter quarantined")
        if capsule.supporting_evidence_ids:
            raise ValueError("a proposal cannot declare its own supporting evidence")
        self.capsules[capsule.id] = copy.deepcopy(capsule)

    def get(self, capsule_id: str) -> ExpansionCapsule | None:
        capsule = self.capsules.get(capsule_id)
        return copy.deepcopy(capsule) if capsule is not None else None

    def promote(
        self,
        capsule_id: str,
        observations: Iterable[Observation],
    ) -> ExpansionCapsule:
        capsule = self.capsules[capsule_id]
        if capsule.status != ExpansionStatus.QUARANTINED:
            raise ValueError("only a quarantined expansion can be promoted")
        evidence = {item.id: item for item in observations}
        usable = {
            evidence_id: item
            for evidence_id, item in evidence.items()
            if item.status in {ObservationStatus.OBSERVED, ObservationStatus.INFERRED}
            and item.domain == capsule.domain
        }
        matched: list[str] = []
        for prediction in capsule.predictions:
            matches = [
                item.id
                for item in usable.values()
                if item.key == prediction.observation_key
                and item.value == prediction.expected_value
            ]
            if not matches:
                raise ValueError(
                    f"no usable evidence supports prediction {prediction.id!r}"
                )
            matched.extend(matches)
        promoted = replace(
            capsule,
            status=ExpansionStatus.SUPPORTED,
            supporting_evidence_ids=tuple(sorted(set(matched))),
        )
        self.capsules[capsule_id] = promoted
        return copy.deepcopy(promoted)

    def reject(self, capsule_id: str) -> ExpansionCapsule:
        capsule = self.capsules[capsule_id]
        if capsule.status != ExpansionStatus.QUARANTINED:
            raise ValueError("only a quarantined expansion can be rejected")
        rejected = replace(capsule, status=ExpansionStatus.REJECTED)
        self.capsules[capsule_id] = rejected
        return copy.deepcopy(rejected)

    def ranked_packets(
        self, packets: Iterable[LocalEvidencePacket]
    ) -> tuple[LocalEvidencePacket, ...]:
        """Rank by conditional information value, never majority frequency."""

        return tuple(
            sorted(
                packets,
                key=lambda packet: (-packet.attention_priority, packet.id),
            )
        )
