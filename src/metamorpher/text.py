"""Language adapters that preserve Metamorpher's evidence boundaries."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .cognition import CandidateStructure, Perception
from .evidence import EvidenceLedger
from .graph import TypedActionGraph
from .model import Decision, DecisionStatus, DomainTag, Observation, ObservationStatus


@dataclass(frozen=True, slots=True)
class TextEvidence:
    """One interpreter claim anchored to an exact source-text span."""

    key: str
    value: Any
    start: int
    end: int
    status: ObservationStatus = ObservationStatus.OBSERVED
    reliability: float = 1.0


class TextInterpreter(Protocol):
    """Extract attributable evidence without receiving execution authority."""

    def interpret(
        self, text: str, domain: DomainTag, context: Mapping[str, Any]
    ) -> Sequence[TextEvidence]: ...


@dataclass(frozen=True, slots=True)
class DiscourseInterpretation:
    """Separate grounded observations from untrusted proposed structure."""

    evidence: tuple[TextEvidence, ...] = ()
    candidates: CandidateStructure | None = None


class DiscourseInterpreter(Protocol):
    def interpret_discourse(
        self, text: str, domain: DomainTag
    ) -> DiscourseInterpretation: ...


class InterpreterPerceiver:
    """Adapt a text interpreter to the cognitive loop's ``Perceiver`` boundary."""

    def __init__(self, interpreter: TextInterpreter, *, source: str = "text") -> None:
        self.interpreter = interpreter
        self.source = source

    def perceive(
        self,
        raw_input: Any,
        domain: DomainTag,
        *,
        action_token: str | None = None,
    ) -> Perception:
        if not isinstance(raw_input, str):
            raise TypeError("InterpreterPerceiver requires text input")
        claims = tuple(self.interpreter.interpret(raw_input, domain, {}))
        text_digest = hashlib.sha256(raw_input.encode()).hexdigest()
        observations: list[Observation] = []
        spans: list[dict[str, Any]] = []
        for index, claim in enumerate(claims):
            if not claim.key or not 0 <= claim.start < claim.end <= len(raw_input):
                raise ValueError("text evidence requires a valid key and source span")
            if not 0.0 <= claim.reliability <= 1.0:
                raise ValueError("text evidence reliability must be in [0, 1]")
            quote = raw_input[claim.start : claim.end]
            seed = (
                f"{domain!r}|{claim.key}|{claim.start}|{claim.end}|{quote}|{index}"
            )
            evidence_id = "text-" + hashlib.sha256(seed.encode()).hexdigest()[:20]
            source = f"{self.source}:sha256:{text_digest}#{claim.start}:{claim.end}"
            observations.append(
                Observation(
                    evidence_id,
                    claim.key,
                    claim.value,
                    claim.status,
                    source,
                    claim.reliability,
                    domain,
                    action_token,
                )
            )
            spans.append(
                {
                    "evidence_id": evidence_id,
                    "start": claim.start,
                    "end": claim.end,
                    "quote": quote,
                }
            )
        representation = {
            "text": raw_input,
            "sha256": text_digest,
            "spans": tuple(spans),
        }
        return Perception(tuple(observations), representation, self.source)


class DiscoursePerceiver:
    """Run one discourse pass and carry its candidate structure to proposal."""

    def __init__(self, interpreter: DiscourseInterpreter, *, source: str = "text") -> None:
        self.interpreter = interpreter
        self.source = source

    def perceive(
        self,
        raw_input: Any,
        domain: DomainTag,
        *,
        action_token: str | None = None,
    ) -> Perception:
        if not isinstance(raw_input, str):
            raise TypeError("DiscoursePerceiver requires text input")
        interpretation = self.interpreter.interpret_discourse(raw_input, domain)

        class EvidenceOnly:
            def interpret(inner_self, text, selected_domain, context):
                return interpretation.evidence

        perception = InterpreterPerceiver(
            EvidenceOnly(), source=self.source
        ).perceive(raw_input, domain, action_token=action_token)
        representation = {
            **dict(perception.representation),
            "candidate_structure": interpretation.candidates,
        }
        return Perception(perception.observations, representation, perception.source)


class DiscourseProposer:
    """Forward only the quarantined structure produced during perception."""

    def propose(self, context, observations, domain) -> CandidateStructure:
        candidate = context.get("representation", {}).get("candidate_structure")
        if not isinstance(candidate, CandidateStructure):
            raise ValueError("discourse perception produced no candidate structure")
        return candidate


class GroundedTextRenderer:
    """Render only claims already carried by a model-relative decision."""

    def render(
        self,
        decision: Decision,
        graph: TypedActionGraph,
        evidence: EvidenceLedger,
    ) -> str:
        selected = decision.action_id or decision.probe_id
        label = graph.nodes[selected].label if selected in graph.nodes else selected
        if decision.status == DecisionStatus.SUPPORTED_UNDER_MODEL:
            lead = f"Under the current model, the next supported action is: {label}."
        elif decision.status == DecisionStatus.REFINEMENT_REQUIRED:
            lead = f"More information is required. Next represented probe: {label}."
        else:
            lead = "The current model cannot support an action, so it abstains."
        details = [decision.reason]
        if decision.unresolved_assumptions:
            details.append(
                "Unresolved: " + ", ".join(decision.unresolved_assumptions) + "."
            )
        if decision.supporting_evidence_ids:
            known = {item.id for item in evidence.events}
            ids = [item for item in decision.supporting_evidence_ids if item in known]
            if ids:
                details.append("Supporting evidence: " + ", ".join(ids) + ".")
        return " ".join((lead, *details))
