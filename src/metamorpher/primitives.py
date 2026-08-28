"""Qualified primitive execution and contributor-set routing."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .cognition import CandidateStructure
from .model import ActionKind, ActionNode, ClaimTier, Constraint, ConstraintKind, DomainTag
from .text import DiscourseInterpretation, TextEvidence
from .version_space import Hypothesis


class PrimitiveError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PrimitiveCall:
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PrimitiveQualification:
    passed: int
    total: int
    evidence_ids: tuple[str, ...]

    @property
    def qualified(self) -> bool:
        return self.total > 0 and self.passed == self.total and bool(self.evidence_ids)


@dataclass(frozen=True, slots=True)
class PrimitiveRecord:
    name: str
    qualification: PrimitiveQualification
    revision: int
    frozen: bool = True


@dataclass(slots=True)
class _Composition:
    text: str
    domain: DomainTag
    evidence: list[TextEvidence] = field(default_factory=list)
    actions: dict[str, ActionNode] = field(default_factory=dict)
    suspicions: list[tuple[str, str, int, int]] = field(default_factory=list)
    before: list[tuple[str, str]] = field(default_factory=list)

    def span(self, quote: str) -> tuple[int, int]:
        if not quote:
            raise PrimitiveError("source quotation cannot be empty")
        start = self.text.find(quote)
        if start < 0:
            raise PrimitiveError(f"source quotation not found: {quote!r}")
        if self.text.find(quote, start + 1) >= 0:
            raise PrimitiveError(f"source quotation is ambiguous: {quote!r}")
        return start, start + len(quote)


PrimitiveImplementation = Callable[[_Composition, Mapping[str, Any]], None]


class PrimitiveRegistry:
    """Append qualified primitives; installed implementations are immutable."""

    def __init__(self) -> None:
        self._implementations: dict[str, PrimitiveImplementation] = {}
        self._records: dict[str, PrimitiveRecord] = {}
        self._revision = 0

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))

    def record(self, name: str) -> PrimitiveRecord | None:
        return self._records.get(name)

    def install(
        self,
        name: str,
        implementation: PrimitiveImplementation,
        qualification: PrimitiveQualification,
    ) -> PrimitiveRecord:
        if not name or name in self._records:
            raise PrimitiveError(f"primitive already exists or has no name: {name!r}")
        if not qualification.qualified:
            raise PrimitiveError("primitive must pass every qualification case")
        self._revision += 1
        record = PrimitiveRecord(name, qualification, self._revision)
        self._implementations[name] = implementation
        self._records[name] = record
        return record

    def execute(self, call: PrimitiveCall, composition: _Composition) -> None:
        implementation = self._implementations.get(call.name)
        if implementation is None:
            raise PrimitiveError(f"unknown or unqualified primitive: {call.name}")
        implementation(composition, call.arguments)


class ContributorRouter(Protocol):
    def route(
        self, text: str, domain: DomainTag, available: Sequence[str]
    ) -> Sequence[PrimitiveCall]: ...


class PrimitiveComposer:
    def __init__(self, registry: PrimitiveRegistry) -> None:
        self.registry = registry

    def compose(
        self,
        text: str,
        domain: DomainTag,
        calls: Sequence[PrimitiveCall],
    ) -> DiscourseInterpretation:
        if not calls:
            raise PrimitiveError("contributor router selected no primitives")
        state = _Composition(text, domain)
        for call in calls:
            self.registry.execute(call, state)
        constraints = []
        for source, target in state.before:
            if source not in state.actions or target not in state.actions:
                raise PrimitiveError("prerequisite references an unknown action")
            constraints.append(
                Constraint(
                    f"candidate-before-{source}-{target}",
                    ConstraintKind.HARD_PREREQUISITE,
                    (source,),
                    target,
                    tier=ClaimTier.CANDIDATE,
                    provenance=("qualified-primitive:add_prerequisite",),
                )
            )
        safe = frozenset(
            action.id
            for action in state.actions.values()
            if action.kind
            in {ActionKind.OBSERVE, ActionKind.TEST, ActionKind.AUDIT, ActionKind.ESCALATE}
        )
        hypotheses = tuple(
            Hypothesis(
                item_id,
                safe,
                domain=domain,
                provenance=(
                    "qualified-primitive:register_suspicion",
                    f"speaker_suspicion:{start}:{end}",
                    f"quote:{quote}",
                ),
            )
            for item_id, quote, start, end in state.suspicions
        )
        candidates = CandidateStructure(
            tuple(state.actions.values()),
            tuple(constraints),
            hypotheses,
            "composed from qualified frozen primitives",
        )
        return DiscourseInterpretation(tuple(state.evidence), candidates)


class PrimitiveDiscourseInterpreter:
    """Route contributors, then execute their qualified implementations."""

    def __init__(self, router: ContributorRouter, registry: PrimitiveRegistry) -> None:
        self.router = router
        self.composer = PrimitiveComposer(registry)

    def interpret_discourse(
        self, text: str, domain: DomainTag
    ) -> DiscourseInterpretation:
        calls = tuple(self.router.route(text, domain, self.composer.registry.names))
        return self.composer.compose(text, domain, calls)


def language_primitives() -> PrimitiveRegistry:
    """Return the reference language primitive set, qualified and frozen."""

    registry = PrimitiveRegistry()
    qualification = PrimitiveQualification(4, 4, ("reference-contract-v1",))

    def observe(state: _Composition, args: Mapping[str, Any]) -> None:
        key, quote = str(args["key"]), str(args["quote"])
        start, end = state.span(quote)
        reliability = float(args.get("reliability", 1.0))
        if not key or not 0.0 <= reliability <= 1.0:
            raise PrimitiveError("invalid observation arguments")
        state.evidence.append(
            TextEvidence(key, args["value"], start, end, reliability=reliability)
        )

    def suspect(state: _Composition, args: Mapping[str, Any]) -> None:
        item_id, quote = str(args["id"]), str(args["quote"])
        start, end = state.span(quote)
        if not item_id:
            raise PrimitiveError("suspicion requires an ID")
        state.suspicions.append((item_id, quote, start, end))

    def action(state: _Composition, args: Mapping[str, Any]) -> None:
        action_id = str(args["id"])
        if not action_id or action_id in state.actions:
            raise PrimitiveError(f"duplicate or empty action: {action_id!r}")
        state.actions[action_id] = ActionNode(
            action_id,
            str(args["label"]),
            ActionKind(str(args["kind"]).lower()),
            information_value=float(args.get("information_value", 0.0)),
            cost=float(args.get("cost", 0.0)),
        )

    def prerequisite(state: _Composition, args: Mapping[str, Any]) -> None:
        state.before.append((str(args["source"]), str(args["target"])))

    registry.install("extract_observation", observe, qualification)
    registry.install("register_suspicion", suspect, qualification)
    registry.install("propose_action", action, qualification)
    registry.install("add_prerequisite", prerequisite, qualification)
    return registry
