from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .model import Observation, ObservationStatus, TruthValue


@dataclass(frozen=True, slots=True)
class ResolvedFact:
    key: str
    state: TruthValue
    value: Any = None
    evidence_ids: tuple[str, ...] = ()
    confidence: float = 0.0


class EvidenceLedger:
    """Append-only evidence. Censored records are never negative evidence."""

    def __init__(self, ambiguity_margin: float = 0.15) -> None:
        self._events: list[Observation] = []
        self._ids: set[str] = set()
        self._by_key: dict[str, list[Observation]] = defaultdict(list)
        self.revision = 0
        self.ambiguity_margin = ambiguity_margin

    @property
    def events(self) -> tuple[Observation, ...]:
        return tuple(self._events)

    def append(self, observation: Observation) -> bool:
        if observation.id in self._ids:
            return False
        if not 0.0 <= observation.reliability <= 1.0:
            raise ValueError("reliability must be in [0, 1]")
        self._ids.add(observation.id)
        self._events.append(observation)
        self._by_key[observation.key].append(observation)
        self.revision += 1
        return True

    def records(self, key: str, *, include_censored: bool = True) -> tuple[Observation, ...]:
        values = self._by_key.get(key, ())
        if include_censored:
            return tuple(values)
        return tuple(x for x in values if x.status != ObservationStatus.CENSORED)

    def resolve(self, key: str) -> ResolvedFact:
        records = [x for x in self._by_key.get(key, ()) if x.status != ObservationStatus.CENSORED]
        if not records:
            return ResolvedFact(key, TruthValue.UNKNOWN)
        weights: dict[str, float] = defaultdict(float)
        originals: dict[str, Any] = {}
        ids: dict[str, list[str]] = defaultdict(list)
        for item in records:
            marker = repr(item.value)
            originals[marker] = item.value
            weight = item.reliability * (0.75 if item.status == ObservationStatus.INFERRED else 1.0)
            weights[marker] += weight
            ids[marker].append(item.id)
        ordered = sorted(weights.items(), key=lambda kv: (-kv[1], kv[0]))
        total = sum(weights.values())
        top_key, top_weight = ordered[0]
        second = ordered[1][1] if len(ordered) > 1 else 0.0
        confidence = top_weight / total if total else 0.0
        if len(ordered) > 1 and (top_weight - second) / max(total, 1e-12) < self.ambiguity_margin:
            return ResolvedFact(key, TruthValue.UNKNOWN, evidence_ids=tuple(x.id for x in records), confidence=confidence)
        return ResolvedFact(key, TruthValue.SATISFIED, originals[top_key], tuple(ids[top_key]), confidence)

    def evaluate(self, key: str, expected: Any) -> ResolvedFact:
        fact = self.resolve(key)
        if fact.state == TruthValue.UNKNOWN:
            return fact
        state = TruthValue.SATISFIED if fact.value == expected else TruthValue.VIOLATED
        return ResolvedFact(key, state, fact.value, fact.evidence_ids, fact.confidence)

    def snapshot(self) -> tuple[Observation, ...]:
        return self.events

    @classmethod
    def from_events(cls, events: Iterable[Observation]) -> EvidenceLedger:
        ledger = cls()
        for event in events:
            ledger.append(event)
        return ledger
