from __future__ import annotations

from dataclasses import dataclass, field

from .model import DomainTag


@dataclass(slots=True)
class MemoryRecord:
    claim: str
    domain: DomainTag
    positive: float = 1.0
    negative: float = 1.0
    censored: int = 0
    evidence_ids: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return self.positive / (self.positive + self.negative)


class DomainMemory:
    def __init__(self) -> None:
        self._records: dict[tuple[str, DomainTag], MemoryRecord] = {}

    def record(self, claim: str, domain: DomainTag, outcome: bool | None, evidence_id: str, *, censored: bool = False) -> MemoryRecord:
        key = (claim, domain)
        record = self._records.setdefault(key, MemoryRecord(claim, domain))
        if evidence_id in record.evidence_ids:
            return record
        record.evidence_ids.append(evidence_id)
        if censored or outcome is None:
            record.censored += 1
            return record
        if outcome:
            record.positive += 1.0
        else:
            record.negative += 1.0
        return record

    def get(self, claim: str, domain: DomainTag) -> MemoryRecord | None:
        return self._records.get((claim, domain))

    def candidates(self, claim: str) -> tuple[MemoryRecord, ...]:
        return tuple(v for (c, _), v in self._records.items() if c == claim)

    def for_domain(self, domain: DomainTag) -> tuple[MemoryRecord, ...]:
        return tuple(
            record
            for (_, record_domain), record in sorted(
                self._records.items(), key=lambda item: item[0][0]
            )
            if record_domain == domain
        )
