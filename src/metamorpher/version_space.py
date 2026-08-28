from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .model import ClassStatus, DomainTag


@dataclass(frozen=True, slots=True)
class Hypothesis:
    id: str
    safe_actions: frozenset[str]
    predictions: Mapping[str, Any] = field(default_factory=dict)
    domain: DomainTag | None = None
    provenance: tuple[str, ...] = ()


@dataclass(slots=True)
class UnresolvedCell:
    """An observational cell; no shared hidden cause is implied."""

    id: str
    hypotheses: dict[str, Hypothesis]
    status: ClassStatus = ClassStatus.PROVISIONAL
    parent_id: str | None = None
    observations: list[tuple[str, Any, str]] = field(default_factory=list)
    evidence_ids: set[str] = field(default_factory=set)
    min_resolution_support: int = 1
    _original_hypotheses: dict[str, Hypothesis] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.min_resolution_support < 1:
            raise ValueError("min_resolution_support must be positive")
        if not self.hypotheses:
            raise ValueError("an unresolved cell requires at least one hypothesis")
        self._original_hypotheses = dict(self.hypotheses)

    @property
    def surviving_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.hypotheses))

    def hypotheses_for(self, domain: DomainTag | None = None) -> dict[str, Hypothesis]:
        """Return hypotheses applicable to ``domain``.

        An untagged hypothesis is portable. A tagged hypothesis applies only
        to the exact domain it names; merely storing domain provenance without
        enforcing it would let evidence from one regime constrain another.
        """

        if domain is None:
            return dict(self.hypotheses)
        return {
            hypothesis_id: hypothesis
            for hypothesis_id, hypothesis in self.hypotheses.items()
            if hypothesis.domain is None or hypothesis.domain == domain
        }

    def common_safe_actions(self, domain: DomainTag | None = None) -> frozenset[str]:
        applicable = self.hypotheses_for(domain)
        if not applicable:
            return frozenset()
        iterator = iter(applicable.values())
        common = set(next(iterator).safe_actions)
        for hypothesis in iterator:
            common.intersection_update(hypothesis.safe_actions)
        return frozenset(common)

    def observe(self, test_id: str, value: Any, evidence_id: str) -> None:
        if evidence_id in self.evidence_ids:
            return
        self.evidence_ids.add(evidence_id)
        self.observations.append((test_id, value, evidence_id))
        informative = [
            (observed_test, observed_value)
            for observed_test, observed_value, _ in self.observations
            if any(
                observed_test in hypothesis.predictions
                for hypothesis in self._original_hypotheses.values()
            )
        ]
        compatible = {
            hypothesis_id: hypothesis
            for hypothesis_id, hypothesis in self._original_hypotheses.items()
            if all(
                test not in hypothesis.predictions
                or hypothesis.predictions[test] == observed_value
                for test, observed_value in informative
            )
        }
        # Recompute from the original represented class on every observation.
        # Otherwise an early narrowing permanently deletes alternatives and a
        # later contradiction cannot truthfully reopen the equivalence class.
        if not compatible:
            self.hypotheses = dict(self._original_hypotheses)
            self.status = ClassStatus.UNRESOLVED
            return
        self.hypotheses = compatible
        if (
            len(self.hypotheses) == 1
            and len(informative) >= self.min_resolution_support
        ):
            self.status = ClassStatus.SUPPORTED
        elif len(self.hypotheses) > 1:
            self.status = ClassStatus.UNRESOLVED


class VersionSpaceManager:
    def __init__(self) -> None:
        self.cells: dict[str, UnresolvedCell] = {}
        self.active_cell_id: str | None = None
        self._active_by_domain: dict[DomainTag, str] = {}

    @property
    def active(self) -> UnresolvedCell | None:
        return self.cells.get(self.active_cell_id) if self.active_cell_id else None

    def active_for(self, domain: DomainTag | None) -> UnresolvedCell | None:
        if domain is not None and domain in self._active_by_domain:
            return self.cells.get(self._active_by_domain[domain])
        return self.active

    @staticmethod
    def _cell_domain(cell: UnresolvedCell) -> DomainTag | None:
        domains = {hypothesis.domain for hypothesis in cell.hypotheses.values()}
        if len(domains) == 1:
            return next(iter(domains))
        return None

    def _activate_cell(self, cell: UnresolvedCell) -> None:
        self.active_cell_id = cell.id
        domain = self._cell_domain(cell)
        if domain is not None:
            self._active_by_domain[domain] = cell.id

    def add(self, cell: UnresolvedCell, *, activate: bool = False) -> None:
        if cell.id in self.cells:
            raise ValueError(f"duplicate unresolved cell: {cell.id}")
        self.cells[cell.id] = cell
        if activate:
            self._activate_cell(cell)

    def upsert(self, cell: UnresolvedCell, *, activate: bool = False) -> None:
        """Install a newly learned revision of an observational cell."""

        self.cells[cell.id] = cell
        if activate:
            self._activate_cell(cell)

    def activate(self, cell_id: str | None) -> None:
        if cell_id is not None and cell_id not in self.cells:
            raise KeyError(cell_id)
        self.active_cell_id = cell_id
        if cell_id is not None:
            self._activate_cell(self.cells[cell_id])

    def observe(
        self,
        test_id: str,
        value: Any,
        evidence_id: str,
        domain: DomainTag | None = None,
    ) -> None:
        active = self.active_for(domain)
        if active is not None:
            active.observe(test_id, value, evidence_id)
