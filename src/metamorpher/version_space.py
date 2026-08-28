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

    @property
    def surviving_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.hypotheses))

    def common_safe_actions(self) -> frozenset[str]:
        if not self.hypotheses:
            return frozenset()
        iterator = iter(self.hypotheses.values())
        common = set(next(iterator).safe_actions)
        for hypothesis in iterator:
            common.intersection_update(hypothesis.safe_actions)
        return frozenset(common)

    def observe(self, test_id: str, value: Any, evidence_id: str) -> None:
        if evidence_id in self.evidence_ids:
            return
        self.evidence_ids.add(evidence_id)
        self.observations.append((test_id, value, evidence_id))
        compatible = {
            hid: h for hid, h in self.hypotheses.items()
            if test_id not in h.predictions or h.predictions[test_id] == value
        }
        # Empty compatibility widens uncertainty; it never invents a winner.
        if compatible:
            self.hypotheses = compatible
        else:
            # The represented space failed to explain the observation.  Keep
            # every surviving possibility and widen the epistemic status; an
            # empty match is not permission to choose a winner.
            self.status = ClassStatus.UNRESOLVED
            return
        if len(self.hypotheses) == 1 and len(self.observations) >= self.min_resolution_support:
            self.status = ClassStatus.SUPPORTED
        elif len(self.hypotheses) > 1:
            self.status = ClassStatus.UNRESOLVED


class VersionSpaceManager:
    def __init__(self) -> None:
        self.cells: dict[str, UnresolvedCell] = {}
        self.active_cell_id: str | None = None

    @property
    def active(self) -> UnresolvedCell | None:
        return self.cells.get(self.active_cell_id) if self.active_cell_id else None

    def add(self, cell: UnresolvedCell, *, activate: bool = False) -> None:
        if cell.id in self.cells:
            raise ValueError(f"duplicate unresolved cell: {cell.id}")
        self.cells[cell.id] = cell
        if activate:
            self.active_cell_id = cell.id

    def activate(self, cell_id: str | None) -> None:
        if cell_id is not None and cell_id not in self.cells:
            raise KeyError(cell_id)
        self.active_cell_id = cell_id

    def observe(self, test_id: str, value: Any, evidence_id: str) -> None:
        if self.active is not None:
            self.active.observe(test_id, value, evidence_id)
