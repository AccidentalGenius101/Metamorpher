from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AuditPolicy:
    """Deterministic precommitted audits, independent of value scores."""

    every: int = 10
    budget: int | None = None
    used: int = 0

    def should_audit(self, case_index: int) -> bool:
        if self.every <= 0:
            return False
        if self.budget is not None and self.used >= self.budget:
            return False
        return case_index % self.every == 0

    def consume(self, case_index: int) -> bool:
        if not self.should_audit(case_index):
            return False
        self.used += 1
        return True
