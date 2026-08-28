"""Narrow integration boundaries for tools, sensors, and relevance proposers.

The runtime deliberately receives observations and execution results through
these interfaces.  Simulator truth and counterfactual outcomes do not belong
in a controller adapter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .model import ActionNode, Decision, DomainTag, Observation


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    action_id: str
    succeeded: bool
    observations: tuple[Observation, ...] = ()
    message: str = ""
    external_reference: str | None = None


class ActionExecutor(Protocol):
    """Executes a committed action in the external environment."""

    def execute(self, decision: Decision) -> ExecutionResult: ...


class ObservationSource(Protocol):
    """Produces evidence without exposing an environment's hidden state."""

    def collect(self, probe: ActionNode, domain: DomainTag) -> Sequence[Observation]: ...


class RelevanceProposer(Protocol):
    """Proposes candidates; proposals do not grant admissibility."""

    def propose(
        self,
        context: Mapping[str, Any],
        domain: DomainTag,
    ) -> Sequence[ActionNode]: ...
