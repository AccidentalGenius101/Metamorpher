"""Issuance and validation helpers for revision-bound decisions.

The controller is deliberately the authority that remembers which decision was
issued.  A token is therefore an identifier, not a bearer credential.  The
deterministic digest keeps traces reproducible while the epoch and evidence
revision make stale decisions explicit.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from .evidence import EvidenceLedger
from .graph import TypedActionGraph
from .model import Decision, DomainTag, StaleDecisionError


def decision_token(
    *,
    sequence: int,
    graph_epoch: int,
    evidence_revision: int,
    domain: DomainTag,
    status: str,
    action_id: str | None,
    probe_id: str | None,
    frontier: Iterable[str],
) -> str:
    """Return a stable opaque identifier for one exact decision snapshot."""

    fields = (
        str(sequence),
        str(graph_epoch),
        str(evidence_revision),
        repr(domain),
        status,
        action_id or "",
        probe_id or "",
        *sorted(frontier),
    )
    return hashlib.sha256("\x1f".join(fields).encode("utf-8")).hexdigest()


def assert_fresh(
    decision: Decision,
    graph: TypedActionGraph,
    evidence: EvidenceLedger,
) -> None:
    """Reject a decision made against a different structural/evidence state."""

    if decision.graph_epoch != graph.epoch:
        raise StaleDecisionError(
            "decision graph epoch is stale: "
            f"issued={decision.graph_epoch}, current={graph.epoch}"
        )
    if decision.evidence_revision != evidence.revision:
        raise StaleDecisionError(
            "decision evidence revision is stale: "
            f"issued={decision.evidence_revision}, current={evidence.revision}"
        )


def supporting_evidence_ids(
    graph: TypedActionGraph,
    evidence: EvidenceLedger,
    action_id: str,
) -> tuple[str, ...]:
    """Collect evidence that supports the constraints on ``action_id``.

    Constraint provenance may include non-evidence references (documents,
    policies, human approvals), so only identifiers present in the ledger are
    returned in this field.  Full provenance remains on the constraint itself.
    """

    ledger_ids = {item.id for item in evidence.events}
    result: set[str] = set()
    for constraint in graph.constraints_for(action_id):
        result.update(x for x in constraint.provenance if x in ledger_ids)
        result.update(
            x for x in constraint.supporting_evidence_ids if x in ledger_ids
        )
        if constraint.fact_key is not None:
            resolved = evidence.evaluate(
                constraint.fact_key,
                constraint.expected_value,
            )
            result.update(resolved.evidence_ids)
    return tuple(sorted(result))
