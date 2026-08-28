"""Metamorpher's public, model-relative control primitives.

The package deliberately keeps importing :mod:`metamorpher` lightweight.  The
controller and optional accelerator backends are loaded lazily so an evolving
or partially installed research checkout can still use the stable data model,
CLI diagnostics, and CPU reference components.
"""

from __future__ import annotations

from importlib import import_module, metadata
from typing import Any

from .adapters import (
    ActionExecutor,
    ExecutionResult,
    ObservationSource,
    RelevanceProposer,
)
from .audit import AuditPolicy
from .evidence import EvidenceLedger, ResolvedFact
from .graph import FrontierResult, TypedActionGraph
from .memory import DomainMemory, MemoryRecord
from .model import (
    ActionKind,
    ActionNode,
    ActionStatus,
    ClaimStatus,
    ClaimTier,
    ClassStatus,
    Constraint,
    ConstraintKind,
    ControllerState,
    Decision,
    DecisionStatus,
    DomainTag,
    Observation,
    ObservationStatus,
    TruthValue,
)
from .policy import DecisionPolicy, HeuristicLookaheadPolicy
from .version_space import Hypothesis, UnresolvedCell, VersionSpaceManager

try:
    __version__ = metadata.version("metamorpher-control")
except metadata.PackageNotFoundError:
    __version__ = "0.1.0"


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # The controller may be absent while developing from a partial checkout.
    "MetamorpherController": (".controller", "MetamorpherController"),
    "ControllerCheckpoint": (".controller", "ControllerCheckpoint"),
    "ObservationReceipt": (".controller", "ObservationReceipt"),
    "FailureCarver": (".carving", "FailureCarver"),
    "AdaptiveFailureCarver": (".carving", "AdaptiveFailureCarver"),
    "CarvingResult": (".carving", "CarvingResult"),
    "CarvedBranch": (".carving", "CarvedBranch"),
    "OutcomeSupport": (".carving", "OutcomeSupport"),
    "ConstraintRevision": (".carving", "ConstraintRevision"),
    "RevisionResult": (".carving", "RevisionResult"),
    "BatchDecisionResult": (".batch", "BatchDecisionResult"),
    "CompiledBatch": (".batch", "CompiledBatch"),
    "GraphBatchCompiler": (".batch", "GraphBatchCompiler"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    try:
        value = getattr(import_module(module_name, __name__), attribute)
    except (ImportError, AttributeError) as exc:
        raise AttributeError(
            f"{name} is not available in this Metamorpher build; "
            "the stable primitives and CLI remain usable"
        ) from exc
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "ActionExecutor",
    "AdaptiveFailureCarver",
    "ActionKind",
    "ActionNode",
    "ActionStatus",
    "AuditPolicy",
    "BatchDecisionResult",
    "CarvedBranch",
    "CarvingResult",
    "ClaimStatus",
    "ClaimTier",
    "ClassStatus",
    "CompiledBatch",
    "Constraint",
    "ConstraintKind",
    "ConstraintRevision",
    "ControllerCheckpoint",
    "ControllerState",
    "Decision",
    "DecisionPolicy",
    "DecisionStatus",
    "DomainMemory",
    "DomainTag",
    "EvidenceLedger",
    "ExecutionResult",
    "FailureCarver",
    "FrontierResult",
    "GraphBatchCompiler",
    "HeuristicLookaheadPolicy",
    "Hypothesis",
    "MemoryRecord",
    "MetamorpherController",
    "Observation",
    "ObservationReceipt",
    "ObservationSource",
    "ObservationStatus",
    "OutcomeSupport",
    "RelevanceProposer",
    "ResolvedFact",
    "RevisionResult",
    "TruthValue",
    "TypedActionGraph",
    "UnresolvedCell",
    "VersionSpaceManager",
    "__version__",
]
