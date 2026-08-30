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
from .representation import (
    DiscriminatingPrediction,
    ExpansionCapsule,
    ExpansionRegistry,
    ExpansionStatus,
    LocalEvidencePacket,
    ProjectionMapping,
    RepresentationBoundary,
    RepresentationStatus,
    ResidualSignature,
)
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
    "AdaptiveLearningLoop": (".carving", "AdaptiveLearningLoop"),
    "AdaptiveLearningRouter": (".carving", "AdaptiveLearningRouter"),
    "CarvingResult": (".carving", "CarvingResult"),
    "CarvedBranch": (".carving", "CarvedBranch"),
    "OutcomeSupport": (".carving", "OutcomeSupport"),
    "ConstraintRevision": (".carving", "ConstraintRevision"),
    "RevisionResult": (".carving", "RevisionResult"),
    "BatchDecisionResult": (".batch", "BatchDecisionResult"),
    "CompiledBatch": (".batch", "CompiledBatch"),
    "GraphBatchCompiler": (".batch", "GraphBatchCompiler"),
    "CandidateStructure": (".cognition", "CandidateStructure"),
    "CapsuleStore": (".cognition", "CapsuleStore"),
    "CognitiveLoop": (".cognition", "CognitiveLoop"),
    "CognitiveIngestion": (".cognition", "CognitiveIngestion"),
    "CognitiveStep": (".cognition", "CognitiveStep"),
    "Discriminator": (".cognition", "Discriminator"),
    "InMemoryCapsuleStore": (".cognition", "InMemoryCapsuleStore"),
    "Perceiver": (".cognition", "Perceiver"),
    "Perception": (".cognition", "Perception"),
    "Proposer": (".cognition", "Proposer"),
    "RefinementProposal": (".cognition", "RefinementProposal"),
    "StructuralCapsule": (".cognition", "StructuralCapsule"),
    "StructureLearner": (".cognition", "StructureLearner"),
    "GroundedTextRenderer": (".text", "GroundedTextRenderer"),
    "DiscourseInterpretation": (".text", "DiscourseInterpretation"),
    "DiscourseInterpreter": (".text", "DiscourseInterpreter"),
    "DiscoursePerceiver": (".text", "DiscoursePerceiver"),
    "DiscourseProposer": (".text", "DiscourseProposer"),
    "InterpreterPerceiver": (".text", "InterpreterPerceiver"),
    "TextEvidence": (".text", "TextEvidence"),
    "TextInterpreter": (".text", "TextInterpreter"),
    "TextProgramCompiler": (".dsl", "TextProgramCompiler"),
    "TextProgramError": (".dsl", "TextProgramError"),
    "ContributorRouter": (".primitives", "ContributorRouter"),
    "PrimitiveCall": (".primitives", "PrimitiveCall"),
    "PrimitiveComposer": (".primitives", "PrimitiveComposer"),
    "PrimitiveDiscourseInterpreter": (".primitives", "PrimitiveDiscourseInterpreter"),
    "PrimitiveError": (".primitives", "PrimitiveError"),
    "PrimitiveQualification": (".primitives", "PrimitiveQualification"),
    "PrimitiveRecord": (".primitives", "PrimitiveRecord"),
    "PrimitiveRegistry": (".primitives", "PrimitiveRegistry"),
    "language_primitives": (".primitives", "language_primitives"),
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
    "ActionKind",
    "ActionNode",
    "ActionStatus",
    "AdaptiveFailureCarver",
    "AdaptiveLearningLoop",
    "AdaptiveLearningRouter",
    "AuditPolicy",
    "BatchDecisionResult",
    "CandidateStructure",
    "CapsuleStore",
    "CarvedBranch",
    "CarvingResult",
    "ClaimStatus",
    "ClaimTier",
    "ClassStatus",
    "CognitiveIngestion",
    "CognitiveLoop",
    "CognitiveStep",
    "CompiledBatch",
    "Constraint",
    "ConstraintKind",
    "ConstraintRevision",
    "ContributorRouter",
    "ControllerCheckpoint",
    "ControllerState",
    "Decision",
    "DecisionPolicy",
    "DecisionStatus",
    "DiscourseInterpretation",
    "DiscourseInterpreter",
    "DiscoursePerceiver",
    "DiscourseProposer",
    "DiscriminatingPrediction",
    "Discriminator",
    "DomainMemory",
    "DomainTag",
    "EvidenceLedger",
    "ExecutionResult",
    "ExpansionCapsule",
    "ExpansionRegistry",
    "ExpansionStatus",
    "FailureCarver",
    "FrontierResult",
    "GraphBatchCompiler",
    "GroundedTextRenderer",
    "HeuristicLookaheadPolicy",
    "Hypothesis",
    "InMemoryCapsuleStore",
    "InterpreterPerceiver",
    "LocalEvidencePacket",
    "MemoryRecord",
    "MetamorpherController",
    "Observation",
    "ObservationReceipt",
    "ObservationSource",
    "ObservationStatus",
    "OutcomeSupport",
    "Perceiver",
    "Perception",
    "PrimitiveCall",
    "PrimitiveComposer",
    "PrimitiveDiscourseInterpreter",
    "PrimitiveError",
    "PrimitiveQualification",
    "PrimitiveRecord",
    "PrimitiveRegistry",
    "ProjectionMapping",
    "Proposer",
    "RefinementProposal",
    "RelevanceProposer",
    "RepresentationBoundary",
    "RepresentationStatus",
    "ResidualSignature",
    "ResolvedFact",
    "RevisionResult",
    "StructuralCapsule",
    "StructureLearner",
    "TextEvidence",
    "TextInterpreter",
    "TextProgramCompiler",
    "TextProgramError",
    "TruthValue",
    "TypedActionGraph",
    "UnresolvedCell",
    "VersionSpaceManager",
    "__version__",
    "language_primitives",
]
