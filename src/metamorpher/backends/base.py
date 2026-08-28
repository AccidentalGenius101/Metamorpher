"""Backend-neutral numerical primitives for Metamorpher.

Backends accelerate two deliberately narrow data-plane operations:

* compute a hard-prerequisite frontier from already-resolved booleans; and
* compute a weighted action score while masking actions outside that frontier.

They do not resolve evidence, classify constraints, revise graphs, or certify an
action.  Those remain control-plane responsibilities.  In particular, no
backend accepts simulator truth or an oracle label.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

CAPABILITIES = ("batched_hard_frontier", "fused_masked_scoring")


@dataclass(frozen=True, slots=True)
class BackendDoctor:
    """Machine-readable backend availability and diagnostics."""

    name: str
    available: bool
    accelerated: bool
    device: str
    capabilities: tuple[str, ...] = CAPABILITIES
    dependencies: tuple[tuple[str, str], ...] = ()
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BatchedScoreResult:
    """Backend-native mask/scores plus backend-neutral selected indices.

    ``selected[b]`` is ``-1`` when batch item ``b`` has no finite frontier
    score.  Otherwise ties are resolved by the smallest action index.
    """

    frontier: Any
    scores: Any
    selected: Any


class AccelerationBackend(ABC):
    """Common API implemented by the reference and optional accelerators.

    Array layout is intentionally explicit:

    * ``pending`` and ``completed``: ``[batch, action]``
    * ``hard_prerequisites``: ``[batch, target_action, source_action]``
    * ``features``: ``[batch, action, feature]``
    * ``weights``: ``[feature]``
    * ``bias``: scalar or ``[batch, action]``

    A backend computes only prerequisites already classified as hard by the
    control plane.  Unknown guards and external safety policy must be handled
    before or after this primitive, never inferred here.
    """

    name = "abstract"
    accelerated = False
    device = "unknown"

    def is_available(self) -> bool:
        return True

    def doctor(self) -> BackendDoctor:
        return BackendDoctor(
            name=self.name,
            available=self.is_available(),
            accelerated=self.accelerated,
            device=self.device,
        )

    @abstractmethod
    def hard_frontier_mask(
        self,
        pending: Any,
        completed: Any,
        hard_prerequisites: Any,
    ) -> Any:
        """Return ``pending & ~completed & all(prerequisites completed)``."""

    @abstractmethod
    def fused_scores(
        self,
        frontier: Any,
        features: Any,
        weights: Any,
        bias: Any | None = None,
    ) -> Any:
        """Return weighted scores; non-frontier/non-finite actions become -inf."""

    @abstractmethod
    def select(self, scores: Any) -> Any:
        """Return one deterministic action index per batch, or -1."""

    def frontier_and_score(
        self,
        pending: Any,
        completed: Any,
        hard_prerequisites: Any,
        features: Any,
        weights: Any,
        bias: Any | None = None,
    ) -> BatchedScoreResult:
        frontier = self.hard_frontier_mask(pending, completed, hard_prerequisites)
        scores = self.fused_scores(frontier, features, weights, bias)
        return BatchedScoreResult(frontier, scores, self.select(scores))


def _to_nested(value: Any, name: str) -> Any:
    """Convert an array-like value without importing an optional dependency."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be an array-like sequence")
    return value


def _shape_2d(value: Any, name: str) -> tuple[list[list[Any]], int, int]:
    raw = _to_nested(value, name)
    rows = [list(row) if isinstance(row, (list, tuple)) else None for row in raw]
    if not rows or any(row is None for row in rows):
        raise ValueError(f"{name} must have shape [batch, action] with a non-empty batch")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError(f"{name} must be a non-ragged [batch, action] matrix")
    return rows, len(rows), width


def _shape_3d(value: Any, name: str) -> tuple[list[list[list[Any]]], int, int, int]:
    raw = _to_nested(value, name)
    batches: list[list[list[Any]]] = []
    for matrix in raw:
        if not isinstance(matrix, (list, tuple)):
            raise TypeError(f"{name} must be a rank-3 array")
        converted: list[list[Any]] = []
        for row in matrix:
            if not isinstance(row, (list, tuple)):
                raise TypeError(f"{name} must be a rank-3 array")
            converted.append(list(row))
        batches.append(converted)
    if not batches or not batches[0] or not batches[0][0]:
        raise ValueError(f"{name} must have non-empty shape [batch, target, source]")
    d1, d2 = len(batches[0]), len(batches[0][0])
    if any(len(matrix) != d1 for matrix in batches):
        raise ValueError(f"{name} must be non-ragged")
    if any(len(row) != d2 for matrix in batches for row in matrix):
        raise ValueError(f"{name} must be non-ragged")
    return batches, len(batches), d1, d2


def _binary(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, Real):
        numeric = float(value)
        if math.isfinite(numeric) and numeric in (0.0, 1.0):
            return bool(numeric)
    raise ValueError(f"{name} must contain only booleans or finite 0/1 values")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must contain real numeric values")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must contain only finite values")
    return result


class ReferenceBackend(AccelerationBackend):
    """Dependency-free CPU reference implementation.

    Feature or per-action bias NaNs/infinities invalidate only the affected
    action.  Non-finite weights are rejected because they invalidate the whole
    scoring configuration.
    """

    name = "reference"
    device = "cpu"

    def doctor(self) -> BackendDoctor:
        return BackendDoctor(
            name=self.name,
            available=True,
            accelerated=False,
            device=self.device,
            dependencies=(("python", "stdlib"),),
        )

    def hard_frontier_mask(
        self,
        pending: Any,
        completed: Any,
        hard_prerequisites: Any,
    ) -> list[list[bool]]:
        pending_rows, batch, actions = _shape_2d(pending, "pending")
        completed_rows, cb, ca = _shape_2d(completed, "completed")
        prerequisites, pb, targets, sources = _shape_3d(
            hard_prerequisites, "hard_prerequisites"
        )
        if (cb, ca) != (batch, actions):
            raise ValueError("pending and completed must have identical shape")
        if (pb, targets, sources) != (batch, actions, actions):
            raise ValueError(
                "hard_prerequisites must have shape [batch, action, action]"
            )

        p = [[_binary(x, "pending") for x in row] for row in pending_rows]
        c = [[_binary(x, "completed") for x in row] for row in completed_rows]
        deps = [
            [[_binary(x, "hard_prerequisites") for x in row] for row in matrix]
            for matrix in prerequisites
        ]
        result: list[list[bool]] = []
        for b in range(batch):
            row: list[bool] = []
            for target in range(actions):
                met = all(not deps[b][target][source] or c[b][source] for source in range(actions))
                row.append(p[b][target] and not c[b][target] and met)
            result.append(row)
        return result

    def fused_scores(
        self,
        frontier: Any,
        features: Any,
        weights: Any,
        bias: Any | None = None,
    ) -> list[list[float]]:
        frontier_rows, batch, actions = _shape_2d(frontier, "frontier")
        feature_rows, fb, fa, feature_count = _shape_3d(features, "features")
        if (fb, fa) != (batch, actions):
            raise ValueError("features must have shape [batch, action, feature]")

        raw_weights = _to_nested(weights, "weights")
        if any(isinstance(x, (list, tuple)) for x in raw_weights):
            raise ValueError("weights must have shape [feature]")
        if len(raw_weights) != feature_count:
            raise ValueError("weights length must equal features.shape[2]")
        numeric_weights = [_finite_number(x, "weights") for x in raw_weights]

        if bias is None:
            bias_rows = [[0.0] * actions for _ in range(batch)]
        elif isinstance(bias, Real) and not isinstance(bias, bool):
            bias_rows = [[bias] * actions for _ in range(batch)]
        else:
            bias_rows, bb, ba = _shape_2d(bias, "bias")
            if (bb, ba) != (batch, actions):
                raise ValueError("bias must be scalar or have shape [batch, action]")

        masks = [[_binary(x, "frontier") for x in row] for row in frontier_rows]
        for value in (
            item
            for matrix in feature_rows
            for row in matrix
            for item in row
        ):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("features must contain real numeric values")
        for value in (item for row in bias_rows for item in row):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError("bias must contain real numeric values")
        result = [[-math.inf] * actions for _ in range(batch)]
        for b in range(batch):
            for action in range(actions):
                if not masks[b][action]:
                    continue
                values = feature_rows[b][action]
                if any(not math.isfinite(float(x)) for x in values):
                    continue
                action_bias = bias_rows[b][action]
                if not math.isfinite(float(action_bias)):
                    continue
                try:
                    score = math.fsum(
                        float(x) * w for x, w in zip(values, numeric_weights)
                    )
                    score += float(action_bias)
                except (OverflowError, ValueError):
                    continue
                if math.isfinite(score):
                    result[b][action] = score
        return result

    def select(self, scores: Any) -> list[int]:
        rows, _, _ = _shape_2d(scores, "scores")
        selected: list[int] = []
        for row in rows:
            candidates = [
                (float(value), index)
                for index, value in enumerate(row)
                if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
            ]
            if not candidates:
                selected.append(-1)
                continue
            best = max(score for score, _ in candidates)
            selected.append(min(index for score, index in candidates if score == best))
        return selected


__all__ = [
    "CAPABILITIES",
    "AccelerationBackend",
    "BackendDoctor",
    "BatchedScoreResult",
    "ReferenceBackend",
]
