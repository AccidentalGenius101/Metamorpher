"""Optional NumPy CPU backend.

NumPy is imported only when availability is inspected or an operation runs, so
importing :mod:`metamorpher.backends` never creates a NumPy dependency.
"""

from __future__ import annotations

import importlib.util
from numbers import Real
from typing import Any

from .base import AccelerationBackend, BackendDoctor


def _numpy() -> Any:
    try:
        import numpy as np
    except (ImportError, OSError) as exc:  # broken binary wheels also fail safely
        raise RuntimeError("NumPy backend is unavailable") from exc
    return np


class NumPyBackend(AccelerationBackend):
    """Vectorized CPU implementation with backend-native NumPy outputs."""

    name = "numpy"
    accelerated = True
    device = "cpu"

    def is_available(self) -> bool:
        if importlib.util.find_spec("numpy") is None:
            return False
        try:
            _numpy()
        except RuntimeError:
            return False
        return True

    def doctor(self) -> BackendDoctor:
        if not self.is_available():
            return BackendDoctor(
                name=self.name,
                available=False,
                accelerated=True,
                device=self.device,
                dependencies=(("numpy", "missing or unloadable"),),
                reason="Install the 'numpy' extra to enable the vectorized CPU backend.",
            )
        np = _numpy()
        return BackendDoctor(
            name=self.name,
            available=True,
            accelerated=True,
            device=self.device,
            dependencies=(("numpy", str(np.__version__)),),
        )

    @staticmethod
    def _binary(value: Any, name: str, *, rank: int) -> Any:
        np = _numpy()
        array = np.asarray(value)
        if array.ndim != rank:
            raise ValueError(f"{name} must be rank {rank}, got shape {array.shape}")
        if 0 in array.shape:
            raise ValueError(f"{name} must have no empty dimensions")
        if array.dtype.kind == "b":
            return np.ascontiguousarray(array)
        if array.dtype.kind not in "iuf":
            raise TypeError(f"{name} must contain booleans or numeric 0/1 values")
        try:
            finite = np.isfinite(array)
        except TypeError as exc:
            raise TypeError(f"{name} must contain booleans or numeric 0/1 values") from exc
        if not bool(np.all(finite)) or not bool(np.all((array == 0) | (array == 1))):
            raise ValueError(f"{name} must contain only booleans or finite 0/1 values")
        return np.ascontiguousarray(array.astype(bool, copy=False))

    @staticmethod
    def _features(value: Any) -> Any:
        np = _numpy()
        array = np.asarray(value)
        if array.ndim != 3 or 0 in array.shape:
            raise ValueError("features must have non-empty shape [batch, action, feature]")
        if array.dtype.kind not in "iuf":
            raise TypeError("features must contain real numeric values")
        return np.ascontiguousarray(array, dtype=np.float64)

    def hard_frontier_mask(
        self,
        pending: Any,
        completed: Any,
        hard_prerequisites: Any,
    ) -> Any:
        np = _numpy()
        pending_array = self._binary(pending, "pending", rank=2)
        completed_array = self._binary(completed, "completed", rank=2)
        prerequisites = self._binary(
            hard_prerequisites, "hard_prerequisites", rank=3
        )
        if completed_array.shape != pending_array.shape:
            raise ValueError("pending and completed must have identical shape")
        batch, actions = pending_array.shape
        if prerequisites.shape != (batch, actions, actions):
            raise ValueError(
                "hard_prerequisites must have shape [batch, action, action]"
            )
        unmet = prerequisites & ~completed_array[:, None, :]
        return np.ascontiguousarray(
            pending_array & ~completed_array & ~np.any(unmet, axis=2)
        )

    def fused_scores(
        self,
        frontier: Any,
        features: Any,
        weights: Any,
        bias: Any | None = None,
    ) -> Any:
        np = _numpy()
        frontier_array = self._binary(frontier, "frontier", rank=2)
        feature_array = self._features(features)
        batch, actions = frontier_array.shape
        if feature_array.shape[:2] != (batch, actions):
            raise ValueError("features must have shape [batch, action, feature]")

        weight_array = np.asarray(weights)
        if weight_array.ndim != 1 or weight_array.shape[0] != feature_array.shape[2]:
            raise ValueError("weights must have shape [features.shape[2]]")
        if weight_array.dtype.kind not in "iuf":
            raise TypeError("weights must contain real numeric values")
        weight_array = np.ascontiguousarray(weight_array, dtype=np.float64)
        if not bool(np.all(np.isfinite(weight_array))):
            raise ValueError("weights must contain only finite values")

        if bias is None:
            bias_array = np.zeros((batch, actions), dtype=np.float64)
        elif isinstance(bias, Real) and not isinstance(bias, bool):
            bias_array = np.full((batch, actions), float(bias), dtype=np.float64)
        else:
            bias_array = np.asarray(bias)
            if bias_array.shape != (batch, actions):
                raise ValueError("bias must be scalar or have shape [batch, action]")
            if bias_array.dtype.kind not in "iuf":
                raise TypeError("bias must contain real numeric values")
            bias_array = np.ascontiguousarray(bias_array, dtype=np.float64)

        finite_actions = np.all(np.isfinite(feature_array), axis=2) & np.isfinite(bias_array)
        # Sanitization prevents one malformed action from contaminating a full
        # vectorized reduction. It never makes that malformed action eligible.
        safe_features = np.where(np.isfinite(feature_array), feature_array, 0.0)
        with np.errstate(over="ignore", invalid="ignore"):
            scores = np.einsum("baf,f->ba", safe_features, weight_array) + np.where(
                np.isfinite(bias_array), bias_array, 0.0
            )
        valid = frontier_array & finite_actions & np.isfinite(scores)
        return np.ascontiguousarray(np.where(valid, scores, -np.inf))

    def select(self, scores: Any) -> Any:
        np = _numpy()
        score_array = np.asarray(scores)
        if score_array.ndim != 2 or 0 in score_array.shape:
            raise ValueError("scores must have non-empty shape [batch, action]")
        if score_array.dtype.kind not in "iuf":
            raise TypeError("scores must contain real numeric values")
        finite = np.isfinite(score_array)
        sanitized = np.where(finite, score_array, -np.inf)
        result = np.argmax(sanitized, axis=1).astype(np.int64, copy=False)
        result[~np.any(finite, axis=1)] = -1
        return result


__all__ = ["NumPyBackend"]
