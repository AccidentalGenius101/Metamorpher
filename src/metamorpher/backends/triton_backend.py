"""Optional CUDA/Triton data-plane backend.

Torch and Triton are deliberately lazy imports.  This module can be imported on
a CPU-only installation and reports itself unavailable instead of making CUDA a
package requirement.
"""

import importlib.util
from numbers import Real
from typing import Any

from .base import AccelerationBackend, BackendDoctor

_KERNELS: tuple[Any, Any] | None = None


def _runtime() -> tuple[Any, Any]:
    try:
        import torch
        import triton
    except (ImportError, OSError) as exc:
        raise RuntimeError("Torch/Triton CUDA backend is unavailable") from exc
    return torch, triton


def _kernels() -> tuple[Any, Any]:
    """Create JIT kernels after (and only after) Triton is requested."""

    global _KERNELS, tl
    if _KERNELS is not None:
        return _KERNELS

    _, triton = _runtime()
    import triton.language as tl

    @triton.jit
    def hard_frontier_kernel(
        pending_ptr,
        completed_ptr,
        prerequisites_ptr,
        output_ptr,
        action_count,
        BLOCK_ACTIONS: tl.constexpr,
    ):
        target_flat = tl.program_id(0)
        batch_id = target_flat // action_count
        source_offsets = tl.arange(0, BLOCK_ACTIONS)
        source_mask = source_offsets < action_count
        dependency_offsets = target_flat * action_count + source_offsets
        dependencies = tl.load(
            prerequisites_ptr + dependency_offsets,
            mask=source_mask,
            other=0,
        ).to(tl.int1)
        completed_sources = tl.load(
            completed_ptr + batch_id * action_count + source_offsets,
            mask=source_mask,
            other=0,
        ).to(tl.int1)
        unmet_count = tl.sum(
            (dependencies & (~completed_sources)).to(tl.int32), axis=0
        )
        pending = tl.load(pending_ptr + target_flat).to(tl.int1)
        target_completed = tl.load(completed_ptr + target_flat).to(tl.int1)
        tl.store(output_ptr + target_flat, pending & (~target_completed) & (unmet_count == 0))

    @triton.jit
    def fused_score_kernel(
        frontier_ptr,
        features_ptr,
        weights_ptr,
        bias_ptr,
        output_ptr,
        feature_count,
        HAS_BIAS: tl.constexpr,
        BLOCK_FEATURES: tl.constexpr,
    ):
        action_flat = tl.program_id(0)
        feature_offsets = tl.arange(0, BLOCK_FEATURES)
        feature_mask = feature_offsets < feature_count
        offset = action_flat * feature_count + feature_offsets
        values = tl.load(features_ptr + offset, mask=feature_mask, other=0.0)
        weights = tl.load(weights_ptr + feature_offsets, mask=feature_mask, other=0.0)
        # Self-equality is Triton's portable NaN predicate inside JIT kernels.
        finite_values = (values == values) & (  # noqa: PLR0124
            tl.abs(values) != float("inf")
        )
        safe_values = tl.where(finite_values, values, 0.0)
        invalid_values = tl.sum(
            ((~finite_values) & feature_mask).to(tl.int32), axis=0
        )
        score = tl.sum(safe_values * weights, axis=0)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + action_flat)
            bias_finite = (bias == bias) & (  # noqa: PLR0124
                tl.abs(bias) != float("inf")
            )
            score += tl.where(bias_finite, bias, 0.0)
        else:
            bias_finite = True
        score_finite = (score == score) & (  # noqa: PLR0124
            tl.abs(score) != float("inf")
        )
        frontier = tl.load(frontier_ptr + action_flat).to(tl.int1)
        valid = frontier & (invalid_values == 0) & bias_finite & score_finite
        tl.store(output_ptr + action_flat, tl.where(valid, score, -float("inf")))

    _KERNELS = hard_frontier_kernel, fused_score_kernel
    return _KERNELS


class TritonBackend(AccelerationBackend):
    """CUDA implementation of frontier masking and masked linear scoring."""

    name = "triton"
    accelerated = True

    def __init__(self, device: str = "cuda") -> None:
        self.device = device

    def is_available(self) -> bool:
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("triton") is None:
            return False
        try:
            torch, _ = _runtime()
            if not torch.cuda.is_available():
                return False
            torch.empty(1, device=self.device)
        except (RuntimeError, AssertionError, ValueError, OSError):
            return False
        return True

    def doctor(self) -> BackendDoctor:
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("triton") is None:
            return BackendDoctor(
                name=self.name,
                available=False,
                accelerated=True,
                device=self.device,
                dependencies=(
                    ("torch", "missing" if importlib.util.find_spec("torch") is None else "present"),
                    ("triton", "missing" if importlib.util.find_spec("triton") is None else "present"),
                ),
                reason="Install the 'cuda' extra to enable Triton acceleration.",
            )
        try:
            torch, triton = _runtime()
            cuda = bool(torch.cuda.is_available())
            available = self.is_available()
            reason = None
            device_name = "unavailable"
            if available:
                device_name = str(torch.cuda.get_device_name(torch.device(self.device)))
            elif not cuda:
                reason = "Torch and Triton are installed, but CUDA is unavailable."
            else:
                reason = f"CUDA device {self.device!r} cannot be initialized."
            return BackendDoctor(
                name=self.name,
                available=available,
                accelerated=True,
                device=self.device,
                dependencies=(
                    ("torch", str(torch.__version__)),
                    ("triton", str(triton.__version__)),
                    ("cuda_available", str(cuda).lower()),
                    ("device_name", device_name),
                ),
                reason=reason,
            )
        except (RuntimeError, AssertionError, ValueError, OSError) as exc:
            return BackendDoctor(
                name=self.name,
                available=False,
                accelerated=True,
                device=self.device,
                reason=f"CUDA runtime check failed: {type(exc).__name__}: {exc}",
            )

    def _tensor(self, value: Any, *, dtype: Any | None = None) -> Any:
        torch, _ = _runtime()
        try:
            return torch.as_tensor(value, dtype=dtype, device=self.device).contiguous()
        except (TypeError, ValueError) as exc:
            raise TypeError("input cannot be converted to a dense CUDA tensor") from exc

    def _binary(self, value: Any, name: str, *, rank: int) -> Any:
        torch, _ = _runtime()
        array = self._tensor(value)
        if array.ndim != rank:
            raise ValueError(f"{name} must be rank {rank}, got shape {tuple(array.shape)}")
        if any(size == 0 for size in array.shape):
            raise ValueError(f"{name} must have no empty dimensions")
        if array.dtype == torch.bool:
            return array
        if array.is_complex() or not (array.is_floating_point() or array.dtype in {
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
        }):
            raise TypeError(f"{name} must contain booleans or numeric 0/1 values")
        valid = torch.isfinite(array) & ((array == 0) | (array == 1))
        if not bool(torch.all(valid).item()):
            raise ValueError(f"{name} must contain only booleans or finite 0/1 values")
        return array.to(dtype=torch.bool)

    def _real_tensor(self, value: Any, name: str) -> Any:
        torch, _ = _runtime()
        array = self._tensor(value)
        if array.dtype == torch.bool or array.is_complex() or not (
            array.is_floating_point()
            or array.dtype
            in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
        ):
            raise TypeError(f"{name} must contain real numeric values")
        return array.to(dtype=torch.float32)

    @staticmethod
    def _block_size(triton: Any, count: int, name: str) -> int:
        block = int(triton.next_power_of_2(count))
        if block > 65536:
            raise ValueError(f"{name} is too large for this Triton kernel")
        return block

    def hard_frontier_mask(
        self,
        pending: Any,
        completed: Any,
        hard_prerequisites: Any,
    ) -> Any:
        torch, triton = _runtime()
        if not self.is_available():
            raise RuntimeError("Triton backend is not available on the requested device")
        pending_array = self._binary(pending, "pending", rank=2)
        completed_array = self._binary(completed, "completed", rank=2)
        prerequisites = self._binary(
            hard_prerequisites, "hard_prerequisites", rank=3
        )
        if tuple(completed_array.shape) != tuple(pending_array.shape):
            raise ValueError("pending and completed must have identical shape")
        batch, actions = pending_array.shape
        if tuple(prerequisites.shape) != (batch, actions, actions):
            raise ValueError(
                "hard_prerequisites must have shape [batch, action, action]"
            )
        output = torch.empty_like(pending_array, dtype=torch.bool)
        frontier_kernel, _ = _kernels()
        block = self._block_size(triton, int(actions), "action dimension")
        frontier_kernel[(int(batch * actions),)](
            pending_array,
            completed_array,
            prerequisites,
            output,
            int(actions),
            BLOCK_ACTIONS=block,
        )
        return output

    def fused_scores(
        self,
        frontier: Any,
        features: Any,
        weights: Any,
        bias: Any | None = None,
    ) -> Any:
        torch, triton = _runtime()
        if not self.is_available():
            raise RuntimeError("Triton backend is not available on the requested device")
        frontier_array = self._binary(frontier, "frontier", rank=2)
        feature_array = self._real_tensor(features, "features")
        if feature_array.ndim != 3 or any(size == 0 for size in feature_array.shape):
            raise ValueError("features must have non-empty shape [batch, action, feature]")
        batch, actions = frontier_array.shape
        if tuple(feature_array.shape[:2]) != (batch, actions):
            raise ValueError("features must have shape [batch, action, feature]")

        weight_array = self._real_tensor(weights, "weights")
        if weight_array.ndim != 1 or weight_array.shape[0] != feature_array.shape[2]:
            raise ValueError("weights must have shape [features.shape[2]]")
        if not bool(torch.all(torch.isfinite(weight_array)).item()):
            raise ValueError("weights must contain only finite values")

        has_bias = bias is not None
        if bias is None:
            bias_array = torch.empty(1, device=self.device, dtype=torch.float32)
        elif isinstance(bias, Real) and not isinstance(bias, bool):
            bias_array = torch.full(
                (batch, actions), float(bias), device=self.device, dtype=torch.float32
            )
        else:
            bias_array = self._real_tensor(bias, "bias")
            if tuple(bias_array.shape) != (batch, actions):
                raise ValueError("bias must be scalar or have shape [batch, action]")

        output = torch.empty((batch, actions), device=self.device, dtype=torch.float32)
        _, score_kernel = _kernels()
        feature_count = int(feature_array.shape[2])
        block = self._block_size(triton, feature_count, "feature dimension")
        score_kernel[(int(batch * actions),)](
            frontier_array,
            feature_array,
            weight_array,
            bias_array,
            output,
            feature_count,
            HAS_BIAS=has_bias,
            BLOCK_FEATURES=block,
        )
        return output

    def select(self, scores: Any) -> Any:
        torch, _ = _runtime()
        score_array = self._real_tensor(scores, "scores")
        if score_array.ndim != 2 or any(size == 0 for size in score_array.shape):
            raise ValueError("scores must have non-empty shape [batch, action]")
        finite = torch.isfinite(score_array)
        sanitized = torch.where(finite, score_array, -torch.inf)
        result = torch.argmax(sanitized, dim=1).to(dtype=torch.int64)
        return torch.where(torch.any(finite, dim=1), result, torch.full_like(result, -1))


__all__ = ["TritonBackend"]
