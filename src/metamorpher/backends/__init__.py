"""Optional numerical backends with deterministic CPU fallback.

Typical use::

    backend = get_backend("auto")
    result = backend.frontier_and_score(
        pending, completed, prerequisites, features, weights
    )

Importing this package never imports NumPy, Torch, Triton, or initializes CUDA.
"""

from __future__ import annotations

from typing import Literal

from .base import (
    AccelerationBackend,
    BackendDoctor,
    BatchedScoreResult,
    ReferenceBackend,
)
from .numpy_backend import NumPyBackend
from .triton_backend import TritonBackend

BackendName = Literal["auto", "reference", "numpy", "triton"]


def _ordered_candidates(preference: BackendName, device: str | None) -> list[AccelerationBackend]:
    reference = ReferenceBackend()
    numpy = NumPyBackend()
    triton = TritonBackend(device=device or "cuda")
    if preference == "reference":
        return [reference]
    if preference == "numpy":
        return [numpy, reference]
    if preference == "triton":
        return [triton, numpy, reference]
    if preference != "auto":
        raise ValueError(f"unknown backend preference: {preference!r}")
    return [triton, numpy, reference]


def get_backend(
    preference: BackendName = "auto",
    *,
    device: str | None = None,
    strict: bool = False,
) -> AccelerationBackend:
    """Select an available backend without introducing hard dependencies.

    ``auto`` tries Triton/CUDA, then NumPy, then the dependency-free reference
    implementation.  Explicit optional preferences use the same safe fallback
    unless ``strict=True``.  The reference backend is always available.
    """

    candidates = _ordered_candidates(preference, device)
    first = candidates[0]
    if strict and not first.is_available():
        report = first.doctor()
        raise RuntimeError(report.reason or f"backend {first.name!r} is unavailable")
    for backend in candidates:
        if backend.is_available():
            return backend
    # Kept as a defensive invariant in case a custom reference implementation
    # is introduced later.
    raise RuntimeError("no Metamorpher numerical backend is available")


def doctor(*, device: str | None = None) -> tuple[BackendDoctor, ...]:
    """Return diagnostics for every backend without requiring optional extras."""

    return (
        ReferenceBackend().doctor(),
        NumPyBackend().doctor(),
        TritonBackend(device=device or "cuda").doctor(),
    )


def doctor_dict(*, device: str | None = None) -> dict[str, dict[str, object]]:
    """JSON-friendly diagnostics keyed by backend name."""

    return {report.name: report.as_dict() for report in doctor(device=device)}


__all__ = [
    "AccelerationBackend",
    "BackendDoctor",
    "BackendName",
    "BatchedScoreResult",
    "NumPyBackend",
    "ReferenceBackend",
    "TritonBackend",
    "doctor",
    "doctor_dict",
    "get_backend",
]
