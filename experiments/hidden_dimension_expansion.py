"""Mechanism benchmark for evidence-gated adoption of a supplied coordinate.

The parent model sees phase coordinates on a circle.  Some worlds also move
along an unrepresented axial coordinate, producing a helix.  A candidate
expansion may add cycle index, but it is promoted only when parent residuals are
structured and held-out evidence supports a prospective improvement.

This is a synthetic mechanism test, not evidence of real-world ontology
discovery.  Cycle index is the exact generating coordinate and is supplied to
the experiment.  Return evaluation uses an oracle regime label to select the
retained parent; autonomous routing is out of scope.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from metamorpher import (
    DiscriminatingPrediction,
    DomainTag,
    ExpansionCapsule,
    ExpansionRegistry,
    ExpansionStatus,
    Observation,
    ProjectionMapping,
    ResidualSignature,
)

DOMAIN = DomainTag("synthetic-phase-trajectory")


@dataclass(frozen=True, slots=True)
class Dataset:
    phase: np.ndarray
    cycle: np.ndarray
    target: np.ndarray


@dataclass(frozen=True, slots=True)
class WorldResult:
    seed: int
    world: str
    drift: float
    residual_cycle_correlation: float
    validation_fixed_mse: float
    validation_expanded_mse: float
    fixed_future_mse: float
    always_expand_future_mse: float
    gated_future_mse: float
    expansion_proposed: bool
    expansion_promoted: bool
    parent_return_mse: float
    destructive_return_mse: float


def make_dataset(
    rng: random.Random,
    cycles: range,
    *,
    phases_per_cycle: int,
    drift: float,
    noise: float,
) -> Dataset:
    phase: list[float] = []
    cycle_values: list[float] = []
    target: list[float] = []
    for cycle in cycles:
        for index in range(phases_per_cycle):
            theta = 2.0 * math.pi * index / phases_per_cycle
            phase.append(theta)
            cycle_values.append(float(cycle))
            target.append(
                0.30 * math.sin(theta)
                - 0.20 * math.cos(theta)
                + drift * cycle
                + rng.gauss(0.0, noise)
            )
    return Dataset(np.asarray(phase), np.asarray(cycle_values), np.asarray(target))


def design(data: Dataset, *, expanded: bool) -> np.ndarray:
    columns = [
        np.ones_like(data.phase),
        np.sin(data.phase),
        np.cos(data.phase),
    ]
    if expanded:
        columns.append(data.cycle)
    return np.column_stack(columns)


def fit(data: Dataset, *, expanded: bool) -> np.ndarray:
    matrix = design(data, expanded=expanded)
    return np.linalg.lstsq(matrix, data.target, rcond=None)[0]


def predict(data: Dataset, weights: np.ndarray, *, expanded: bool) -> np.ndarray:
    return design(data, expanded=expanded) @ weights


def mse(expected: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean((expected - predicted) ** 2))


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def expansion_capsule(seed: int, residuals: tuple[ResidualSignature, ...]) -> ExpansionCapsule:
    return ExpansionCapsule(
        f"phase-to-axial:{seed}",
        DOMAIN,
        "phase-circle",
        "phase-helix",
        ProjectionMapping(
            "phase-circle",
            "phase-helix",
            ("phase-sine", "phase-cosine"),
            coordinate_map=(("phase", "phase"),),
        ),
        residuals,
        (
            DiscriminatingPrediction(
                "held-out-improvement",
                "expansion_improves_held_out_prediction",
                True,
                "axial coordinate must improve unseen-cycle prediction",
            ),
        ),
        proposed_distinctions=("cycle-index / axial displacement",),
        rationale="parent residuals covary with traversal count",
    )


def run_world(
    seed: int,
    *,
    drift: float,
    noise: float,
    phases_per_cycle: int,
    correlation_threshold: float,
    improvement_ratio: float,
) -> WorldResult:
    rng = random.Random(seed)
    parent_train = make_dataset(
        rng, range(6), phases_per_cycle=phases_per_cycle, drift=0.0, noise=noise
    )
    shifted_train = make_dataset(
        rng, range(6), phases_per_cycle=phases_per_cycle, drift=drift, noise=noise
    )
    validation = make_dataset(
        rng, range(6, 8), phases_per_cycle=phases_per_cycle, drift=drift, noise=noise
    )
    future = make_dataset(
        rng, range(8, 12), phases_per_cycle=phases_per_cycle, drift=drift, noise=noise
    )
    # Return to the parent regime after the shifted segment. The benchmark uses
    # an oracle regime label to select the retained parent here; it does not test
    # autonomous return routing.
    returned_parent = make_dataset(
        rng, range(12, 16), phases_per_cycle=phases_per_cycle, drift=0.0, noise=noise
    )

    parent = fit(parent_train, expanded=False)
    expanded = fit(shifted_train, expanded=True)
    train_residuals = shifted_train.target - predict(
        shifted_train, parent, expanded=False
    )
    residual_correlation = correlation(train_residuals, shifted_train.cycle)
    validation_fixed = mse(
        validation.target, predict(validation, parent, expanded=False)
    )
    validation_expanded = mse(
        validation.target, predict(validation, expanded, expanded=True)
    )

    proposed = abs(residual_correlation) >= correlation_threshold
    promoted = False
    registry = ExpansionRegistry()
    if proposed:
        residuals = tuple(
            ResidualSignature(
                "axial-target",
                float(predicted),
                float(observed),
                f"train-residual:{seed}:{index}",
                (("cycle", float(cycle)), ("phase", float(phase))),
            )
            for index, (predicted, observed, cycle, phase) in enumerate(
                zip(
                    predict(shifted_train, parent, expanded=False),
                    shifted_train.target,
                    shifted_train.cycle,
                    shifted_train.phase,
                )
            )
        )
        capsule = expansion_capsule(seed, residuals)
        registry.propose(capsule)
        improvement_supported = validation_expanded < validation_fixed * improvement_ratio
        if improvement_supported:
            promoted_capsule = registry.promote(
                capsule.id,
                (
                    Observation(
                        f"held-out:{seed}",
                        "expansion_improves_held_out_prediction",
                        True,
                        source="synthetic-held-out-evaluator",
                        domain=DOMAIN,
                        independent_audit=False,
                    ),
                ),
            )
            promoted = promoted_capsule.status == ExpansionStatus.SUPPORTED
        else:
            registry.reject(capsule.id)

    fixed_future = mse(future.target, predict(future, parent, expanded=False))
    expanded_future = mse(future.target, predict(future, expanded, expanded=True))
    gated_future = expanded_future if promoted else fixed_future
    parent_return = mse(
        returned_parent.target,
        predict(returned_parent, parent, expanded=False),
    )
    destructive_return = mse(
        returned_parent.target,
        predict(returned_parent, expanded, expanded=True),
    )
    return WorldResult(
        seed,
        "helix" if drift else "circle",
        drift,
        residual_correlation,
        validation_fixed,
        validation_expanded,
        fixed_future,
        expanded_future,
        gated_future,
        proposed,
        promoted,
        parent_return,
        destructive_return,
    )


def summarize(results: list[WorldResult]) -> dict[str, Any]:
    circles = [item for item in results if item.world == "circle"]
    helices = [item for item in results if item.world == "helix"]

    def mean(items: list[float]) -> float:
        return statistics.fmean(items)

    return {
        "worlds": len(results),
        "circle_worlds": len(circles),
        "helix_worlds": len(helices),
        "helix_expansion_detection_rate": mean(
            [float(item.expansion_promoted) for item in helices]
        ),
        "circle_false_expansion_rate": mean(
            [float(item.expansion_promoted) for item in circles]
        ),
        "helix_fixed_future_mse": mean([item.fixed_future_mse for item in helices]),
        "helix_gated_future_mse": mean([item.gated_future_mse for item in helices]),
        "helix_always_expand_future_mse": mean(
            [item.always_expand_future_mse for item in helices]
        ),
        "circle_fixed_future_mse": mean([item.fixed_future_mse for item in circles]),
        "circle_gated_future_mse": mean([item.gated_future_mse for item in circles]),
        "circle_always_expand_future_mse": mean(
            [item.always_expand_future_mse for item in circles]
        ),
        "parent_return_mse": mean([item.parent_return_mse for item in helices]),
        "destructive_replacement_return_mse": mean(
            [item.destructive_return_mse for item in helices]
        ),
    }


def run_benchmark(
    *,
    seeds: int = 100,
    noise: float = 0.05,
    drift: float = 0.35,
    phases_per_cycle: int = 16,
    correlation_threshold: float = 0.65,
    improvement_ratio: float = 0.50,
) -> tuple[list[WorldResult], dict[str, Any]]:
    results: list[WorldResult] = []
    for seed in range(seeds):
        results.append(
            run_world(
                seed,
                drift=0.0,
                noise=noise,
                phases_per_cycle=phases_per_cycle,
                correlation_threshold=correlation_threshold,
                improvement_ratio=improvement_ratio,
            )
        )
        results.append(
            run_world(
                100_000 + seed,
                drift=drift,
                noise=noise,
                phases_per_cycle=phases_per_cycle,
                correlation_threshold=correlation_threshold,
                improvement_ratio=improvement_ratio,
            )
        )
    return results, summarize(results)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--seeds", type=int, default=100)
    value.add_argument("--noise", type=float, default=0.05)
    value.add_argument("--drift", type=float, default=0.35)
    value.add_argument("--phases-per-cycle", type=int, default=16)
    value.add_argument("--correlation-threshold", type=float, default=0.65)
    value.add_argument("--improvement-ratio", type=float, default=0.50)
    value.add_argument("--raw", action="store_true", help="include per-world records")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.seeds < 1 or args.phases_per_cycle < 4:
        raise SystemExit("seeds must be positive and phases-per-cycle at least four")
    results, summary = run_benchmark(
        seeds=args.seeds,
        noise=args.noise,
        drift=args.drift,
        phases_per_cycle=args.phases_per_cycle,
        correlation_threshold=args.correlation_threshold,
        improvement_ratio=args.improvement_ratio,
    )
    report: dict[str, Any] = {"configuration": vars(args), "summary": summary}
    if args.raw:
        report["world_results"] = [asdict(item) for item in results]
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
