"""Test whether predictive expansion must be resolved for control.

Both worlds contain the same evidence-supported helix coordinate. In the
descriptive-only world, candidate hidden states share a common-safe action. In
the control-relevant world they do not. The benchmark therefore separates
epistemic promotion from decision-time refinement.

Cycle index and action consequences are supplied by the synthetic generator.
This is a mechanism test, not autonomous coordinate or action-model discovery.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Literal

from metamorpher import Hypothesis, UnresolvedCell

try:
    from .hidden_dimension_expansion import run_world
except ImportError:  # Support direct execution from the repository root.
    from hidden_dimension_expansion import run_world

World = Literal["descriptive_only", "control_relevant"]
Policy = Literal["never_refine", "always_refine", "decision_gated"]
ACTIONS = ("coast", "stabilize")
FUTURE_CYCLES = tuple(range(8, 12))
CONTROL_BOUNDARY = 10


@dataclass(frozen=True, slots=True)
class ControlResult:
    seed: int
    world: World
    policy: Policy
    epistemic_expansion_promoted: bool
    fixed_future_mse: float
    expanded_future_mse: float
    common_safe_actions: tuple[str, ...]
    control_refinement_warranted: bool
    coordinate_measurements: int
    decisions: int
    action_errors: int
    unsafe_action_disagreements: int
    control_regret: float


def safe_action(world: World, cycle: int) -> str:
    if world == "descriptive_only":
        return "coast"
    return "coast" if cycle < CONTROL_BOUNDARY else "stabilize"


def control_cell(world: World) -> UnresolvedCell:
    low = Hypothesis(
        "cycle-below-boundary",
        frozenset({safe_action(world, CONTROL_BOUNDARY - 1)}),
        predictions={"cycle-region": "low"},
    )
    high = Hypothesis(
        "cycle-at-or-above-boundary",
        frozenset({safe_action(world, CONTROL_BOUNDARY)}),
        predictions={"cycle-region": "high"},
    )
    return UnresolvedCell("future-cycle-region", {low.id: low, high.id: high})


def selected_action(
    policy: Policy,
    world: World,
    cycle: int,
    *,
    expansion_promoted: bool,
) -> tuple[str, bool]:
    cell = control_cell(world)
    common = cell.common_safe_actions()
    if policy == "always_refine":
        return safe_action(world, cycle), True
    if policy == "decision_gated" and expansion_promoted and not common:
        return safe_action(world, cycle), True
    # The coarse policy uses a represented common-safe action when one exists.
    # With no common action, never-refine falls back to the pre-boundary action.
    return (next(iter(common)) if common else "coast"), False


def run_control_world(
    seed: int,
    world: World,
    policy: Policy,
    *,
    phases_per_cycle: int = 16,
    drift: float = 0.35,
    noise: float = 0.05,
    correlation_threshold: float = 0.65,
    improvement_ratio: float = 0.50,
) -> ControlResult:
    epistemic = run_world(
        100_000 + seed,
        drift=drift,
        noise=noise,
        phases_per_cycle=phases_per_cycle,
        correlation_threshold=correlation_threshold,
        improvement_ratio=improvement_ratio,
    )
    cell = control_cell(world)
    common = tuple(sorted(cell.common_safe_actions()))
    refinement_warranted = epistemic.expansion_promoted and not common
    measurements = 0
    errors = 0
    unsafe = 0
    regret = 0.0
    decisions = 0
    for cycle in FUTURE_CYCLES:
        for _ in range(phases_per_cycle):
            action, measured = selected_action(
                policy,
                world,
                cycle,
                expansion_promoted=epistemic.expansion_promoted,
            )
            expected = safe_action(world, cycle)
            decisions += 1
            measurements += int(measured)
            if action != expected:
                errors += 1
                unsafe += 1
                regret += 1.0
    return ControlResult(
        seed=seed,
        world=world,
        policy=policy,
        epistemic_expansion_promoted=epistemic.expansion_promoted,
        fixed_future_mse=epistemic.fixed_future_mse,
        expanded_future_mse=epistemic.gated_future_mse,
        common_safe_actions=common,
        control_refinement_warranted=refinement_warranted,
        coordinate_measurements=measurements,
        decisions=decisions,
        action_errors=errors,
        unsafe_action_disagreements=unsafe,
        control_regret=regret,
    )


def run_benchmark(
    *, seeds: int = 100
) -> tuple[list[ControlResult], dict[str, Any]]:
    if seeds < 1:
        raise ValueError("seeds must be positive")
    results = [
        run_control_world(seed, world, policy)
        for seed in range(seeds)
        for world in ("descriptive_only", "control_relevant")
        for policy in ("never_refine", "always_refine", "decision_gated")
    ]
    summary: dict[str, Any] = {}
    for world in ("descriptive_only", "control_relevant"):
        summary[world] = {}
        for policy in ("never_refine", "always_refine", "decision_gated"):
            selected = [
                result
                for result in results
                if result.world == world and result.policy == policy
            ]
            summary[world][policy] = {
                "epistemic_promotion_rate": statistics.fmean(
                    float(result.epistemic_expansion_promoted)
                    for result in selected
                ),
                "fixed_future_mse": statistics.fmean(
                    result.fixed_future_mse for result in selected
                ),
                "expanded_future_mse": statistics.fmean(
                    result.expanded_future_mse for result in selected
                ),
                "control_refinement_rate": statistics.fmean(
                    float(result.control_refinement_warranted)
                    for result in selected
                ),
                "mean_coordinate_measurements": statistics.fmean(
                    result.coordinate_measurements for result in selected
                ),
                "mean_action_error_rate": statistics.fmean(
                    result.action_errors / result.decisions for result in selected
                ),
                "mean_unsafe_action_disagreement_rate": statistics.fmean(
                    result.unsafe_action_disagreements / result.decisions
                    for result in selected
                ),
                "mean_control_regret": statistics.fmean(
                    result.control_regret for result in selected
                ),
            }
    return results, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    results, summary = run_benchmark(seeds=args.seeds)
    report: dict[str, Any] = {
        "configuration": {
            "seeds": args.seeds,
            "future_cycles": FUTURE_CYCLES,
            "control_boundary": CONTROL_BOUNDARY,
            "candidate_coordinate_supplied": True,
            "action_consequences_supplied": True,
        },
        "summary": summary,
    }
    if args.raw:
        report["results"] = [asdict(result) for result in results]
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
