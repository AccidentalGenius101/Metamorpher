"""Benchmark collision detection when raw inputs cannot identify the rule.

The stream first teaches three observable nonlinear regions, then supplies a
fourth regime with the XOR region's inputs and exactly inverted labels.  No
task or regime identifier is available to the tested learners.  An oracle with
privileged routing is reported only as an upper bound.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

try:
    from .continual_replay import (
        ReplayBuffer,
        TASKS,
        TaskData,
        TinyMLP,
        accuracy,
        make_task,
    )
except ImportError:  # Support direct execution from the repository root.
    from continual_replay import (
        ReplayBuffer,
        TASKS,
        TaskData,
        TinyMLP,
        accuracy,
        make_task,
    )

Method = Literal["always_replay", "boundary", "oracle"]
SCHEDULE = ("ring", "xor", "wave", "collision", "ring", "xor", "wave")


@dataclass(frozen=True, slots=True)
class BoundaryResult:
    seed: int
    method: Method
    resolved_accuracy: float
    xor_accuracy: float
    collision_accuracy: float
    coverage: float
    selective_accuracy: float
    boundary_detected: bool
    detection_delay_batches: int | None
    false_boundary_batches: int
    harmful_updates_after_detection: int
    replay_examples: int
    audit_examples: int


def collision_task(rng: np.random.Generator, samples: int) -> TaskData:
    normal = make_task(rng, "xor", samples)
    return TaskData(normal.features, 1.0 - normal.labels)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator == 0.0 else float(left @ right / denominator)


def local_audit(
    memory: ReplayBuffer,
    features: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select remembered observations near the current observable region."""

    distances = np.min(
        np.sum(
            (memory.features[: memory.size, None, :] - features[None, :, :]) ** 2,
            axis=2,
        ),
        axis=1,
    )
    indices = np.argsort(distances)[: min(count, memory.size)]
    return memory.features[indices], memory.labels[indices]


def contradiction_score(
    memory: ReplayBuffer,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    radius: float = 0.40,
) -> tuple[int, float]:
    """Count nearby remembered cases and their label-disagreement rate."""

    squared = np.sum(
        (features[:, None, :] - memory.features[None, : memory.size, :]) ** 2,
        axis=2,
    )
    nearest = np.argmin(squared, axis=1)
    distances = np.sqrt(squared[np.arange(len(features)), nearest])
    nearby = distances <= radius
    if not np.any(nearby):
        return 0, 0.0
    disagreement = memory.labels[nearest[nearby]] != labels[nearby]
    return int(np.sum(nearby)), float(np.mean(disagreement))


def in_unresolved_region(features: np.ndarray, center: np.ndarray | None) -> bool:
    if center is None:
        return False
    return float(np.linalg.norm(np.mean(features, axis=0) - center)) < 1.35


def run_method(
    seed: int,
    method: Method,
    *,
    phase_samples: int = 512,
    evaluation_samples: int = 800,
    batch_size: int = 16,
    memory_capacity: int = 192,
    hidden: int = 24,
    learning_rate: float = 0.20,
    updates_per_batch: int = 3,
    conflict_threshold: float = -0.05,
    damage_threshold: float = 0.005,
    persistence: int = 2,
) -> BoundaryResult:
    if method not in {"always_replay", "boundary", "oracle"}:
        raise ValueError(f"unknown method: {method}")
    rng = np.random.default_rng(seed)
    model = TinyMLP(np.random.default_rng(seed + 10_000), hidden=hidden)
    oracle_models = {
        name: TinyMLP(np.random.default_rng(seed + 30_000 + index), hidden=hidden)
        for index, name in enumerate((*TASKS, "collision"))
    }
    memory = ReplayBuffer(memory_capacity, np.random.default_rng(seed + 20_000))
    evaluation = {task: make_task(rng, task, evaluation_samples) for task in TASKS}
    collision_evaluation = TaskData(
        evaluation["xor"].features,
        1.0 - evaluation["xor"].labels,
    )
    phases = [
        collision_task(rng, phase_samples)
        if name == "collision"
        else make_task(rng, name, phase_samples)
        for name in SCHEDULE
    ]
    unresolved_center: np.ndarray | None = None
    conflict_streak = 0
    collision_batches_seen = 0
    detection_delay: int | None = None
    false_boundaries = 0
    harmful_after = 0
    replay_examples = 0
    audit_examples = 0

    for phase_name, phase in zip(SCHEDULE, phases):
        order = rng.permutation(phase_samples)
        for start in range(0, phase_samples, batch_size):
            indices = order[start : start + batch_size]
            current_x = phase.features[indices]
            current_y = phase.labels[indices]
            if phase_name == "collision":
                collision_batches_seen += 1

            if method == "oracle":
                target = oracle_models[phase_name]
                for _ in range(updates_per_batch):
                    target.train(current_x, current_y, learning_rate)
                continue

            if method == "boundary" and in_unresolved_region(
                current_x, unresolved_center
            ):
                continue

            snapshot = model.snapshot()
            conflict = False
            if method == "boundary" and memory.size >= batch_size:
                audit_x, audit_y = local_audit(memory, current_x, 48)
                audit_examples += len(audit_x)
                before = float(np.mean(model.losses(audit_x, audit_y)))
                current_gradient = model.gradient_vector(current_x, current_y)
                retained_gradient = model.gradient_vector(audit_x, audit_y)
                nearby, disagreement = contradiction_score(
                    memory, current_x, current_y
                )
                model.train(current_x, current_y, learning_rate)
                after = float(np.mean(model.losses(audit_x, audit_y)))
                relative_damage = (after - before) / max(before, 1e-7)
                model.restore(snapshot)
                gradient_conflict = (
                    cosine(current_gradient, retained_gradient)
                    < conflict_threshold
                    and relative_damage > damage_threshold
                )
                conflict = (
                    nearby >= batch_size // 2
                    and disagreement >= 0.75
                    and (gradient_conflict or disagreement >= 0.90)
                )
                conflict_streak = conflict_streak + 1 if conflict else 0
                if conflict_streak >= persistence:
                    unresolved_center = np.mean(current_x, axis=0)
                    if phase_name == "collision":
                        detection_delay = collision_batches_seen
                    else:
                        false_boundaries += 1
                    continue

            replay_x: np.ndarray | None = None
            replay_y: np.ndarray | None = None
            replay_indices: np.ndarray | None = None
            if memory.size:
                replay_x, replay_y, replay_indices = memory.sample(
                    batch_size, prioritized=True
                )
                replay_examples += len(replay_x)
                train_x = np.concatenate((current_x, replay_x))
                train_y = np.concatenate((current_y, replay_y))
            else:
                train_x, train_y = current_x, current_y
            for _ in range(updates_per_batch):
                model.train(train_x, train_y, learning_rate)
            if (
                replay_indices is not None
                and replay_x is not None
                and replay_y is not None
            ):
                memory.update_priorities(
                    replay_indices, model.losses(replay_x, replay_y)
                )
            if unresolved_center is not None and phase_name == "collision":
                harmful_after += 1
            memory.add(current_x, current_y, model.losses(current_x, current_y))

    if method == "oracle":
        normal_scores = [
            accuracy(oracle_models[task], evaluation[task]) for task in TASKS
        ]
    else:
        normal_scores = [accuracy(model, evaluation[task]) for task in TASKS]
    xor_score = normal_scores[1]
    if method == "oracle":
        collision_score = accuracy(oracle_models["collision"], collision_evaluation)
        coverage = 1.0
        selective = statistics.fmean((*normal_scores, collision_score))
        detected = False
    elif unresolved_center is not None:
        collision_score = float("nan")
        coverage = 0.75
        selective = statistics.fmean(normal_scores)
        detected = True
    else:
        collision_score = accuracy(model, collision_evaluation)
        coverage = 1.0
        selective = statistics.fmean((*normal_scores, collision_score))
        detected = False
    return BoundaryResult(
        seed=seed,
        method=method,
        resolved_accuracy=statistics.fmean(normal_scores),
        xor_accuracy=xor_score,
        collision_accuracy=collision_score,
        coverage=coverage,
        selective_accuracy=selective,
        boundary_detected=detected,
        detection_delay_batches=detection_delay,
        false_boundary_batches=false_boundaries,
        harmful_updates_after_detection=harmful_after,
        replay_examples=replay_examples,
        audit_examples=audit_examples,
    )


def run_benchmark(*, seeds: int = 20) -> tuple[list[BoundaryResult], dict[str, Any]]:
    if seeds < 1:
        raise ValueError("seeds must be positive")
    results = [
        run_method(seed, method)
        for seed in range(seeds)
        for method in ("always_replay", "boundary", "oracle")
    ]
    summary: dict[str, Any] = {}
    for method in ("always_replay", "boundary", "oracle"):
        selected = [result for result in results if result.method == method]
        delays = [
            result.detection_delay_batches
            for result in selected
            if result.detection_delay_batches is not None
        ]
        summary[method] = {
            "resolved_accuracy": statistics.fmean(
                r.resolved_accuracy for r in selected
            ),
            "xor_accuracy": statistics.fmean(r.xor_accuracy for r in selected),
            "collision_accuracy": statistics.fmean(
                r.collision_accuracy
                for r in selected
                if not np.isnan(r.collision_accuracy)
            )
            if any(not np.isnan(r.collision_accuracy) for r in selected)
            else None,
            "coverage": statistics.fmean(r.coverage for r in selected),
            "selective_accuracy": statistics.fmean(
                r.selective_accuracy for r in selected
            ),
            "detection_rate": statistics.fmean(
                float(r.boundary_detected) for r in selected
            ),
            "mean_detection_delay_batches": (
                statistics.fmean(delays) if delays else None
            ),
            "false_boundary_batches": statistics.fmean(
                r.false_boundary_batches for r in selected
            ),
            "harmful_updates_after_detection": statistics.fmean(
                r.harmful_updates_after_detection for r in selected
            ),
            "replay_examples": statistics.fmean(r.replay_examples for r in selected),
            "audit_examples": statistics.fmean(r.audit_examples for r in selected),
        }
    return results, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    results, summary = run_benchmark(seeds=args.seeds)
    report: dict[str, Any] = {
        "configuration": {
            "seeds": args.seeds,
            "schedule": SCHEDULE,
            "task_id_at_inference": False,
            "oracle_receives_hidden_context": True,
        },
        "summary": summary,
    }
    if args.raw:
        report["results"] = [asdict(result) for result in results]
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
