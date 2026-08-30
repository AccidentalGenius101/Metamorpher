"""Small NumPy continual-learning benchmark with bounded replay.

Three recurring nonlinear input regions define one composite classification
problem. The learner receives only two raw coordinates, never a task ID. This
is a domain-incremental mechanism test: input distributions identify context,
so it does not claim to solve contradictory labels for indistinguishable
inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

ReplayKind = Literal["none", "random", "prioritized", "gated"]
TASKS = ("ring", "xor", "wave")
DEFAULT_SCHEDULE = ("ring", "xor", "wave", "ring", "xor", "wave")
DEFAULT_DAMAGE_THRESHOLD = 0.01


@dataclass(frozen=True, slots=True)
class TaskData:
    features: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True, slots=True)
class MethodResult:
    seed: int
    method: ReplayKind
    phase_accuracy: tuple[tuple[float, ...], ...]
    final_average_accuracy: float
    final_worst_task_accuracy: float
    mean_forgetting: float
    replay_batches: int
    replay_examples: int
    audit_examples: int
    gradient_updates: int
    rejected_candidate_batches: int


class TinyMLP:
    """One-hidden-layer binary classifier with deterministic SGD."""

    def __init__(self, rng: np.random.Generator, hidden: int = 24) -> None:
        self.w1 = rng.normal(0.0, math.sqrt(2.0 / (2 + hidden)), (2, hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = rng.normal(0.0, math.sqrt(2.0 / (hidden + 1)), (hidden, 1))
        self.b2 = np.zeros(1)

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        hidden = np.tanh(features @ self.w1 + self.b1)
        logits = (hidden @ self.w2 + self.b2).reshape(-1)
        logits = np.clip(logits, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def losses(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        probabilities = np.clip(self.probabilities(features), 1e-7, 1.0 - 1e-7)
        return -(
            labels * np.log(probabilities)
            + (1.0 - labels) * np.log(1.0 - probabilities)
        )

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.w1.copy(), self.b1.copy(), self.w2.copy(), self.b2.copy()

    def restore(
        self,
        snapshot: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        self.w1, self.b1, self.w2, self.b2 = (
            value.copy() for value in snapshot
        )

    def train(self, features: np.ndarray, labels: np.ndarray, rate: float) -> None:
        hidden = np.tanh(features @ self.w1 + self.b1)
        logits = (hidden @ self.w2 + self.b2).reshape(-1)
        logits = np.clip(logits, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        output_gradient = (probabilities - labels)[:, None] / len(features)
        w2_gradient = hidden.T @ output_gradient
        b2_gradient = output_gradient.sum(axis=0)
        hidden_gradient = (output_gradient @ self.w2.T) * (1.0 - hidden**2)
        w1_gradient = features.T @ hidden_gradient
        b1_gradient = hidden_gradient.sum(axis=0)
        self.w2 -= rate * w2_gradient
        self.b2 -= rate * b2_gradient
        self.w1 -= rate * w1_gradient
        self.b1 -= rate * b1_gradient


class ReplayBuffer:
    """Reservoir memory with uniform or current-loss-prioritized sampling."""

    def __init__(self, capacity: int, rng: np.random.Generator) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.capacity = capacity
        self.rng = rng
        self.features = np.empty((capacity, 2))
        self.labels = np.empty(capacity)
        self.priorities = np.ones(capacity)
        self.size = 0
        self.seen = 0

    def add(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        priorities: np.ndarray,
    ) -> None:
        for feature, label, priority in zip(features, labels, priorities):
            self.seen += 1
            if self.size < self.capacity:
                index = self.size
                self.size += 1
            else:
                candidate = int(self.rng.integers(0, self.seen))
                if candidate >= self.capacity:
                    continue
                index = candidate
            self.features[index] = feature
            self.labels[index] = label
            self.priorities[index] = max(float(priority), 1e-6)

    def sample(
        self,
        count: int,
        *,
        prioritized: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.size == 0:
            raise ValueError("cannot sample an empty replay buffer")
        replace = self.size < count
        probabilities = None
        if prioritized:
            scaled = np.sqrt(self.priorities[: self.size])
            probabilities = scaled / scaled.sum()
        indices = self.rng.choice(
            self.size,
            size=count,
            replace=replace,
            p=probabilities,
        )
        return self.features[indices], self.labels[indices], indices

    def update_priorities(self, indices: np.ndarray, losses: np.ndarray) -> None:
        for index, loss in zip(indices, losses):
            self.priorities[int(index)] = max(float(loss), 1e-6)


def make_task(
    rng: np.random.Generator,
    task: str,
    samples: int,
) -> TaskData:
    """Generate one observable region and its nonlinear local decision rule."""

    local = rng.normal(0.0, 0.78, (samples, 2))
    if task == "ring":
        center = np.asarray((-2.6, 0.0))
        labels = (np.sum(local**2, axis=1) > 0.62).astype(float)
    elif task == "xor":
        center = np.asarray((2.6, 0.0))
        labels = (local[:, 0] * local[:, 1] > 0.0).astype(float)
    elif task == "wave":
        center = np.asarray((0.0, 2.8))
        labels = (local[:, 1] > 0.48 * np.sin(2.4 * local[:, 0])).astype(float)
    else:
        raise ValueError(f"unknown task: {task}")
    return TaskData(local + center, labels)


def accuracy(model: TinyMLP, data: TaskData) -> float:
    predicted = model.probabilities(data.features) >= 0.5
    return float(np.mean(predicted == data.labels))


def run_method(
    seed: int,
    method: ReplayKind,
    *,
    schedule: tuple[str, ...] = DEFAULT_SCHEDULE,
    phase_samples: int = 640,
    evaluation_samples: int = 800,
    stream_batch: int = 16,
    replay_batch: int = 16,
    memory_capacity: int = 96,
    hidden: int = 24,
    learning_rate: float = 0.20,
    updates_per_batch: int = 3,
    damage_threshold: float = DEFAULT_DAMAGE_THRESHOLD,
    audit_batch: int = 96,
) -> MethodResult:
    """Train one method with identical streams, initialization, and compute."""

    if method not in {"none", "random", "prioritized", "gated"}:
        raise ValueError(f"unknown replay method: {method}")
    if phase_samples % stream_batch:
        raise ValueError("phase samples must be divisible by stream batch")
    data_rng = np.random.default_rng(seed)
    model_rng = np.random.default_rng(seed + 10_000)
    replay_rng = np.random.default_rng(seed + 20_000)
    model = TinyMLP(model_rng, hidden=hidden)
    memory = ReplayBuffer(memory_capacity, replay_rng)
    evaluation = {
        task: make_task(data_rng, task, evaluation_samples) for task in TASKS
    }
    phases = [make_task(data_rng, task, phase_samples) for task in schedule]
    phase_accuracy: list[tuple[float, ...]] = []
    replay_batches = 0
    replay_examples = 0
    audit_examples = 0
    gradient_updates = 0
    rejected_candidate_batches = 0

    for phase in phases:
        order = data_rng.permutation(phase_samples)
        for start in range(0, phase_samples, stream_batch):
            indices = order[start : start + stream_batch]
            current_x = phase.features[indices]
            current_y = phase.labels[indices]
            replay_indices: np.ndarray | None = None
            candidate_x: np.ndarray | None = None
            candidate_y: np.ndarray | None = None
            should_replay = method in {"random", "prioritized"} and memory.size > 0
            if method == "gated" and memory.size > 0:
                audit_x, audit_y, _ = memory.sample(
                    audit_batch,
                    prioritized=False,
                )
                audit_examples += len(audit_x)
                retained_loss_before = float(np.mean(model.losses(audit_x, audit_y)))
                candidate = model.snapshot()
                repeated_x, repeated_y = repeat_current(
                    current_x,
                    current_y,
                    replay_batch,
                )
                candidate_x = np.concatenate((current_x, repeated_x))
                candidate_y = np.concatenate((current_y, repeated_y))
                model.train(candidate_x, candidate_y, learning_rate)
                gradient_updates += 1
                retained_loss_after = float(np.mean(model.losses(audit_x, audit_y)))
                relative_damage = (
                    retained_loss_after - retained_loss_before
                ) / max(retained_loss_before, 1e-7)
                should_replay = relative_damage > damage_threshold
                if should_replay:
                    model.restore(candidate)
                    rejected_candidate_batches += 1

            if not should_replay:
                repeats = math.ceil(replay_batch / len(current_x))
                extra_x = np.tile(current_x, (repeats, 1))[:replay_batch]
                extra_y = np.tile(current_y, repeats)[:replay_batch]
            else:
                extra_x, extra_y, replay_indices = memory.sample(
                    replay_batch,
                    prioritized=method in {"prioritized", "gated"},
                )
                replay_batches += 1
                replay_examples += replay_batch
            train_x = np.concatenate((current_x, extra_x))
            train_y = np.concatenate((current_y, extra_y))
            if method == "gated" and memory.size > 0 and not should_replay:
                assert candidate_x is not None and candidate_y is not None
                for _ in range(updates_per_batch - 1):
                    model.train(candidate_x, candidate_y, learning_rate)
                    gradient_updates += 1
            else:
                for _ in range(updates_per_batch):
                    model.train(train_x, train_y, learning_rate)
                    gradient_updates += 1
            if replay_indices is not None and method in {"prioritized", "gated"}:
                memory.update_priorities(
                    replay_indices,
                    model.losses(extra_x, extra_y),
                )
            memory.add(current_x, current_y, model.losses(current_x, current_y))
        phase_accuracy.append(
            tuple(accuracy(model, evaluation[task]) for task in TASKS)
        )

    final = phase_accuracy[-1]
    forgetting = []
    for task_index, task in enumerate(TASKS):
        first_learned_phase = schedule.index(task)
        best_after_learning = max(
            values[task_index] for values in phase_accuracy[first_learned_phase:]
        )
        forgetting.append(best_after_learning - final[task_index])
    return MethodResult(
        seed=seed,
        method=method,
        phase_accuracy=tuple(phase_accuracy),
        final_average_accuracy=statistics.fmean(final),
        final_worst_task_accuracy=min(final),
        mean_forgetting=statistics.fmean(forgetting),
        replay_batches=replay_batches,
        replay_examples=replay_examples,
        audit_examples=audit_examples,
        gradient_updates=gradient_updates,
        rejected_candidate_batches=rejected_candidate_batches,
    )


def repeat_current(
    features: np.ndarray,
    labels: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    repeats = math.ceil(count / len(features))
    return (
        np.tile(features, (repeats, 1))[:count],
        np.tile(labels, repeats)[:count],
    )


def run_benchmark(
    *,
    seeds: int = 20,
    memory_capacity: int = 96,
) -> tuple[list[MethodResult], dict[str, Any]]:
    if seeds < 1:
        raise ValueError("seeds must be positive")
    results = [
        run_method(seed, method, memory_capacity=memory_capacity)
        for seed in range(seeds)
        for method in ("none", "random", "prioritized", "gated")
    ]
    summary: dict[str, Any] = {}
    for method in ("none", "random", "prioritized", "gated"):
        selected = [item for item in results if item.method == method]
        final_average = [item.final_average_accuracy for item in selected]
        final_worst = [item.final_worst_task_accuracy for item in selected]
        forgetting = [item.mean_forgetting for item in selected]
        replay_examples = [item.replay_examples for item in selected]
        audit_examples = [item.audit_examples for item in selected]
        gradient_updates = [item.gradient_updates for item in selected]
        rejected = [item.rejected_candidate_batches for item in selected]
        summary[method] = {
            "final_average_accuracy": statistics.fmean(final_average),
            "final_average_accuracy_std": statistics.pstdev(final_average),
            "final_worst_task_accuracy": statistics.fmean(final_worst),
            "final_worst_task_accuracy_std": statistics.pstdev(final_worst),
            "mean_forgetting": statistics.fmean(forgetting),
            "mean_forgetting_std": statistics.pstdev(forgetting),
            "replay_examples": statistics.fmean(replay_examples),
            "audit_examples": statistics.fmean(audit_examples),
            "gradient_updates": statistics.fmean(gradient_updates),
            "rejected_candidate_batches": statistics.fmean(rejected),
        }
    return results, summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--seeds", type=int, default=20)
    value.add_argument("--memory-capacity", type=int, default=96)
    value.add_argument("--raw", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    results, summary = run_benchmark(
        seeds=args.seeds,
        memory_capacity=args.memory_capacity,
    )
    report: dict[str, Any] = {
        "configuration": {
            "seeds": args.seeds,
            "memory_capacity": args.memory_capacity,
            "schedule": DEFAULT_SCHEDULE,
            "task_id_at_inference": False,
            "gated_relative_damage_threshold": DEFAULT_DAMAGE_THRESHOLD,
        },
        "summary": summary,
    }
    if args.raw:
        report["results"] = [asdict(item) for item in results]
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
