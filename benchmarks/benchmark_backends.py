"""Reproducible end-to-end backend benchmark with reference parity checks."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections.abc import Sequence
from typing import Any

from metamorpher.backends import ReferenceBackend, get_backend


def _nested(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def fixture(
    *,
    batch: int,
    actions: int,
    features: int,
    density: float,
    seed: int,
) -> tuple[Any, ...]:
    rng = random.Random(seed)
    pending = [[rng.random() < 0.85 for _ in range(actions)] for _ in range(batch)]
    completed = [[rng.random() < 0.25 for _ in range(actions)] for _ in range(batch)]
    dependencies = [
        [
            [source < target and rng.random() < density for source in range(actions)]
            for target in range(actions)
        ]
        for _ in range(batch)
    ]
    values = [
        [
            [rng.uniform(-2.0, 2.0) for _ in range(features)]
            for _ in range(actions)
        ]
        for _ in range(batch)
    ]
    weights = [rng.uniform(-1.0, 1.0) for _ in range(features)]
    return pending, completed, dependencies, values, weights


def parity(reference: Any, candidate: Any, *, tolerance: float = 1e-4) -> None:
    rf = _nested(reference.frontier)
    cf = _nested(candidate.frontier)
    if rf != cf:
        raise AssertionError("frontier parity failure")
    rs = _nested(reference.selected)
    cs = _nested(candidate.selected)
    if rs != cs:
        raise AssertionError(f"selection parity failure: reference={rs}, candidate={cs}")
    r_scores = _nested(reference.scores)
    c_scores = _nested(candidate.scores)
    for r_row, c_row in zip(r_scores, c_scores):
        for r, c in zip(r_row, c_row):
            if math.isinf(r) and math.isinf(c) and r < 0 and c < 0:
                continue
            if not math.isclose(float(r), float(c), rel_tol=tolerance, abs_tol=tolerance):
                raise AssertionError(f"score parity failure: {r} != {c}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    data = fixture(
        batch=args.batch,
        actions=args.actions,
        features=args.features,
        density=args.density,
        seed=args.seed,
    )
    reference_engine = ReferenceBackend()
    engine = get_backend(args.backend, strict=args.strict)
    reference = reference_engine.frontier_and_score(*data)
    candidate = engine.frontier_and_score(*data)
    parity(reference, candidate)

    for _ in range(args.warmup):
        engine.frontier_and_score(*data)
    samples: list[float] = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        engine.frontier_and_score(*data)
        samples.append((time.perf_counter() - started) * 1000.0)

    return {
        "backend": engine.name,
        "device": engine.device,
        "batch": args.batch,
        "actions": args.actions,
        "features": args.features,
        "edge_density": args.density,
        "seed": args.seed,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "cases_per_second_median": args.batch / (statistics.median(samples) / 1000.0),
        "reference_parity": True,
        "timing_scope": "packing/conversion + frontier + scoring + selection",
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--backend", choices=("auto", "reference", "numpy", "triton"), default="auto")
    value.add_argument("--strict", action="store_true")
    value.add_argument("--batch", type=int, default=512)
    value.add_argument("--actions", type=int, default=128)
    value.add_argument("--features", type=int, default=7)
    value.add_argument("--density", type=float, default=0.03)
    value.add_argument("--seed", type=int, default=20260828)
    value.add_argument("--warmup", type=int, default=3)
    value.add_argument("--repeats", type=int, default=10)
    value.add_argument("--json", action="store_true")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.batch <= 0 or args.actions <= 0 or args.features <= 0:
        raise SystemExit("batch, actions, and features must be positive")
    if not 0.0 <= args.density <= 1.0:
        raise SystemExit("density must be in [0, 1]")
    report = run(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"{report['backend']} on {report['device']}: "
            f"median {report['median_ms']:.3f} ms; "
            f"{report['cases_per_second_median']:.0f} cases/s; parity ok"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
