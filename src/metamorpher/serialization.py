"""Stable, dependency-free serialization helpers for audits and replay."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .model import Observation
from .trace import TraceEvent, primitive


def dump_trace(events: Iterable[TraceEvent], path: str | Path) -> None:
    destination = Path(path)
    with destination.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(primitive(event), sort_keys=True) + "\n")


def load_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return tuple(json.loads(line) for line in handle if line.strip())


def observation_from_mapping(value: dict[str, Any]) -> Observation:
    """Conservative loader; callers must explicitly supply required fields."""
    allowed = {
        "id", "key", "value", "status", "source", "reliability", "domain",
        "action_token", "timestamp", "censoring_reason", "independent_audit",
    }
    unknown = set(value).difference(allowed)
    if unknown:
        raise ValueError(f"unknown observation fields: {sorted(unknown)}")
    return Observation(**value)
