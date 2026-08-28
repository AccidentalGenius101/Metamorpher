from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


def primitive(value: Any) -> Any:
    if is_dataclass(value):
        return {k: primitive(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): primitive(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [primitive(x) for x in value]
    if hasattr(value, "value"):
        return value.value
    return value


@dataclass(frozen=True, slots=True)
class TraceEvent:
    index: int
    kind: str
    payload: dict[str, Any]


class EventTrace:
    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def append(self, kind: str, **payload: Any) -> TraceEvent:
        event = TraceEvent(len(self._events) + 1, kind, primitive(payload))
        self._events.append(event)
        return event

    def canonical_json(self) -> str:
        return json.dumps(primitive(self._events), sort_keys=True, separators=(",", ":"))

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def write_jsonl(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(primitive(event), sort_keys=True) + "\n" for event in self._events)
