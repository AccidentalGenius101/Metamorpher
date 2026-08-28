from __future__ import annotations

import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def graph_snapshot(graph):
    """Small stable snapshot used to prove failed transactions are atomic."""
    return (
        copy.deepcopy(graph.nodes),
        copy.deepcopy(graph.constraints),
        graph.epoch,
    )
