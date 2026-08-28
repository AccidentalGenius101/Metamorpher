"""Small, dependency-free command line interface for the reference package."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _doctor_report() -> tuple[dict[str, Any], bool]:
    report: dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "supported": sys.version_info >= (3, 11),
        },
        "platform": platform.platform(),
        "package_version": _distribution_version("metamorpher-control") or "source checkout",
        "core": {"available": False},
        "optional": {},
        "backends": {},
        "backend": {"selected": "reference", "reason": "dependency-free CPU fallback"},
    }

    try:
        from .evidence import EvidenceLedger  # noqa: F401
        from .graph import TypedActionGraph  # noqa: F401
        from .model import DecisionStatus  # noqa: F401

        report["core"] = {"available": True}
    except Exception as exc:  # noqa: BLE001 - doctor must diagnose, not crash
        report["core"] = {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }

    for distribution in ("numpy", "torch", "triton"):
        version = _distribution_version(distribution)
        report["optional"][distribution] = {
            "installed": version is not None,
            "version": version,
        }

    try:
        from .backends import doctor_dict, get_backend

        report["backends"] = doctor_dict()
        selected = get_backend("auto")
        report["backend"] = {
            "selected": selected.name,
            "reason": f"automatic selection on {selected.device}",
        }
    except Exception as exc:  # noqa: BLE001 - optional backend diagnostics
        # Diagnostics must never make the dependency-free core unusable.
        report["backends"] = {"error": f"{type(exc).__name__}: {exc}"}
        report["backend"] = {
            "selected": "reference",
            "reason": "backend inspection failed; use the CPU reference",
        }

    healthy = bool(report["python"]["supported"] and report["core"]["available"])
    return report, healthy


def _print_doctor(report: dict[str, Any]) -> None:
    print("Metamorpher doctor")
    print(
        f"  Python:  {report['python']['version']} "
        f"({report['python']['implementation']})"
    )
    print(f"  Package: {report['package_version']}")
    core = report["core"]
    print(f"  Core:    {'ok' if core['available'] else 'ERROR'}")
    if not core["available"]:
        print(f"           {core.get('error', 'unknown import error')}")
    for name, item in report["optional"].items():
        value = item["version"] if item["installed"] else "not installed (optional)"
        print(f"  {name:<8} {value}")
    for name, item in report["backends"].items():
        if not isinstance(item, dict) or "available" not in item:
            continue
        status = "available" if item["available"] else "unavailable"
        print(f"  {name + ' backend':<18} {status} ({item['device']})")
    backend = report["backend"]
    print(f"  Backend: {backend['selected']} — {backend['reason']}")
    print()
    print("Decision support is model-relative; this diagnostic is not a safety certification.")


def _cmd_doctor(args: argparse.Namespace) -> int:
    report, healthy = _doctor_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_doctor(report)
    return 0 if healthy else 1


def _fallback_sierra_demo() -> tuple[dict[str, Any], Any]:
    """Run a deterministic missing-fastener example on the stable CPU core."""

    from .evidence import EvidenceLedger
    from .graph import TypedActionGraph
    from .model import (
        ActionKind,
        ActionNode,
        ActionStatus,
        Constraint,
        ConstraintKind,
        ControllerState,
        DecisionStatus,
        DomainTag,
        Observation,
    )
    from .policy import HeuristicLookaheadPolicy
    from .trace import EventTrace

    graph = TypedActionGraph()
    for node in (
        ActionNode(
            "inspect_fasteners",
            "Inspect the manifold fasteners",
            ActionKind.OBSERVE,
            cost=0.2,
            information_value=10.0,
        ),
        ActionNode(
            "listen_locally",
            "Listen through a tube to localize the leak",
            ActionKind.TEST,
            cost=0.4,
            information_value=3.0,
        ),
        ActionNode(
            "restore_clamping",
            "Restore missing manifold clamping hardware",
            ActionKind.REPAIR,
            cost=2.0,
            decision_value=12.0,
            reversible=False,
        ),
        ActionNode(
            "replace_gasket",
            "Replace the exhaust manifold gasket",
            ActionKind.REPAIR,
            cost=8.0,
            harm=2.0,
            decision_value=4.0,
            reversible=False,
        ),
        ActionNode(
            "replace_manifold",
            "Replace the exhaust manifold",
            ActionKind.REPAIR,
            cost=15.0,
            harm=4.0,
            decision_value=3.0,
            reversible=False,
        ),
    ):
        graph.add_node(node)

    for constraint in (
        Constraint(
            "inspect-before-clamping-repair",
            ConstraintKind.HARD_PREREQUISITE,
            ("inspect_fasteners",),
            "restore_clamping",
        ),
        Constraint(
            "missing-fastener-guard",
            ConstraintKind.GUARD,
            (),
            "restore_clamping",
            fact_key="fasteners_intact",
            expected_value=False,
        ),
        Constraint(
            "inspect-before-gasket",
            ConstraintKind.HARD_PREREQUISITE,
            ("inspect_fasteners",),
            "replace_gasket",
        ),
        Constraint(
            "intact-fastener-gasket-guard",
            ConstraintKind.GUARD,
            (),
            "replace_gasket",
            fact_key="fasteners_intact",
            expected_value=True,
        ),
        Constraint(
            "inspect-before-manifold",
            ConstraintKind.HARD_PREREQUISITE,
            ("inspect_fasteners",),
            "replace_manifold",
        ),
        Constraint(
            "intact-fastener-manifold-guard",
            ConstraintKind.GUARD,
            (),
            "replace_manifold",
            fact_key="fasteners_intact",
            expected_value=True,
        ),
        Constraint(
            "inspection-dominates-disassembly",
            ConstraintKind.SOFT_EPISTEMIC,
            ("inspect_fasteners",),
            "replace_manifold",
        ),
    ):
        graph.add_constraint(constraint)
    graph.validate()

    evidence = EvidenceLedger()
    state = ControllerState()
    policy = HeuristicLookaheadPolicy()
    trace = EventTrace()

    initial = graph.frontier(state, evidence)
    first_action = policy.select(graph, state, initial.certified)
    trace.append(
        "decision",
        status=DecisionStatus.SUPPORTED_UNDER_MODEL,
        action_id=first_action,
        frontier=initial,
    )

    state.action_status[first_action] = ActionStatus.COMPLETED
    observation = Observation(
        "obs-fasteners-1",
        "fasteners_intact",
        False,
        source="visual inspection",
        domain=DomainTag.from_mapping("sierra-demo", {"side": "driver"}),
    )
    evidence.append(observation)
    trace.append("observation", observation=observation)

    revised = graph.frontier(state, evidence)
    second_action = policy.select(graph, state, revised.certified)
    trace.append(
        "decision",
        status=DecisionStatus.SUPPORTED_UNDER_MODEL,
        action_id=second_action,
        frontier=revised,
    )

    result = {
        "status": DecisionStatus.SUPPORTED_UNDER_MODEL.value,
        "initial_frontier": list(initial.certified),
        "first_action": first_action,
        "observation": {"fasteners_intact": False},
        "observation_summary": "fasteners_intact = false",
        "revised_frontier": list(revised.certified),
        "second_action": second_action,
        "blocked_after_observation": list(revised.blocked),
        "lesson": (
            "The case-specific fastener observation changes the repair frontier; "
            "gasket and manifold replacement remain blocked under this toy model."
        ),
        "warning": "Supported under this supplied model is not a repair or safety guarantee.",
    }
    return result, trace


def _sierra_demo() -> tuple[dict[str, Any], Any]:
    """Prefer the integrated scenario, retaining a stable partial-checkout demo."""

    try:
        from .simulations.sierra import run_sierra_demo
    except (ImportError, AttributeError):
        return _fallback_sierra_demo()

    run = run_sierra_demo()
    domain_result = run.result
    result = {
        "case_id": domain_result.case_id,
        "status": domain_result.status.value,
        "initial_frontier": list(domain_result.initial_frontier),
        "first_action": domain_result.first_action,
        "observation": domain_result.raw_observation.as_mapping(),
        "observation_summary": (
            "three manifold-end fastener heads are absent and a soot witness is present"
        ),
        "revised_frontier": list(domain_result.post_revision_frontier),
        "second_action": run.decision.probe_id or run.decision.action_id,
        "blocked_after_observation": list(domain_result.abstained_from),
        "unresolved_hypotheses": list(domain_result.unresolved_hypotheses),
        "common_safe_actions": list(domain_result.common_safe_actions),
        "requested_evidence": list(domain_result.requested_evidence),
        "irreversible_actions_executed": list(
            domain_result.irreversible_actions_executed
        ),
        "reason_code": domain_result.reason_code,
        "trace_digest": run.trace_digest,
        "lesson": (
            "The fastener observation retracts the premature part-choice rule, "
            "but current evidence still cannot identify gasket versus manifold condition."
        ),
        "warning": "Supported under this supplied model is not a repair or safety guarantee.",
    }
    return result, run.trace


def _cmd_demo_sierra(args: argparse.Namespace) -> int:
    try:
        result, trace = _sierra_demo()
    except Exception as exc:  # noqa: BLE001 - CLI converts demo failures to status 1
        print(f"Sierra demo could not run: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("Run `metamorpher doctor` to inspect this checkout.", file=sys.stderr)
        return 1

    if args.trace is not None:
        path = Path(args.trace)
        path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(trace, "write_jsonl"):
            trace.write_jsonl(path)
        else:
            from .serialization import dump_trace

            dump_trace(trace, path)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("Metamorpher Sierra demo (simulation only)")
        print(f"  Initial frontier: {', '.join(result['initial_frontier'])}")
        print(f"  Selected first:   {result['first_action']}")
        print(f"  Observation:      {result['observation_summary']}")
        print(f"  Revised frontier: {', '.join(result['revised_frontier'])}")
        print(f"  Next refinement:  {result['second_action']}")
        print(f"  Abstained from:   {', '.join(result['blocked_after_observation'])}")
        if args.trace is not None:
            print(f"  Trace:            {args.trace}")
        print()
        print(result["lesson"])
        print(result["warning"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metamorpher",
        description="Reference tools for model-relative epistemic frontier control.",
    )
    parser.add_argument("--version", action="version", version=_distribution_version("metamorpher-control") or "0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="inspect the core and optional acceleration backends")
    doctor.add_argument("--json", action="store_true", help="emit a machine-readable report")
    doctor.set_defaults(handler=_cmd_doctor)

    demo = commands.add_parser("demo", help="run a deterministic reference scenario")
    demos = demo.add_subparsers(dest="demo", required=True)
    sierra = demos.add_parser("sierra", help="show how a fastener observation revises the frontier")
    sierra.add_argument("--json", action="store_true", help="emit machine-readable results")
    sierra.add_argument("--trace", metavar="PATH", help="write an append-only JSONL trace")
    sierra.set_defaults(handler=_cmd_demo_sierra)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
