"""Deterministic Sierra exhaust-tick demonstration.

This module is deliberately an example adapter, not a mechanical expert system.
The simulator owns the hidden/raw visual result.  The runtime receives only
explicit :class:`~metamorpher.model.Observation` records produced from that
result; it never imports or inspects simulator state.

The example demonstrates a narrow controller property: a semantically salient
repair proposal must not outrank a cheap upstream observation, and observations
that establish lost clamping still do not identify gasket/manifold condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..controller import MetamorpherController
from ..evidence import EvidenceLedger
from ..graph import FrontierResult, TypedActionGraph
from ..model import (
    ActionKind,
    ActionNode,
    ActionStatus,
    ClaimStatus,
    Constraint,
    ConstraintKind,
    ControllerState,
    Decision,
    DecisionStatus,
    DomainTag,
    Observation,
    ObservationStatus,
)
from ..policy import HeuristicLookaheadPolicy
from ..trace import EventTrace, TraceEvent
from ..version_space import Hypothesis, UnresolvedCell, VersionSpaceManager

CASE_ID = "sierra_exhaust_tick_001"
DOMAIN = DomainTag.from_mapping(
    "sierra_exhaust_tick",
    {"case_id": CASE_ID, "scope": "diagnosis_only"},
)
MAX_HARM = 1.0

RELEVANCE_PROPOSAL: tuple[tuple[str, float], ...] = (
    ("a.replace_gasket", 0.92),
    ("a.replace_manifold", 0.81),
    ("a.remove_manifold", 0.74),
    ("a.inspect_fasteners", 0.64),
    ("a.listen_tube", 0.60),
    ("a.inspect_soot", 0.58),
)

INITIAL_EQUIVALENCE_CLASS: tuple[str, ...] = (
    "h.fastener_clamp_loss",
    "h.gasket_only",
    "h.manifold_only",
    "h.other_exhaust_source",
)

SEAL_STATE_EQUIVALENCE_CLASS: tuple[str, ...] = (
    "h.both",
    "h.gasket_only",
    "h.manifold_only",
    "h.neither",
)

REQUESTED_EVIDENCE: tuple[str, ...] = (
    "gasket_condition_after_removal",
    "manifold_crack_inspection",
    "manifold_flatness_vs_service_spec",
    "feasible_fastener_repair_method",
)


@dataclass(frozen=True, slots=True)
class SierraVisualObservation:
    """Raw, deterministic simulator output; no diagnostic inference lives here."""

    driver_front_fastener: str = "head_absent"
    driver_rear_fastener: str = "head_absent"
    passenger_front_fastener: str = "present"
    passenger_rear_fastener: str = "head_absent"
    driver_rear_soot: bool = True
    source: str = "user_visual_inspection"

    def as_mapping(self) -> dict[str, Any]:
        return {
            "driver_front_fastener": self.driver_front_fastener,
            "driver_rear_fastener": self.driver_rear_fastener,
            "passenger_front_fastener": self.passenger_front_fastener,
            "passenger_rear_fastener": self.passenger_rear_fastener,
            "driver_rear_soot": self.driver_rear_soot,
        }

    def clamp_loss_observed(self) -> bool:
        """Return the adapter-level semantic fact supported by the picture.

        This deliberately says only that a represented fastener location is
        not visibly clamping.  It does not infer whether a bolt is snapped,
        missing, stripped, or hidden from view.
        """

        return any(
            value != "present"
            for key, value in self.as_mapping().items()
            if key.endswith("_fastener")
        )

    def to_runtime_observation(self, action_token: str) -> Observation:
        """Translate the raw visual bundle at the runtime boundary.

        One committed runtime action accepts one completion observation.  Its
        value remains the raw structured bundle.  The clamping conclusion is
        recorded separately as an inference in the scenario trace, so a missing
        head is never silently relabelled as a snapped bolt.
        """

        return Observation(
            id="obs.visual.fastener_inspection",
            key="fastener_inspection",
            value=self.as_mapping(),
            status=ObservationStatus.OBSERVED,
            source=self.source,
            reliability=1.0,
            domain=DOMAIN,
            action_token=action_token,
        )

    def to_runtime_evidence(self, action_token: str) -> tuple[Observation, ...]:
        """Return raw provenance plus a separately labelled semantic inference."""

        return (
            self.to_runtime_observation(action_token),
            Observation(
                id="obs.inferred.fastener_clamp_loss",
                key="fastener_clamp_loss",
                value=self.clamp_loss_observed(),
                status=ObservationStatus.INFERRED,
                source="sierra_demo_adapter",
                reliability=1.0,
                domain=DOMAIN,
                action_token=action_token,
            ),
        )

    def to_runtime_observations(self) -> tuple[Observation, ...]:
        """Expose the adapter's raw-versus-inferred boundary for inspection.

        These untokened records describe translation semantics and are not fed
        one-by-one through the committed-action protocol.  The live demo uses
        :meth:`to_runtime_observation` because one inspection is one atomic
        external completion.
        """

        raw = tuple(
            Observation(
                id=f"obs.visual.{key}",
                key=key,
                value=value,
                status=ObservationStatus.OBSERVED,
                source=self.source,
                reliability=1.0,
                domain=DOMAIN,
            )
            for key, value in self.as_mapping().items()
        )
        inferred = Observation(
            id="obs.inferred.fastener_clamp_loss",
            key="fastener_clamp_loss",
            value=self.clamp_loss_observed(),
            status=ObservationStatus.INFERRED,
            source="sierra_demo_adapter",
            reliability=1.0,
            domain=DOMAIN,
        )
        return (*raw, inferred)


class SierraSimulator:
    """Tiny deterministic world kept separate from controller/runtime state."""

    def inspect_fasteners(self) -> SierraVisualObservation:
        return SierraVisualObservation()


@dataclass(frozen=True, slots=True)
class SierraDemoResult:
    case_id: str
    status: DecisionStatus
    first_action: str
    initial_frontier: tuple[str, ...]
    post_revision_frontier: tuple[str, ...]
    rejected_action: str
    rejection_reasons: tuple[str, ...]
    supported_claims: tuple[str, ...]
    unresolved_hypotheses: tuple[str, ...]
    common_safe_actions: tuple[str, ...]
    requested_evidence: tuple[str, ...]
    abstained_from: tuple[str, ...]
    irreversible_actions_executed: tuple[str, ...]
    reason_code: str
    raw_observation: SierraVisualObservation


@dataclass(frozen=True, slots=True)
class SierraDemoRun:
    result: SierraDemoResult
    decision: Decision
    trace: tuple[TraceEvent, ...]
    trace_digest: str


def _node(
    node_id: str,
    label: str,
    kind: ActionKind,
    *,
    cost: float,
    harm: float,
    information_value: float = 0.0,
    decision_value: float = 0.0,
    reversible: bool = True,
    probe_for: tuple[str, ...] = (),
) -> ActionNode:
    return ActionNode(
        id=node_id,
        label=label,
        kind=kind,
        cost=cost,
        harm=harm,
        information_value=information_value,
        decision_value=decision_value,
        reversible=reversible,
        irreversible=not reversible,
        probe_for=probe_for,
        metadata={"simulation": "sierra", "scope": "diagnosis_only"},
    )


def _build_graph() -> TypedActionGraph:
    graph = TypedActionGraph()
    nodes = (
        _node(
            "a.inspect_fasteners",
            "Visually inspect all manifold-end fasteners",
            ActionKind.OBSERVE,
            cost=0.5,
            harm=0.0,
            information_value=9.5,
            decision_value=9.5,
            probe_for=(
                "driver_front_fastener",
                "driver_rear_fastener",
                "passenger_front_fastener",
                "passenger_rear_fastener",
            ),
        ),
        _node(
            "a.inspect_soot",
            "Inspect for a local soot witness",
            ActionKind.OBSERVE,
            cost=0.5,
            harm=0.0,
            information_value=4.0,
            decision_value=4.0,
            probe_for=("driver_rear_soot",),
        ),
        _node(
            "a.listen_tube",
            "Localize the tick with a listening tube",
            ActionKind.TEST,
            cost=0.5,
            harm=0.0,
            information_value=3.5,
            decision_value=3.5,
            probe_for=("localized_tick",),
        ),
        _node(
            "a.remove_manifold",
            "Remove the manifold",
            ActionKind.ACT,
            cost=8.0,
            harm=3.0,
            decision_value=5.0,
            reversible=False,
        ),
        _node(
            "a.inspect_gasket_face",
            "Inspect the removed gasket and witness marks",
            ActionKind.TEST,
            cost=1.0,
            harm=0.0,
            information_value=7.0,
            decision_value=7.0,
            probe_for=("gasket_condition",),
        ),
        _node(
            "a.inspect_manifold_crack",
            "Inspect the removed manifold for cracks",
            ActionKind.TEST,
            cost=1.0,
            harm=0.0,
            information_value=7.0,
            decision_value=7.0,
            probe_for=("manifold_crack",),
        ),
        _node(
            "a.measure_manifold_flatness",
            "Measure flange flatness against the service specification",
            ActionKind.TEST,
            cost=1.0,
            harm=0.0,
            information_value=8.0,
            decision_value=8.0,
            probe_for=("manifold_flatness",),
        ),
        _node(
            "a.assess_fastener_repair",
            "Assess a feasible fastener-repair method",
            ActionKind.ESCALATE,
            cost=1.0,
            harm=0.0,
            information_value=5.0,
            decision_value=6.0,
            probe_for=("fastener_repair_method",),
        ),
        _node(
            "a.restore_clamping",
            "Restore manifold clamping",
            ActionKind.REPAIR,
            cost=6.0,
            harm=2.0,
            decision_value=8.0,
            reversible=False,
        ),
        _node(
            "a.replace_gasket",
            "Replace the exhaust-manifold gasket",
            ActionKind.REPAIR,
            cost=7.0,
            harm=3.0,
            decision_value=9.0,
            reversible=False,
        ),
        _node(
            "a.replace_manifold",
            "Replace the exhaust manifold and gasket",
            ActionKind.REPAIR,
            cost=10.0,
            harm=3.0,
            decision_value=9.0,
            reversible=False,
        ),
        _node(
            "a.defer_part_choice",
            "Do not choose gasket versus manifold without identifying evidence",
            ActionKind.ESCALATE,
            cost=0.0,
            harm=0.0,
            decision_value=0.0,
        ),
        _node(
            "a.request_offline_measurements",
            "Request removal-time gasket, crack, and flatness observations",
            ActionKind.ESCALATE,
            cost=0.0,
            harm=0.0,
            information_value=9.0,
            decision_value=9.0,
            probe_for=(
                "gasket_condition",
                "manifold_crack",
                "manifold_flatness",
                "fastener_repair_method",
            ),
        ),
        _node(
            "a.choose_gasket_or_manifold",
            "Choose gasket-only versus manifold repair",
            ActionKind.ACT,
            cost=0.0,
            harm=0.0,
            decision_value=10.0,
        ),
    )
    for node in nodes:
        graph.add_node(node)

    constraints = (
        Constraint(
            "c.fasteners_before_removal",
            ConstraintKind.SOFT_EPISTEMIC,
            ("a.inspect_fasteners",),
            "a.remove_manifold",
            confidence=1.0,
            provenance=("sierra_demo_spec",),
            domain=DOMAIN,
        ),
        Constraint(
            "c.fasteners_before_gasket",
            ConstraintKind.SOFT_EPISTEMIC,
            ("a.inspect_fasteners",),
            "a.replace_gasket",
            confidence=1.0,
            provenance=("sierra_demo_spec",),
            domain=DOMAIN,
        ),
        Constraint(
            "c.fasteners_before_manifold",
            ConstraintKind.SOFT_EPISTEMIC,
            ("a.inspect_fasteners",),
            "a.replace_manifold",
            confidence=1.0,
            provenance=("sierra_demo_spec",),
            domain=DOMAIN,
        ),
        Constraint(
            "c.remove_before_gasket_inspection",
            ConstraintKind.HARD_PREREQUISITE,
            ("a.remove_manifold",),
            "a.inspect_gasket_face",
            domain=DOMAIN,
        ),
        Constraint(
            "c.remove_before_crack_inspection",
            ConstraintKind.HARD_PREREQUISITE,
            ("a.remove_manifold",),
            "a.inspect_manifold_crack",
            domain=DOMAIN,
        ),
        Constraint(
            "c.remove_before_flatness_measurement",
            ConstraintKind.HARD_PREREQUISITE,
            ("a.remove_manifold",),
            "a.measure_manifold_flatness",
            domain=DOMAIN,
        ),
        Constraint(
            "c.remove_before_gasket_replacement",
            ConstraintKind.HARD_PREREQUISITE,
            ("a.remove_manifold",),
            "a.replace_gasket",
            domain=DOMAIN,
        ),
        Constraint(
            "c.gasket_damage_guard",
            ConstraintKind.GUARD,
            (),
            "a.replace_gasket",
            fact_key="gasket_condition",
            expected_value="damaged",
            probe_action_id="a.inspect_gasket_face",
            domain=DOMAIN,
        ),
        Constraint(
            "c.manifold_serviceable_guard",
            ConstraintKind.GUARD,
            (),
            "a.replace_gasket",
            fact_key="manifold_serviceability",
            expected_value="serviceable",
            probe_action_id="a.measure_manifold_flatness",
            domain=DOMAIN,
        ),
        Constraint(
            "c.remove_before_manifold_replacement",
            ConstraintKind.HARD_PREREQUISITE,
            ("a.remove_manifold",),
            "a.replace_manifold",
            domain=DOMAIN,
        ),
        Constraint(
            "c.manifold_not_serviceable_guard",
            ConstraintKind.GUARD,
            (),
            "a.replace_manifold",
            fact_key="manifold_serviceability",
            expected_value="not_serviceable",
            probe_action_id="a.measure_manifold_flatness",
            domain=DOMAIN,
        ),
        Constraint(
            "c.clamp_loss_before_fastener_assessment",
            ConstraintKind.GUARD,
            (),
            "a.assess_fastener_repair",
            fact_key="fastener_clamp_loss",
            expected_value=True,
            probe_action_id="a.inspect_fasteners",
            domain=DOMAIN,
        ),
        Constraint(
            "c.assess_before_restore",
            ConstraintKind.HARD_PREREQUISITE,
            ("a.assess_fastener_repair",),
            "a.restore_clamping",
            domain=DOMAIN,
        ),
        Constraint(
            "c.clamp_loss_before_measurement_request",
            ConstraintKind.GUARD,
            (),
            "a.request_offline_measurements",
            fact_key="fastener_clamp_loss",
            expected_value=True,
            probe_action_id="a.inspect_fasteners",
            domain=DOMAIN,
        ),
        Constraint(
            "c.seal_state_before_part_choice",
            ConstraintKind.GUARD,
            (),
            "a.choose_gasket_or_manifold",
            fact_key="seal_state",
            expected_value="identified",
            probe_action_id="a.request_offline_measurements",
            domain=DOMAIN,
        ),
    )
    for constraint in constraints:
        graph.add_constraint(constraint)
    for node in nodes:
        if node.harm <= MAX_HARM:
            continue
        graph.add_constraint(
            Constraint(
                f"c.intervention_authorized.{node.id}",
                ConstraintKind.GUARD,
                (),
                node.id,
                fact_key="intervention_authorized",
                expected_value=True,
                provenance=("diagnosis_only_safety_envelope",),
                domain=DOMAIN,
                externally_governed=True,
            )
        )
    graph.validate()
    return graph


def _initial_version_space() -> VersionSpaceManager:
    manager = VersionSpaceManager()
    common = frozenset(
        {
            "a.defer_part_choice",
            "a.inspect_fasteners",
            "a.inspect_soot",
            "a.listen_tube",
        }
    )
    hypotheses = {
        hid: Hypothesis(
            id=hid,
            safe_actions=common,
            domain=DOMAIN,
            provenance=("provisional_joint_leak_generalization",),
        )
        for hid in INITIAL_EQUIVALENCE_CLASS
    }
    manager.add(
        UnresolvedCell(
            id="eq.joint_leak.v0",
            hypotheses=hypotheses,
            parent_id=None,
        ),
        activate=True,
    )
    return manager


def build_sierra_controller() -> MetamorpherController:
    """Build the runtime controller used by the deterministic Sierra example."""

    evidence = EvidenceLedger()
    evidence.append(
        Observation(
            id="obs.policy.intervention_not_authorized",
            key="intervention_authorized",
            value=False,
            status=ObservationStatus.OBSERVED,
            source="diagnosis_only_safety_envelope",
            reliability=1.0,
            domain=DOMAIN,
            independent_audit=True,
        )
    )
    return MetamorpherController(
        graph=_build_graph(),
        state=ControllerState(),
        evidence=evidence,
        version_space=_initial_version_space(),
        policy=HeuristicLookaheadPolicy(),
        trace=EventTrace(),
        default_domain=DOMAIN,
    )


def _resolved_probe(controller: MetamorpherController, node: ActionNode) -> bool:
    if not node.probe_for:
        return False
    bundle = controller.evidence.resolve("fastener_inspection")
    bundled_values = bundle.value if isinstance(bundle.value, dict) else {}
    return all(
        controller.evidence.resolve(key).state.value != "unknown"
        or key in bundled_values
        for key in node.probe_for
    )


def _safe_frontier(controller: MetamorpherController) -> tuple[tuple[str, ...], FrontierResult]:
    """Apply the example's explicit harm budget and dominance policy."""

    raw = controller.graph.frontier(controller.state, controller.evidence)
    suppressed: set[str] = set()
    for constraint in controller.graph.constraints.values():
        if constraint.kind != ConstraintKind.SOFT_EPISTEMIC:
            continue
        if constraint.status == ClaimStatus.REJECTED:
            continue
        if any(controller.state.status_of(source) == ActionStatus.PENDING for source in constraint.sources):
            suppressed.add(constraint.target)
    safe = tuple(
        node_id
        for node_id in raw.certified
        if node_id not in suppressed
        and not _resolved_probe(controller, controller.graph.nodes[node_id])
    )
    return safe, raw


def _install_seal_version_space(controller: MetamorpherController) -> None:
    common = frozenset(
        {
            "a.assess_fastener_repair",
            "a.defer_part_choice",
            "a.listen_tube",
            "a.request_offline_measurements",
        }
    )
    hypotheses = {
        hid: Hypothesis(
            id=hid,
            safe_actions=common,
            domain=DOMAIN,
            provenance=("obs.inferred.fastener_clamp_loss",),
        )
        for hid in SEAL_STATE_EQUIVALENCE_CLASS
    }
    cell = UnresolvedCell(
        id="eq.seal_state.v1",
        hypotheses=hypotheses,
        parent_id="eq.joint_leak.v0",
    )
    controller.version_space.add(cell, activate=True)


def run_sierra_demo() -> SierraDemoRun:
    """Run the complete deterministic diagnostic-control trace.

    The function intentionally stops at a refinement certificate.  It does not
    simulate removal or prescribe a physical repair.
    """

    controller = build_sierra_controller()
    simulator = SierraSimulator()
    # The controller keeps its own low-level protocol trace.  This second trace
    # is the stable, replayable domain transaction shown by the example.
    trace = EventTrace()

    trace.append(
        "CASE_OPENED",
        case_id=CASE_ID,
        state="S0",
        observations={"driver_side_tick": True, "tick_intermittent": True},
        unknown=(
            "fastener_integrity",
            "gasket_condition",
            "manifold_crack",
            "manifold_flatness",
        ),
    )
    trace.append(
        "PROPOSAL_RECORDED",
        state="S0",
        equivalence_class="eq.joint_leak.v0",
        provisional_rule="joint_leak -> replace_gasket",
        relevance_ranking=RELEVANCE_PROPOSAL,
    )
    trace.append(
        "GRAPH_COMPILED",
        state="S0",
        graph_epoch=controller.graph.epoch,
        max_harm=MAX_HARM,
        scope="diagnosis_only",
    )

    initial_frontier, initial_raw = _safe_frontier(controller)
    trace.append(
        "FRONTIER_COMPUTED",
        state="S0",
        certified=initial_frontier,
        refinement=initial_raw.refinement,
        blocked=initial_raw.blocked,
    )

    rejected_action = "a.replace_gasket"
    rejection_reasons = (
        "dominated_by:a.inspect_fasteners",
        "missing:gasket_condition",
        "missing:manifold_serviceability",
        "harm_budget",
    )
    trace.append(
        "ACTION_REJECTED",
        state="S0",
        action_id=rejected_action,
        reasons=rejection_reasons,
        state_unchanged=True,
    )

    first_decision = controller.next()
    first_action = first_decision.action_id or first_decision.probe_id
    if first_action != "a.inspect_fasteners":
        raise AssertionError(f"Sierra demo expected fastener inspection, got {first_action}")
    trace.append(
        "ACTION_SELECTED",
        state="S0",
        action_id=first_action,
        reason="epistemic_dominance + maximum decision value",
    )

    controller.commit(first_decision)
    raw_observation = simulator.inspect_fasteners()
    controller.observe_many(
        raw_observation.to_runtime_evidence(first_decision.token),
        token=first_decision.token,
    )
    trace.append(
        "OBSERVATION_RECORDED",
        state="S1",
        action_id=first_action,
        raw_observation=raw_observation.as_mapping(),
        source=raw_observation.source,
        inference={"fastener_clamp_loss": True},
    )

    _install_seal_version_space(controller)
    trace.append(
        "VERSION_SPACE_REVISED",
        state="S2",
        local_rule_retracted="joint_leak -> replace_gasket",
        supported=("h.fastener_clamp_loss",),
        carved_class="class.loss_of_clamping_observed",
        active_equivalence_class="eq.seal_state.v1",
        unresolved=SEAL_STATE_EQUIVALENCE_CLASS,
        preserved_unknown=(
            "gasket_condition",
            "manifold_crack",
            "manifold_flatness",
        ),
        provenance=(
            "obs.visual.fastener_inspection",
            "obs.inferred.fastener_clamp_loss",
        ),
    )

    post_frontier, post_raw = _safe_frontier(controller)
    exact_post = (
        "a.assess_fastener_repair",
        "a.defer_part_choice",
        "a.listen_tube",
        "a.request_offline_measurements",
    )
    if tuple(sorted(post_frontier)) != exact_post:
        raise AssertionError(
            f"unexpected post-revision frontier: {tuple(sorted(post_frontier))}"
        )
    trace.append(
        "FRONTIER_COMPUTED",
        state="S2",
        certified=exact_post,
        refinement=post_raw.refinement,
        blocked=post_raw.blocked,
    )

    active = controller.version_space.active
    if active is None:
        raise AssertionError("seal-state equivalence class was not activated")
    common_safe_actions = tuple(sorted(active.common_safe_actions()))
    decision = controller.next()
    if decision.status != DecisionStatus.REFINEMENT_REQUIRED:
        raise AssertionError(f"expected a refinement certificate, got {decision.status}")
    if decision.probe_id != "a.request_offline_measurements":
        raise AssertionError(f"unexpected refinement probe: {decision.probe_id}")
    if decision.common_safe_actions != common_safe_actions:
        raise AssertionError("controller certificate lost the active version-space envelope")
    trace.append(
        "CERTIFICATE_ISSUED",
        state="S3",
        decision=decision,
        supported=(
            "driver_end_fastener_clamping_is_lost",
            "any_complete_repair_plan_must_address_fasteners",
        ),
        requested_evidence=REQUESTED_EVIDENCE,
    )
    trace.append(
        "CASE_PAUSED",
        state="S3",
        irreversible_actions_executed=controller.state.irreversible_effects,
    )

    result = SierraDemoResult(
        case_id=CASE_ID,
        status=DecisionStatus.REFINEMENT_REQUIRED,
        first_action=first_action,
        initial_frontier=initial_frontier,
        post_revision_frontier=exact_post,
        rejected_action=rejected_action,
        rejection_reasons=rejection_reasons,
        supported_claims=(
            "driver_end_fastener_clamping_is_lost",
            "any_complete_repair_plan_must_address_fasteners",
        ),
        unresolved_hypotheses=SEAL_STATE_EQUIVALENCE_CLASS,
        common_safe_actions=common_safe_actions,
        requested_evidence=REQUESTED_EVIDENCE,
        abstained_from=("a.replace_gasket", "a.replace_manifold"),
        irreversible_actions_executed=tuple(controller.state.irreversible_effects),
        reason_code="NON_IDENTIFIABLE_WITH_CURRENT_OBSERVATIONS",
        raw_observation=raw_observation,
    )
    return SierraDemoRun(
        result=result,
        decision=decision,
        trace=trace.events,
        trace_digest=trace.digest(),
    )


__all__ = [
    "SierraDemoResult",
    "SierraDemoRun",
    "SierraSimulator",
    "SierraVisualObservation",
    "build_sierra_controller",
    "run_sierra_demo",
]
