from __future__ import annotations

import inspect
import unittest
from dataclasses import fields, replace

import _support  # noqa: F401

from metamorpher.carving import AdaptiveFailureCarver, AdaptiveLearningLoop, ConstraintRevision
from metamorpher.controller import MetamorpherController
from metamorpher.graph import TypedActionGraph
from metamorpher.model import (
    ActionKind,
    ActionNode,
    ActionStatus,
    Constraint,
    ConstraintKind,
    ClassStatus,
    DecisionStatus,
    DomainTag,
    InvalidGraphError,
    Observation,
    ObservationStatus,
    StaleDecisionError,
    UnsafeExecutionError,
)
from metamorpher.version_space import Hypothesis, UnresolvedCell, VersionSpaceManager

DOMAIN = DomainTag.from_mapping("test", {"regime": "A"})


class EscapingPolicy:
    def __init__(self, action_id: str) -> None:
        self.action_id = action_id

    def select(self, graph, state, candidates):
        return self.action_id


def simple_graph(*action_ids: str) -> TypedActionGraph:
    graph = TypedActionGraph()
    for index, action_id in enumerate(action_ids):
        graph.add_node(ActionNode(action_id, action_id, decision_value=float(len(action_ids) - index)))
    graph.validate()
    return graph


def refinement_graph() -> TypedActionGraph:
    graph = TypedActionGraph()
    graph.add_node(
        ActionNode(
            "probe",
            "Observe temperature",
            ActionKind.OBSERVE,
            information_value=10.0,
            decision_value=10.0,
            probe_for=("safe_condition",),
        )
    )
    graph.add_node(ActionNode("repair", "Repair", ActionKind.REPAIR, decision_value=5.0))
    graph.add_constraint(
        Constraint(
            "safety_guard",
            ConstraintKind.GUARD,
            (),
            "repair",
            fact_key="safe_condition",
            expected_value=True,
            probe_action_id="probe",
        )
    )
    graph.validate()
    return graph


class ControllerThreeWayTests(unittest.TestCase):
    def test_observation_batches_automatically_carve_and_install_version_space(self) -> None:
        learning = AdaptiveLearningLoop(
            AdaptiveFailureCarver("regime", min_branch_support=2),
            outcome_key="result",
            feature_keys=("sensor",),
            safe_actions_by_outcome={"left": {"left"}, "right": {"right"}},
            domain=DOMAIN,
        )
        controller = MetamorpherController(
            simple_graph("left", "right"),
            adaptive_learning=learning,
            default_domain=DOMAIN,
        )
        controller.observe_many(
            (
                Observation("first-result", "result", "left", independent_audit=True, domain=DOMAIN),
                Observation("first-sensor", "sensor", "A", independent_audit=True, domain=DOMAIN),
            )
        )
        self.assertEqual(
            controller.version_space.active.common_safe_actions(DOMAIN),
            frozenset(),
            "an unobserved declared outcome must still constrain provisional safety",
        )
        for index, (outcome, sensor) in enumerate(
            (("right", "B"), ("left", "A"), ("right", "B"))
        ):
            controller.observe_many(
                (
                    Observation(f"result-{index}", "result", outcome, independent_audit=True, domain=DOMAIN),
                    Observation(f"sensor-{index}", "sensor", sensor, independent_audit=True, domain=DOMAIN),
                )
            )

        self.assertEqual(controller.version_space.active.status, ClassStatus.CARVED)
        self.assertEqual(controller.next().status, DecisionStatus.ABSTAIN)
        controller.observe(
            Observation("current-sensor", "sensor", "A", independent_audit=True, domain=DOMAIN)
        )
        self.assertEqual(controller.next().action_id, "left")

    def test_incomplete_learning_case_is_rejected_atomically(self) -> None:
        controller = MetamorpherController(
            simple_graph("inspect"),
            adaptive_learning=AdaptiveLearningLoop(
                AdaptiveFailureCarver("case", min_branch_support=2),
                outcome_key="result",
                feature_keys=("sensor",),
                safe_actions_by_outcome={"ok": {"inspect"}},
            ),
        )
        with self.assertRaisesRegex(ValueError, "missing features"):
            controller.observe(Observation("result", "result", "ok", independent_audit=True))
        self.assertEqual(controller.evidence.revision, 0)
        self.assertIsNone(controller.version_space.active)

    def test_supported_action_is_selected_from_frontier(self) -> None:
        controller = MetamorpherController(simple_graph("inspect", "repair"), default_domain=DOMAIN)
        decision = controller.next()
        self.assertEqual(decision.status, DecisionStatus.SUPPORTED_UNDER_MODEL)
        self.assertEqual(decision.action_id, "inspect")
        self.assertIsNone(decision.probe_id)
        self.assertIn(decision.action_id, decision.frontier)
        self.assertEqual(decision.domain, DOMAIN)

    def test_unknown_guard_issues_targeted_refinement(self) -> None:
        controller = MetamorpherController(refinement_graph(), default_domain=DOMAIN)
        decision = controller.next()
        self.assertEqual(decision.status, DecisionStatus.REFINEMENT_REQUIRED)
        self.assertEqual(decision.probe_id, "probe")
        self.assertIsNone(decision.action_id)
        self.assertIn("safety_guard", decision.unresolved_assumptions)

    def test_refinement_observation_opens_supported_branch(self) -> None:
        controller = MetamorpherController(refinement_graph(), default_domain=DOMAIN)
        refinement = controller.next()
        selected = controller.commit(refinement)
        self.assertEqual(selected.id, "probe")
        receipt = controller.observe(
            Observation(
                "safe",
                "safe_condition",
                True,
                source="sensor",
                action_token=refinement.token,
            ),
            token=refinement.token,
        )
        self.assertEqual(receipt.action_id, "probe")
        self.assertEqual(receipt.action_status, ActionStatus.COMPLETED)
        supported = controller.next()
        self.assertEqual(supported.status, DecisionStatus.SUPPORTED_UNDER_MODEL)
        self.assertEqual(supported.action_id, "repair")

    def test_incompatible_version_space_with_empty_safe_intersection_abstains(self) -> None:
        manager = VersionSpaceManager()
        manager.add(
            UnresolvedCell(
                "ambiguous",
                {
                    "h1": Hypothesis("h1", frozenset({"left"})),
                    "h2": Hypothesis("h2", frozenset({"right"})),
                },
            ),
            activate=True,
        )
        controller = MetamorpherController(
            simple_graph("left", "right"),
            version_space=manager,
            default_domain=DOMAIN,
        )
        decision = controller.next()
        self.assertEqual(decision.status, DecisionStatus.ABSTAIN)
        self.assertIsNone(decision.action_id)
        self.assertIsNone(decision.probe_id)
        self.assertEqual(decision.common_safe_actions, ())
        self.assertEqual(set(decision.represented_hypotheses), {"h1", "h2"})
        with self.assertRaises(UnsafeExecutionError):
            controller.commit(decision)

    def test_unresolved_hypotheses_can_execute_common_safe_action(self) -> None:
        manager = VersionSpaceManager()
        manager.add(
            UnresolvedCell(
                "ambiguous",
                {
                    "h1": Hypothesis("h1", frozenset({"inspect", "left"})),
                    "h2": Hypothesis("h2", frozenset({"inspect", "right"})),
                },
            ),
            activate=True,
        )
        graph = simple_graph("inspect", "left", "right")
        graph.nodes["inspect"].decision_value = 20.0
        controller = MetamorpherController(graph, version_space=manager, default_domain=DOMAIN)
        decision = controller.next()
        self.assertEqual(decision.status, DecisionStatus.SUPPORTED_UNDER_MODEL)
        self.assertEqual(decision.action_id, "inspect")
        self.assertEqual(decision.common_safe_actions, ("inspect",))

    def test_policy_cannot_escape_safe_frontier(self) -> None:
        graph = simple_graph("inspect", "repair")
        graph.add_constraint(
            Constraint(
                "inspect_first",
                ConstraintKind.HARD_PREREQUISITE,
                ("inspect",),
                "repair",
            )
        )
        controller = MetamorpherController(graph, policy=EscapingPolicy("repair"))
        decision = controller.next()
        self.assertEqual(decision.status, DecisionStatus.ABSTAIN)
        self.assertIsNone(decision.action_id)
        self.assertIn("repair", decision.blockers)


class ControllerFreshnessAndObservationTests(unittest.TestCase):
    def test_graph_change_rejects_issued_decision(self) -> None:
        controller = MetamorpherController(simple_graph("inspect"), default_domain=DOMAIN)
        decision = controller.next()
        transaction = controller.graph.transaction()
        with transaction:
            transaction.add_node(ActionNode("new", "new"))
            transaction.commit()
        with self.assertRaises(StaleDecisionError):
            controller.commit(decision)

    def test_independent_audit_invalidates_pending_decision(self) -> None:
        controller = MetamorpherController(simple_graph("inspect"), default_domain=DOMAIN)
        decision = controller.next()
        controller.observe(
            Observation(
                "audit",
                "unexpected_blocker",
                True,
                source="independent_auditor",
                independent_audit=True,
            )
        )
        self.assertIsNone(controller.pending_decision)
        with self.assertRaises(StaleDecisionError):
            controller.commit(decision)

    def test_untokened_endogenous_observation_is_rejected(self) -> None:
        controller = MetamorpherController(simple_graph("inspect"), default_domain=DOMAIN)
        before = controller.evidence.revision
        with self.assertRaises(StaleDecisionError):
            controller.observe(Observation("x", "result", True, source="tool"))
        self.assertEqual(controller.evidence.revision, before)

    def test_wrong_observation_token_is_rejected_atomically(self) -> None:
        controller = MetamorpherController(simple_graph("inspect"), default_domain=DOMAIN)
        decision = controller.next()
        controller.commit(decision)
        before = controller.evidence.revision
        with self.assertRaises(StaleDecisionError):
            controller.observe(
                Observation("x", "result", True, source="tool", action_token="wrong"),
                token="wrong",
            )
        self.assertEqual(controller.evidence.revision, before)
        self.assertEqual(controller.state.status_of("inspect"), ActionStatus.PENDING)

    def test_invalid_revision_does_not_append_prospective_observation(self) -> None:
        graph = simple_graph("inspect")
        controller = MetamorpherController(graph, default_domain=DOMAIN)
        decision = controller.next()
        controller.commit(decision)
        invalid_revision = ConstraintRevision(
            Constraint(
                "unknown_source",
                ConstraintKind.HARD_PREREQUISITE,
                ("does_not_exist",),
                "inspect",
            ),
            ("outcome",),
            "Adversarial invalid edge.",
        )
        with self.assertRaises(InvalidGraphError):
            controller.observe(
                Observation(
                    "outcome",
                    "result",
                    True,
                    source="tool",
                    action_token=decision.token,
                ),
                token=decision.token,
                revisions=(invalid_revision,),
            )
        self.assertEqual(controller.evidence.revision, 0)
        self.assertEqual(controller.state.status_of("inspect"), ActionStatus.PENDING)

    def test_censored_observation_does_not_narrow_version_space(self) -> None:
        manager = VersionSpaceManager()
        manager.add(
            UnresolvedCell(
                "cell",
                {
                    "h1": Hypothesis("h1", frozenset({"probe"}), {"separator": "one"}),
                    "h2": Hypothesis("h2", frozenset({"probe"}), {"separator": "two"}),
                },
            ),
            activate=True,
        )
        controller = MetamorpherController(simple_graph("probe"), version_space=manager)
        decision = controller.next()
        controller.commit(decision)
        controller.observe(
            Observation(
                "censored",
                "separator",
                "one",
                status=ObservationStatus.CENSORED,
                source="sensor_unavailable",
                censoring_reason="not_observed",
                action_token=decision.token,
            ),
            token=decision.token,
        )
        self.assertEqual(manager.active.surviving_ids, ("h1", "h2"))

    def test_completed_irreversible_action_is_recorded(self) -> None:
        graph = simple_graph("irreversible")
        graph.nodes["irreversible"].irreversible = True
        graph.nodes["irreversible"].reversible = False
        controller = MetamorpherController(graph)
        decision = controller.next()
        controller.commit(decision)
        controller.observe(
            Observation(
                "done",
                "result",
                "ok",
                source="executor",
                action_token=decision.token,
            ),
            token=decision.token,
        )
        self.assertEqual(controller.state.irreversible_effects, ["irreversible"])

    def test_observe_many_records_one_action_completion_atomically(self) -> None:
        controller = MetamorpherController(simple_graph("probe"))
        decision = controller.next()
        controller.commit(decision)
        observations = (
            Observation(
                "raw",
                "raw_bundle",
                {"visible": True},
                status=ObservationStatus.OBSERVED,
                source="sensor",
                action_token=decision.token,
            ),
            Observation(
                "derived",
                "semantic_fact",
                True,
                status=ObservationStatus.INFERRED,
                source="adapter",
                action_token=decision.token,
            ),
        )
        receipt = controller.observe_many(observations, token=decision.token)
        self.assertEqual(receipt.observation_ids, ("raw", "derived"))
        self.assertEqual(controller.evidence.revision, 2)
        self.assertEqual(controller.state.status_of("probe"), ActionStatus.COMPLETED)

    def test_observe_many_duplicate_batch_fails_before_any_mutation(self) -> None:
        controller = MetamorpherController(simple_graph("probe"))
        decision = controller.next()
        controller.commit(decision)
        observations = (
            Observation("same", "one", True, source="sensor", action_token=decision.token),
            Observation("same", "two", True, source="sensor", action_token=decision.token),
        )
        with self.assertRaises(ValueError):
            controller.observe_many(observations, token=decision.token)
        self.assertEqual(controller.evidence.revision, 0)
        self.assertEqual(controller.state.status_of("probe"), ActionStatus.PENDING)
        self.assertEqual(controller.committed_decision, decision)


class OracleFirewallTests(unittest.TestCase):
    def test_observation_schema_contains_no_simulator_truth_fields(self) -> None:
        field_names = {item.name for item in fields(Observation)}
        forbidden = {
            "true_state",
            "oracle_action",
            "oracle_frontier",
            "is_action_valid",
            "hidden_subtype",
            "ground_truth",
        }
        self.assertTrue(field_names.isdisjoint(forbidden))

    def test_controller_observe_accepts_no_oracle_parameters(self) -> None:
        parameters = set(inspect.signature(MetamorpherController.observe).parameters)
        forbidden = {
            "true_state",
            "oracle_action",
            "oracle_frontier",
            "valid_action",
            "hidden_state",
        }
        self.assertTrue(parameters.isdisjoint(forbidden))

    def test_unrepresented_evidence_does_not_magically_choose_hypothesis(self) -> None:
        manager = VersionSpaceManager()
        manager.add(
            UnresolvedCell(
                "cell",
                {
                    "h1": Hypothesis("h1", frozenset({"audit"})),
                    "h2": Hypothesis("h2", frozenset({"audit"})),
                },
            ),
            activate=True,
        )
        controller = MetamorpherController(simple_graph("audit"), version_space=manager)
        controller.observe(
            Observation(
                "outside",
                "unrepresented_separator",
                "h1",
                source="independent_auditor",
                independent_audit=True,
            )
        )
        self.assertEqual(manager.active.surviving_ids, ("h1", "h2"))


class ControllerRollbackTests(unittest.TestCase):
    def test_rollback_restores_derived_state_but_retains_evidence(self) -> None:
        graph = simple_graph("inspect", "repair")
        controller = MetamorpherController(graph, default_domain=DOMAIN)
        checkpoint = controller.checkpoint("before-revision")

        controller.observe(
            Observation(
                "audit",
                "new_fact",
                True,
                source="independent_auditor",
                independent_audit=True,
            )
        )
        transaction = controller.graph.transaction()
        with transaction:
            transaction.add_constraint(
                Constraint(
                    "inspect_first",
                    ConstraintKind.HARD_PREREQUISITE,
                    ("inspect",),
                    "repair",
                )
            )
            transaction.commit()
        controller.state.action_status["inspect"] = ActionStatus.COMPLETED
        controller.memory.record("claim", DOMAIN, True, "memory-event")
        mutated_epoch = controller.graph.epoch

        controller.rollback(checkpoint)
        self.assertNotIn("inspect_first", controller.graph.constraints)
        self.assertEqual(controller.state.status_of("inspect"), ActionStatus.PENDING)
        self.assertIsNone(controller.memory.get("claim", DOMAIN))
        self.assertEqual(controller.evidence.revision, 1)
        self.assertEqual(controller.evidence.resolve("new_fact").value, True)
        self.assertGreater(controller.graph.epoch, mutated_epoch)

    def test_rollback_invalidates_issued_decision(self) -> None:
        controller = MetamorpherController(simple_graph("inspect"))
        checkpoint = controller.checkpoint()
        decision = controller.next()
        controller.rollback(checkpoint)
        self.assertIsNone(controller.pending_decision)
        with self.assertRaises(StaleDecisionError):
            controller.commit(decision)

    def test_tampered_checkpoint_payload_is_rejected(self) -> None:
        controller = MetamorpherController(simple_graph("inspect"))
        checkpoint = controller.checkpoint("trusted")
        forged = replace(checkpoint, label="forged")
        with self.assertRaises(StaleDecisionError):
            controller.rollback(forged)

    def test_rollback_refuses_to_erase_completed_irreversible_effect(self) -> None:
        graph = simple_graph("cut")
        graph.nodes["cut"].irreversible = True
        graph.nodes["cut"].reversible = False
        controller = MetamorpherController(graph)
        checkpoint = controller.checkpoint()
        decision = controller.next()
        controller.commit(decision)
        controller.observe(
            Observation(
                "done",
                "result",
                "ok",
                source="executor",
                action_token=decision.token,
            ),
            token=decision.token,
        )
        with self.assertRaises(UnsafeExecutionError):
            controller.rollback(checkpoint)
        self.assertEqual(controller.state.irreversible_effects, ["cut"])

    def test_rollback_refuses_irreversible_action_in_flight(self) -> None:
        graph = simple_graph("cut")
        graph.nodes["cut"].irreversible = True
        graph.nodes["cut"].reversible = False
        controller = MetamorpherController(graph)
        checkpoint = controller.checkpoint()
        decision = controller.next()
        controller.commit(decision)
        with self.assertRaises(UnsafeExecutionError):
            controller.rollback(checkpoint)
        self.assertEqual(controller.committed_decision, decision)


if __name__ == "__main__":
    unittest.main()
