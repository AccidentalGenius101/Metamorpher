from __future__ import annotations

import unittest

import _support  # noqa: F401

from metamorpher.model import ActionStatus, DecisionStatus, ObservationStatus
from metamorpher.simulations.sierra import (
    DOMAIN,
    REQUESTED_EVIDENCE,
    SEAL_STATE_EQUIVALENCE_CLASS,
    SierraVisualObservation,
    build_sierra_controller,
    run_sierra_demo,
)


class SierraAdapterTests(unittest.TestCase):
    def test_visual_adapter_preserves_observed_vs_inferred_boundary(self) -> None:
        observation = SierraVisualObservation().to_runtime_observation("issued-token")
        self.assertEqual(observation.status, ObservationStatus.OBSERVED)
        self.assertEqual(observation.key, "fastener_inspection")
        self.assertEqual(observation.action_token, "issued-token")
        self.assertIsInstance(observation.value, dict)
        self.assertNotIn("snapped_bolt", observation.value)
        self.assertNotIn("gasket_condition", observation.value)
        self.assertNotIn("manifold_condition", observation.value)

    def test_raw_simulator_does_not_expose_repair_oracle(self) -> None:
        keys = set(SierraVisualObservation().as_mapping())
        self.assertNotIn("correct_repair", keys)
        self.assertNotIn("gasket_condition", keys)
        self.assertNotIn("manifold_condition", keys)
        self.assertNotIn("oracle_frontier", keys)

    def test_graph_does_not_embed_exact_default_simulator_world(self) -> None:
        controller = build_sierra_controller()
        controller.state.action_status["a.inspect_fasteners"] = ActionStatus.COMPLETED
        # This is a different raw bundle with the same consequential fact: an
        # end fastener has no visible head, hence clamping loss needs assessment.
        variant = SierraVisualObservation(
            driver_front_fastener="present",
            driver_rear_fastener="head_absent",
            passenger_front_fastener="present",
            passenger_rear_fastener="present",
            driver_rear_soot=False,
        )
        for observation in variant.to_runtime_evidence("variant-token"):
            controller.evidence.append(observation)
        frontier = controller.graph.frontier(controller.state, controller.evidence)
        self.assertIn(
            "a.assess_fastener_repair",
            frontier.certified,
            "the graph must guard on the inferred clamp-loss fact, not equality "
            "with the simulator's exact default observation dictionary",
        )
        self.assertIn("a.request_offline_measurements", frontier.certified)

    def test_raw_bundle_alone_does_not_smuggle_semantic_inference(self) -> None:
        controller = build_sierra_controller()
        controller.state.action_status["a.inspect_fasteners"] = ActionStatus.COMPLETED
        raw = SierraVisualObservation().to_runtime_observation("raw-only-token")
        controller.evidence.append(raw)
        frontier = controller.graph.frontier(controller.state, controller.evidence)
        self.assertNotIn("a.assess_fastener_repair", frontier.certified)
        self.assertIn("a.assess_fastener_repair", frontier.refinement)


class SierraIntegrationTests(unittest.TestCase):
    def test_builder_uses_sierra_as_default_domain(self) -> None:
        self.assertEqual(build_sierra_controller().default_domain, DOMAIN)

    def test_missing_fastener_path_stops_at_refinement(self) -> None:
        run = run_sierra_demo()
        result = run.result
        self.assertEqual(result.status, DecisionStatus.REFINEMENT_REQUIRED)
        self.assertEqual(result.first_action, "a.inspect_fasteners")
        self.assertEqual(result.rejected_action, "a.replace_gasket")
        self.assertEqual(set(result.unresolved_hypotheses), set(SEAL_STATE_EQUIVALENCE_CLASS))
        self.assertEqual(set(result.requested_evidence), set(REQUESTED_EVIDENCE))
        self.assertEqual(
            set(result.common_safe_actions),
            {
                "a.assess_fastener_repair",
                "a.defer_part_choice",
                "a.listen_tube",
                "a.request_offline_measurements",
            },
        )
        self.assertEqual(
            set(result.abstained_from),
            {"a.replace_gasket", "a.replace_manifold"},
        )
        self.assertEqual(result.irreversible_actions_executed, ())
        self.assertIsNone(run.decision.action_id)
        self.assertEqual(run.decision.probe_id, "a.request_offline_measurements")

    def test_sierra_trace_is_deterministic(self) -> None:
        first = run_sierra_demo()
        second = run_sierra_demo()
        self.assertEqual(first.trace_digest, second.trace_digest)
        self.assertEqual(first.trace, second.trace)

    def test_trace_never_claims_gasket_or_manifold_identified(self) -> None:
        run = run_sierra_demo()
        serialized = repr(run.trace).lower()
        self.assertNotIn("gasket_confirmed", serialized)
        self.assertNotIn("manifold_confirmed", serialized)
        self.assertEqual(run.result.reason_code, "NON_IDENTIFIABLE_WITH_CURRENT_OBSERVATIONS")
        revised = next(
            event for event in run.trace if event.kind == "VERSION_SPACE_REVISED"
        )
        self.assertEqual(
            set(revised.payload["preserved_unknown"]),
            {"gasket_condition", "manifold_crack", "manifold_flatness"},
        )

    def test_trace_does_not_mislabel_version_space_change_as_graph_revision(self) -> None:
        run = run_sierra_demo()
        initial_epoch = next(
            event.payload["graph_epoch"]
            for event in run.trace
            if event.kind == "GRAPH_COMPILED"
        )
        graph_revision_events = [event for event in run.trace if event.kind == "GRAPH_REVISED"]
        if graph_revision_events:
            self.assertGreater(
                run.decision.graph_epoch,
                initial_epoch,
                "GRAPH_REVISED is truthful only if the typed graph epoch changed; "
                "an active equivalence-class change should be labelled "
                "VERSION_SPACE_REVISED or REPRESENTATION_REVISED",
            )
        else:
            self.assertTrue(
                any(
                    event.kind in {"VERSION_SPACE_REVISED", "REPRESENTATION_REVISED"}
                    for event in run.trace
                )
            )


if __name__ == "__main__":
    unittest.main()
