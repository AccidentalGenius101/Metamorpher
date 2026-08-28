from __future__ import annotations

import unittest

import _support  # noqa: F401

from metamorpher import (
    ActionNode,
    ActionKind,
    CandidateStructure,
    ClaimTier,
    CognitiveLoop,
    Constraint,
    ConstraintKind,
    DecisionStatus,
    DiscourseInterpretation,
    DiscoursePerceiver,
    DiscourseProposer,
    DomainTag,
    EvidenceLedger,
    GroundedTextRenderer,
    InterpreterPerceiver,
    MetamorpherController,
    TextEvidence,
    TypedActionGraph,
    Hypothesis,
)


class Interpreter:
    def interpret(self, text, domain, context):
        start = text.index("ticks")
        return (TextEvidence("engine_tick", True, start, start + len("ticks")),)


class TextBoundaryTests(unittest.TestCase):
    def test_interpreter_preserves_exact_text_provenance(self) -> None:
        domain = DomainTag("engine")
        result = InterpreterPerceiver(Interpreter()).perceive(
            "The truck ticks when cold.", domain
        )
        observation = result.observations[0]
        self.assertEqual(observation.key, "engine_tick")
        self.assertEqual(observation.domain, domain)
        self.assertIn("#10:15", observation.source)
        self.assertEqual(result.representation["spans"][0]["quote"], "ticks")

    def test_invalid_interpreter_span_fails_closed(self) -> None:
        class Invalid:
            def interpret(self, text, domain, context):
                return (TextEvidence("claim", True, 0, len(text) + 1),)

        with self.assertRaises(ValueError):
            InterpreterPerceiver(Invalid()).perceive("text", DomainTag("test"))

    def test_renderer_keeps_model_relative_language(self) -> None:
        graph = TypedActionGraph()
        graph.add_node(ActionNode("inspect", "Inspect the fastener"))
        controller = MetamorpherController(graph)
        decision = controller.next()
        rendered = GroundedTextRenderer().render(decision, graph, EvidenceLedger())
        self.assertEqual(decision.status, DecisionStatus.SUPPORTED_UNDER_MODEL)
        self.assertIn("Under the current model", rendered)
        self.assertNotIn("safe to execute", rendered.lower())

    def test_renderer_states_abstention_without_inventing_action(self) -> None:
        graph = TypedActionGraph()
        controller = MetamorpherController(graph)
        rendered = GroundedTextRenderer().render(
            controller.next(), graph, controller.evidence
        )
        self.assertIn("abstains", rendered)
        self.assertNotIn("None", rendered)

    def test_discourse_keeps_suspicion_quarantined_and_selects_inspection(self) -> None:
        domain = DomainTag("engine")
        text = "It might be the gasket, but I never inspected the bolts."

        class Discourse:
            def interpret_discourse(self, source, selected_domain):
                start = source.index("never inspected the bolts")
                return DiscourseInterpretation(
                    evidence=(
                        TextEvidence(
                            "fasteners_inspected",
                            False,
                            start,
                            start + len("never inspected the bolts"),
                        ),
                    ),
                    candidates=CandidateStructure(
                        nodes=(
                            ActionNode(
                                "inspect_fasteners",
                                "Inspect manifold fasteners",
                                ActionKind.OBSERVE,
                                information_value=10.0,
                            ),
                            ActionNode(
                                "replace_gasket",
                                "Replace manifold gasket",
                                ActionKind.REPAIR,
                                cost=10.0,
                            ),
                        ),
                        constraints=(
                            Constraint(
                                "inspect-before-replace",
                                ConstraintKind.HARD_PREREQUISITE,
                                ("inspect_fasteners",),
                                "replace_gasket",
                                tier=ClaimTier.CANDIDATE,
                            ),
                        ),
                        hypotheses=(
                            Hypothesis(
                                "gasket_failed",
                                frozenset({"inspect_fasteners"}),
                                domain=selected_domain,
                                provenance=("speaker_suspicion",),
                            ),
                        ),
                        rationale="speaker suspects a gasket but reports no inspection",
                    ),
                )

        class NoExecution:
            def execute(self, decision):
                raise AssertionError("not executed")

        controller = MetamorpherController(TypedActionGraph(), default_domain=domain)
        loop = CognitiveLoop(
            controller,
            executor=NoExecution(),
            perceiver=DiscoursePerceiver(Discourse()),
            proposer=DiscourseProposer(),
        )
        ingestion = loop.ingest(text, domain)
        decision = controller.next()
        self.assertEqual(decision.status, DecisionStatus.SUPPORTED_UNDER_MODEL)
        self.assertEqual(decision.action_id, "inspect_fasteners")
        self.assertNotEqual(decision.action_id, "replace_gasket")
        self.assertIsNone(controller.version_space.active)
        self.assertTrue(
            any(cell_id.startswith("candidate-cell-") for cell_id in controller.version_space.cells)
        )
        self.assertEqual(
            ingestion.proposal.hypotheses[0].provenance,
            ("speaker_suspicion",),
        )


if __name__ == "__main__":
    unittest.main()
