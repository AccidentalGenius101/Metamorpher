from __future__ import annotations

import unittest

from metamorpher import (
    ActionNode,
    DecisionStatus,
    DomainTag,
    EvidenceLedger,
    GroundedTextRenderer,
    InterpreterPerceiver,
    MetamorpherController,
    TextEvidence,
    TypedActionGraph,
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


if __name__ == "__main__":
    unittest.main()
