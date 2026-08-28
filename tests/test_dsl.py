from __future__ import annotations

import unittest

import _support  # noqa: F401

from metamorpher import (
    CognitiveLoop,
    DecisionStatus,
    DiscoursePerceiver,
    DiscourseProposer,
    DomainTag,
    MetamorpherController,
    TextProgramCompiler,
    TextProgramError,
    TypedActionGraph,
)


SOURCE = "It might be the gasket, but I never inspected the bolts."
PROGRAM = '''
OBSERVE fasteners_inspected false @ "never inspected the bolts"
SUSPECT gasket_failed @ "might be the gasket"
ACTION inspect_fasteners observe "Inspect manifold fasteners"
ACTION replace_gasket repair "Replace manifold gasket"
BEFORE inspect_fasteners replace_gasket
'''


class ProgramInterpreter:
    def interpret_discourse(self, text, domain):
        return TextProgramCompiler().compile(text, PROGRAM, domain)


class NoExecution:
    def execute(self, decision):
        raise AssertionError("not executed")


class TextProgramTests(unittest.TestCase):
    def test_program_compiles_to_safe_next_action(self) -> None:
        domain = DomainTag("engine")
        controller = MetamorpherController(TypedActionGraph(), default_domain=domain)
        loop = CognitiveLoop(
            controller,
            executor=NoExecution(),
            perceiver=DiscoursePerceiver(ProgramInterpreter()),
            proposer=DiscourseProposer(),
        )
        ingestion = loop.ingest(SOURCE, domain)
        decision = controller.next()
        self.assertEqual(decision.status, DecisionStatus.SUPPORTED_UNDER_MODEL)
        self.assertEqual(decision.action_id, "inspect_fasteners")
        self.assertIsNone(controller.version_space.active)
        self.assertEqual(ingestion.proposal.hypotheses[0].id, "gasket_failed")
        self.assertEqual(
            controller.evidence.events[0].value,
            False,
        )

    def test_invented_quote_fails_closed(self) -> None:
        with self.assertRaisesRegex(TextProgramError, "quotation not found"):
            TextProgramCompiler().compile(
                SOURCE,
                'SUSPECT crack @ "visible crack"',
                DomainTag("engine"),
            )

    def test_unknown_action_reference_fails_closed(self) -> None:
        with self.assertRaisesRegex(TextProgramError, "unknown action"):
            TextProgramCompiler().compile(
                SOURCE,
                'ACTION inspect observe "Inspect"\nBEFORE inspect replace',
                DomainTag("engine"),
            )

    def test_json_wrapping_and_unknown_operations_are_rejected(self) -> None:
        for invalid in ('{"observation": true}', "CERTIFY gasket_failed"):
            with self.subTest(invalid=invalid), self.assertRaises(TextProgramError):
                TextProgramCompiler().compile(SOURCE, invalid, DomainTag("engine"))


if __name__ == "__main__":
    unittest.main()
