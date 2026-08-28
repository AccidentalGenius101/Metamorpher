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
    PrimitiveCall,
    PrimitiveComposer,
    PrimitiveDiscourseInterpreter,
    PrimitiveError,
    PrimitiveQualification,
    PrimitiveRegistry,
    TypedActionGraph,
    language_primitives,
)


TEXT = "It might be the gasket, but I never inspected the bolts."


class Router:
    def route(self, text, domain, available):
        self.available = tuple(available)
        return (
            PrimitiveCall(
                "extract_observation",
                {
                    "key": "fasteners_inspected",
                    "value": False,
                    "quote": "never inspected the bolts",
                },
            ),
            PrimitiveCall(
                "register_suspicion",
                {"id": "gasket_failed", "quote": "might be the gasket"},
            ),
            PrimitiveCall(
                "propose_action",
                {
                    "id": "inspect_fasteners",
                    "kind": "observe",
                    "label": "Inspect manifold fasteners",
                    "information_value": 10,
                },
            ),
            PrimitiveCall(
                "propose_action",
                {
                    "id": "replace_gasket",
                    "kind": "repair",
                    "label": "Replace manifold gasket",
                    "cost": 10,
                },
            ),
            PrimitiveCall(
                "add_prerequisite",
                {"source": "inspect_fasteners", "target": "replace_gasket"},
            ),
        )


class NoExecution:
    def execute(self, decision):
        raise AssertionError("not executed")


class PrimitiveTests(unittest.TestCase):
    def test_contributor_plan_composes_safe_structure(self) -> None:
        domain = DomainTag("engine")
        router = Router()
        controller = MetamorpherController(TypedActionGraph(), default_domain=domain)
        loop = CognitiveLoop(
            controller,
            executor=NoExecution(),
            perceiver=DiscoursePerceiver(
                PrimitiveDiscourseInterpreter(router, language_primitives())
            ),
            proposer=DiscourseProposer(),
        )
        ingestion = loop.ingest(TEXT, domain)
        decision = controller.next()
        self.assertEqual(decision.status, DecisionStatus.SUPPORTED_UNDER_MODEL)
        self.assertEqual(decision.action_id, "inspect_fasteners")
        self.assertEqual(ingestion.proposal.hypotheses[0].id, "gasket_failed")
        self.assertIsNone(controller.version_space.active)
        self.assertIn("register_suspicion", router.available)

    def test_unknown_contributor_fails_before_controller_mutation(self) -> None:
        registry = language_primitives()
        with self.assertRaisesRegex(PrimitiveError, "unknown or unqualified"):
            PrimitiveComposer(registry).compose(
                TEXT,
                DomainTag("engine"),
                (PrimitiveCall("invent_truth", {"value": True}),),
            )

    def test_unqualified_primitive_cannot_be_installed(self) -> None:
        registry = PrimitiveRegistry()
        with self.assertRaisesRegex(PrimitiveError, "pass every"):
            registry.install(
                "broken",
                lambda state, args: None,
                PrimitiveQualification(2, 3, ("q1", "q2", "q3")),
            )
        self.assertEqual(registry.names, ())

    def test_qualified_primitives_are_frozen_and_growth_is_local(self) -> None:
        registry = language_primitives()
        original = registry.record("extract_observation")
        with self.assertRaisesRegex(PrimitiveError, "already exists"):
            registry.install(
                "extract_observation",
                lambda state, args: None,
                PrimitiveQualification(1, 1, ("replacement",)),
            )
        added = registry.install(
            "resolve_reference",
            lambda state, args: None,
            PrimitiveQualification(3, 3, ("r1", "r2", "r3")),
        )
        self.assertEqual(registry.record("extract_observation"), original)
        self.assertGreater(added.revision, original.revision)

    def test_bad_reference_discards_composition(self) -> None:
        calls = (
            PrimitiveCall(
                "propose_action",
                {"id": "inspect", "kind": "observe", "label": "Inspect"},
            ),
            PrimitiveCall(
                "add_prerequisite",
                {"source": "inspect", "target": "invented"},
            ),
        )
        with self.assertRaisesRegex(PrimitiveError, "unknown action"):
            PrimitiveComposer(language_primitives()).compose(
                TEXT, DomainTag("engine"), calls
            )


if __name__ == "__main__":
    unittest.main()
