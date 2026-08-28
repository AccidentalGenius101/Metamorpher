from __future__ import annotations

import os
import subprocess
import sys
import unittest

import _support

from metamorpher import (
    ActionNode,
    CandidateStructure,
    ClaimTier,
    CognitiveLoop,
    Constraint,
    ConstraintKind,
    ConstraintRevision,
    DomainTag,
    ExecutionResult,
    InMemoryCapsuleStore,
    MetamorpherController,
    Observation,
    StructuralCapsule,
    TypedActionGraph,
)
from metamorpher.model import InvalidGraphError


def graph_with(action_id: str) -> TypedActionGraph:
    graph = TypedActionGraph()
    graph.add_node(ActionNode(action_id, action_id))
    graph.validate()
    return graph


class Executor:
    def execute(self, decision):
        return ExecutionResult(
            decision.action_id or decision.probe_id,
            True,
            (Observation("e1", "worked", True, source="tool"),),
        )


class Learner:
    def revise(self, controller, decision, result):
        return (
            ConstraintRevision(
                Constraint(
                    "learned",
                    ConstraintKind.GUARD,
                    (),
                    "act",
                    tier=ClaimTier.SUPPORTED,
                    fact_key="worked",
                ),
                ("e1",),
                "the committed outcome supports this guard",
            ),
        )


class CognitiveLoopTests(unittest.TestCase):
    def test_top_level_import_does_not_eagerly_load_cognition_or_controller(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys, metamorpher; "
                    "assert 'metamorpher.cognition' not in sys.modules; "
                    "assert 'metamorpher.controller' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PYTHONPATH": str(_support.SRC_ROOT),
            },
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_candidate_constraints_are_quarantined(self) -> None:
        loop = CognitiveLoop(MetamorpherController(graph_with("seed")), executor=Executor())
        epoch = loop.install_candidates(
            CandidateStructure(
                nodes=(ActionNode("later", "later"),),
                constraints=(
                    Constraint(
                        "candidate-edge",
                        ConstraintKind.HARD_PREREQUISITE,
                        ("seed",),
                        "later",
                        tier=ClaimTier.CANDIDATE,
                    ),
                ),
                rationale="retrieved possible dependency",
            )
        )
        self.assertEqual(epoch, 1)
        self.assertEqual(
            loop.controller.graph.constraints["candidate-edge"].tier,
            ClaimTier.CANDIDATE,
        )

    def test_supported_proposal_cannot_bypass_evidence_revision(self) -> None:
        loop = CognitiveLoop(MetamorpherController(graph_with("seed")), executor=Executor())
        with self.assertRaises(InvalidGraphError):
            loop.install_candidates(
                CandidateStructure(
                    nodes=(ActionNode("later", "later"),),
                    constraints=(
                        Constraint(
                            "unsafe-promotion",
                            ConstraintKind.HARD_PREREQUISITE,
                            ("seed",),
                            "later",
                            tier=ClaimTier.SUPPORTED,
                        ),
                    ),
                    rationale="model said so",
                )
            )

    def test_step_binds_execution_observation_and_learning_atomically(self) -> None:
        controller = MetamorpherController(graph_with("act"))
        step = CognitiveLoop(controller, executor=Executor(), learner=Learner()).step()
        self.assertTrue(step.result.succeeded)
        self.assertEqual(step.result.observations[0].action_token, step.decision.token)
        self.assertIn("learned", controller.graph.constraints)
        self.assertEqual(controller.graph.constraints["learned"].supporting_evidence_ids, ("e1",))

    def test_capsules_are_domain_bounded_and_copied(self) -> None:
        store = InMemoryCapsuleStore()
        domain = DomainTag("engine")
        store.put(StructuralCapsule("fasteners", domain, evidence_ids=("e1",)))
        self.assertEqual([item.id for item in store.candidates(domain)], ["fasteners"])
        self.assertEqual(store.candidates(DomainTag("kitchen")), ())


if __name__ == "__main__":
    unittest.main()
