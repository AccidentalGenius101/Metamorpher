from __future__ import annotations

import os
import subprocess
import sys
import unittest

import _support

from metamorpher import (
    ActionNode,
    AuditPolicy,
    CandidateStructure,
    ClaimTier,
    CognitiveLoop,
    Constraint,
    ConstraintKind,
    ConstraintRevision,
    DomainTag,
    ExecutionResult,
    InMemoryCapsuleStore,
    Hypothesis,
    MetamorpherController,
    Observation,
    StructuralCapsule,
    RefinementProposal,
    TypedActionGraph,
    UnresolvedCell,
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


class RaisingExecutor:
    def execute(self, decision):
        raise RuntimeError("tool disconnected")


class RaisingLearner:
    def revise(self, controller, decision, result):
        raise RuntimeError("learner failed")


class PerceiverAdapter:
    def perceive(self, raw_input, domain, *, action_token=None):
        from metamorpher import Perception

        return Perception(
            (Observation("raw", "signal", raw_input, source="camera"),),
            {"embedding": (1.0, 2.0)},
            "camera",
        )


class ProposerAdapter:
    def __init__(self):
        self.context = None

    def propose(self, context, observations, domain):
        self.context = context
        return CandidateStructure(
            nodes=(ActionNode("inspect", "inspect"),),
            rationale="perceived an inspectable signal",
        )


class DiscriminatorAdapter:
    def discriminate(self, cell, context, domain):
        return RefinementProposal(
            ActionNode("probe", "probe"),
            targets=tuple(cell.hypotheses),
            rationale="probe separates the surviving hypotheses",
        )


class AuditorAdapter:
    def collect(self, probe, domain):
        return (Observation("audit", "audited", True, source="auditor"),)


class CognitiveLoopTests(unittest.TestCase):
    def test_precommitted_audit_is_ingested_outside_value_policy(self) -> None:
        domain = DomainTag("engine")
        controller = MetamorpherController(graph_with("act"), default_domain=domain)
        step = CognitiveLoop(
            controller,
            executor=Executor(),
            audit_policy=AuditPolicy(every=1, budget=1),
            auditor=AuditorAdapter(),
        ).step()
        self.assertIsNotNone(step.audit_receipt)
        self.assertTrue(controller.evidence.resolve("audited").value)
    def test_discriminator_probe_remains_quarantined_until_promoted(self) -> None:
        domain = DomainTag("engine")
        controller = MetamorpherController(graph_with("inspect"))
        controller.version_space.add(
            UnresolvedCell(
                "active",
                {
                    "h1": Hypothesis("h1", frozenset({"inspect"}), domain=domain),
                    "h2": Hypothesis("h2", frozenset({"inspect"}), domain=domain),
                },
            ),
            activate=True,
        )
        loop = CognitiveLoop(
            controller,
            executor=Executor(),
            discriminator=DiscriminatorAdapter(),
        )
        loop.request_refinement(domain)
        candidate_id = next(
            cell_id
            for cell_id in controller.version_space.cells
            if cell_id.startswith("candidate-cell-")
        )
        self.assertNotIn(
            "probe",
            controller.version_space.active_for(domain).common_safe_actions(domain),
        )
        controller.observe(
            Observation("safe-probe", "probe_safe", True, domain=domain, independent_audit=True)
        )
        loop.promote_candidate_cell(candidate_id, ("safe-probe",), "probe safety observed")
        self.assertIn(
            "probe",
            controller.version_space.active_for(domain).common_safe_actions(domain),
        )
    def test_ingestion_connects_perception_capsule_retrieval_and_proposal(self) -> None:
        domain = DomainTag("engine")
        capsules = InMemoryCapsuleStore()
        capsules.put(StructuralCapsule("prior", domain))
        proposer = ProposerAdapter()
        controller = MetamorpherController(graph_with("seed"))
        ingestion = CognitiveLoop(
            controller,
            executor=Executor(),
            capsules=capsules,
            perceiver=PerceiverAdapter(),
            proposer=proposer,
        ).ingest("tick", domain)
        self.assertEqual(ingestion.perception.source, "camera")
        self.assertIn("inspect", controller.graph.nodes)
        self.assertEqual(proposer.context["capsules"][0].id, "prior")
        self.assertEqual(proposer.context["memory"], ())
        self.assertEqual(controller.evidence.resolve("signal").value, "tick")

    def test_supported_active_structure_can_be_captured_as_capsule(self) -> None:
        domain = DomainTag("engine")
        controller = MetamorpherController(graph_with("inspect"))
        controller.version_space.add(
            UnresolvedCell(
                "active",
                {"h": Hypothesis("h", frozenset({"inspect"}), domain=domain)},
            ),
            activate=True,
        )
        controller.observe(
            Observation("support", "worked", True, domain=domain, independent_audit=True)
        )
        loop = CognitiveLoop(controller, executor=Executor())
        capsule = loop.capture_capsule(
            "learned", domain, evidence_ids=("support",)
        )
        self.assertEqual(loop.capsules.get("learned"), capsule)
        self.assertEqual(controller.memory.get("worked", domain).positive, 2.0)
    def test_executor_exception_fails_committed_action_without_claiming_outcome(self) -> None:
        controller = MetamorpherController(graph_with("act"))
        with self.assertRaisesRegex(RuntimeError, "tool disconnected"):
            CognitiveLoop(controller, executor=RaisingExecutor()).step()
        self.assertIsNone(controller.committed_decision)
        self.assertEqual(controller.state.status_of("act").value, "failed")
        failure = controller.evidence.records("execution_outcome")[0]
        self.assertEqual(failure.status.value, "censored")

    def test_learner_exception_still_records_real_external_observation(self) -> None:
        controller = MetamorpherController(graph_with("act"))
        with self.assertRaisesRegex(RuntimeError, "learner failed"):
            CognitiveLoop(
                controller,
                executor=Executor(),
                learner=RaisingLearner(),
            ).step()
        self.assertIsNone(controller.committed_decision)
        self.assertTrue(controller.evidence.resolve("worked").value)

    def test_step_fills_decision_domain_at_external_boundary(self) -> None:
        domain = DomainTag("engine")
        controller = MetamorpherController(graph_with("act"), default_domain=domain)
        step = CognitiveLoop(controller, executor=Executor()).step()
        self.assertEqual(step.result.observations[0].domain, domain)
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

    def test_candidate_hypotheses_require_evidence_before_activation(self) -> None:
        domain = DomainTag("engine")
        controller = MetamorpherController(graph_with("inspect"))
        loop = CognitiveLoop(controller, executor=Executor())
        loop.install_candidates(
            CandidateStructure(
                hypotheses=(
                    Hypothesis("h", frozenset({"inspect"}), domain=domain),
                ),
                rationale="retrieved provisional model",
            )
        )
        cell_id = next(iter(controller.version_space.cells))
        self.assertIsNone(controller.version_space.active_for(domain))
        controller.observe(
            Observation(
                "support",
                "inspection_supported",
                True,
                domain=domain,
                independent_audit=True,
            )
        )
        loop.promote_candidate_cell(cell_id, ("support",), "observed support")
        self.assertEqual(controller.version_space.active_for(domain).id, cell_id)

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
