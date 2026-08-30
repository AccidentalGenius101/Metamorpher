from __future__ import annotations

import unittest

import _support  # noqa: F401

from metamorpher import (
    DiscriminatingPrediction,
    DomainTag,
    ExpansionCapsule,
    ExpansionRegistry,
    ExpansionStatus,
    LocalEvidencePacket,
    Observation,
    ProjectionMapping,
    RepresentationBoundary,
    RepresentationStatus,
    ResidualSignature,
)

DOMAIN = DomainTag("trajectory")


def residual(evidence_id: str = "failure") -> ResidualSignature:
    return ResidualSignature(
        "future-height",
        0,
        1,
        evidence_id,
        (("phase", "next-cycle"),),
    )


def capsule() -> ExpansionCapsule:
    return ExpansionCapsule(
        "circle-to-helix",
        DOMAIN,
        "circle",
        "helix",
        ProjectionMapping(
            "circle",
            "helix",
            ("angular-recurrence",),
            coordinate_map=(("angle", "angle"),),
        ),
        (residual(),),
        (DiscriminatingPrediction("height-test", "future_height", 1),),
        proposed_distinctions=("axial displacement",),
        rationale="recurrence is preserved while height changes",
    )


class RepresentationExpansionTests(unittest.TestCase):
    def test_boundary_separates_incompleteness_from_claim_truth(self) -> None:
        boundary = RepresentationBoundary(
            "circle",
            DOMAIN,
            RepresentationStatus.PREDICTIVE_BUT_INCOMPLETE,
            preserved_constraint_ids=("angular-recurrence",),
            residuals=(residual(),),
            evidence_ids=("failure",),
        )
        self.assertEqual(boundary.status.value, "predictive_but_incomplete")
        with self.assertRaisesRegex(ValueError, "cannot retain residuals"):
            RepresentationBoundary(
                "circle",
                DOMAIN,
                RepresentationStatus.ADEQUATE_IN_SCOPE,
                residuals=(residual(),),
            )

    def test_proposal_cannot_authorize_itself(self) -> None:
        registry = ExpansionRegistry()
        registry.propose(capsule())
        self.assertEqual(
            registry.get("circle-to-helix").status,
            ExpansionStatus.QUARANTINED,
        )
        with self.assertRaisesRegex(ValueError, "no usable evidence"):
            registry.promote(
                "circle-to-helix",
                (Observation("pretty", "elegance", True, domain=DOMAIN),),
            )

    def test_discriminating_evidence_promotes_expansion(self) -> None:
        registry = ExpansionRegistry()
        registry.propose(capsule())
        promoted = registry.promote(
            "circle-to-helix",
            (Observation("prediction-hit", "future_height", 1, domain=DOMAIN),),
        )
        self.assertEqual(promoted.status, ExpansionStatus.SUPPORTED)
        self.assertEqual(promoted.supporting_evidence_ids, ("prediction-hit",))
        self.assertEqual(
            promoted.projection.preserved_constraint_ids,
            ("angular-recurrence",),
        )

    def test_wrong_domain_does_not_promote_expansion(self) -> None:
        registry = ExpansionRegistry()
        registry.propose(capsule())
        with self.assertRaisesRegex(ValueError, "no usable evidence"):
            registry.promote(
                "circle-to-helix",
                (
                    Observation(
                        "prediction-hit",
                        "future_height",
                        1,
                        domain=DomainTag("different"),
                    ),
                ),
            )

    def test_packet_ranking_is_not_a_vote(self) -> None:
        surprising = LocalEvidencePacket(
            "rare",
            DOMAIN,
            (("device", "edge-1"),),
            "parent-prediction",
            ("rare-evidence",),
            (residual("rare-evidence"),),
            reproduction_prediction_id="height-test",
            reliability=0.9,
        )
        confirmations = tuple(
            LocalEvidencePacket(
                f"normal-{index}",
                DOMAIN,
                (("device", f"edge-{index + 2}"),),
                "parent-prediction",
                (f"normal-evidence-{index}",),
                (residual(f"normal-evidence-{index}"),),
                reliability=1.0,
                expected_under_parent=True,
            )
            for index in range(20)
        )
        ranked = ExpansionRegistry().ranked_packets((*confirmations, surprising))
        self.assertEqual(ranked[0].id, "rare")


if __name__ == "__main__":
    unittest.main()
