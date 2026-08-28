from __future__ import annotations

import unittest

import _support  # noqa: F401 - installs the source tree on sys.path

from metamorpher.audit import AuditPolicy
from metamorpher.evidence import EvidenceLedger
from metamorpher.memory import DomainMemory
from metamorpher.model import DomainTag, Observation, ObservationStatus, TruthValue


class EvidenceLedgerTests(unittest.TestCase):
    def test_censored_is_unknown_not_negative(self) -> None:
        ledger = EvidenceLedger()
        for index in range(100):
            ledger.append(
                Observation(
                    f"c{index}",
                    "hidden_blocker",
                    False,
                    status=ObservationStatus.CENSORED,
                    source="not_audited",
                    censoring_reason="probe_not_run",
                )
            )
        fact = ledger.resolve("hidden_blocker")
        self.assertEqual(fact.state, TruthValue.UNKNOWN)
        self.assertEqual(fact.evidence_ids, ())
        self.assertEqual(len(ledger.records("hidden_blocker")), 100)
        self.assertEqual(ledger.records("hidden_blocker", include_censored=False), ())

    def test_duplicate_observation_is_idempotent(self) -> None:
        ledger = EvidenceLedger()
        observation = Observation("same", "bolt_present", True, source="inspection")
        self.assertTrue(ledger.append(observation))
        first_revision = ledger.revision
        self.assertFalse(ledger.append(observation))
        self.assertEqual(ledger.revision, first_revision)
        self.assertEqual(len(ledger.events), 1)

    def test_conflicting_observations_remain_unresolved(self) -> None:
        ledger = EvidenceLedger(ambiguity_margin=0.15)
        ledger.append(Observation("yes", "crack", True, source="camera"))
        ledger.append(Observation("no", "crack", False, source="human"))
        fact = ledger.resolve("crack")
        self.assertEqual(fact.state, TruthValue.UNKNOWN)
        self.assertEqual(set(fact.evidence_ids), {"yes", "no"})

    def test_explicit_negative_is_not_censorship(self) -> None:
        ledger = EvidenceLedger()
        ledger.append(Observation("seen", "bolt_present", False, source="inspection"))
        result = ledger.evaluate("bolt_present", True)
        self.assertEqual(result.state, TruthValue.VIOLATED)
        self.assertEqual(result.value, False)

    def test_action_token_preserves_delayed_attribution(self) -> None:
        ledger = EvidenceLedger()
        ledger.append(
            Observation(
                "late",
                "outcome",
                "failure",
                source="delayed_monitor",
                action_token="action-at-t0",
                timestamp=5.0,
            )
        )
        record = ledger.records("outcome")[0]
        self.assertEqual(record.action_token, "action-at-t0")
        self.assertEqual(record.timestamp, 5.0)


class DomainMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = DomainMemory()
        self.domain_a = DomainTag.from_mapping("truck", {"engine": "6.0", "side": "driver"})
        self.domain_b = DomainTag.from_mapping("truck", {"engine": "6.0", "side": "passenger"})

    def test_memory_is_domain_scoped(self) -> None:
        self.memory.record("missing_end_bolt", self.domain_a, False, "a1")
        self.assertIsNotNone(self.memory.get("missing_end_bolt", self.domain_a))
        self.assertIsNone(self.memory.get("missing_end_bolt", self.domain_b))

    def test_censored_record_does_not_change_confidence(self) -> None:
        record = self.memory.record("missing_end_bolt", self.domain_a, None, "c1", censored=True)
        self.assertEqual(record.confidence, 0.5)
        self.assertEqual(record.positive, 1.0)
        self.assertEqual(record.negative, 1.0)
        self.assertEqual(record.censored, 1)

    def test_observed_duplicate_is_idempotent(self) -> None:
        first = self.memory.record("claim", self.domain_a, True, "same")
        confidence = first.confidence
        second = self.memory.record("claim", self.domain_a, True, "same")
        self.assertEqual(second.confidence, confidence)
        self.assertEqual(second.evidence_ids, ["same"])

    def test_censored_duplicate_is_idempotent(self) -> None:
        first = self.memory.record("claim", self.domain_a, None, "same", censored=True)
        self.assertEqual(first.censored, 1)
        second = self.memory.record("claim", self.domain_a, None, "same", censored=True)
        self.assertEqual(second.censored, 1, "replaying censored evidence must not change memory")


class AuditPolicyTests(unittest.TestCase):
    def test_schedule_is_precommitted_and_budgeted(self) -> None:
        audit = AuditPolicy(every=3, budget=2)
        consumed = [index for index in range(1, 10) if audit.consume(index)]
        self.assertEqual(consumed, [3, 6])
        self.assertEqual(audit.used, 2)

    def test_disabled_schedule_never_audits(self) -> None:
        audit = AuditPolicy(every=0)
        self.assertFalse(any(audit.consume(index) for index in range(20)))


if __name__ == "__main__":
    unittest.main()
