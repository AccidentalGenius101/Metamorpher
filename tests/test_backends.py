from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import unittest
from unittest.mock import patch

import _support

from metamorpher.backends import (
    NumPyBackend,
    ReferenceBackend,
    TritonBackend,
    doctor_dict,
    get_backend,
)

PENDING = [[1, 1, 1, 1], [1, 1, 1, 1]]
COMPLETED = [[1, 0, 0, 0], [1, 1, 0, 0]]
PREREQUISITES = [
    [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
    ],
    [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
    ],
]
FEATURES = [
    [[0.0, 0.0], [4.0, 1.0], [100.0, 0.0], [3.0, 0.0]],
    [[0.0, 0.0], [9.0, 0.0], [5.0, 1.0], [4.0, 0.0]],
]
WEIGHTS = [1.0, -1.0]


def nested(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


class ReferenceBackendTests(unittest.TestCase):
    def test_hard_frontier_uses_all_prerequisites(self) -> None:
        backend = ReferenceBackend()
        actual = backend.hard_frontier_mask(PENDING, COMPLETED, PREREQUISITES)
        self.assertEqual(actual, [[False, True, False, True], [False, False, True, True]])

    def test_masked_scoring_never_selects_blocked_high_value_action(self) -> None:
        backend = ReferenceBackend()
        result = backend.frontier_and_score(PENDING, COMPLETED, PREREQUISITES, FEATURES, WEIGHTS)
        self.assertEqual(result.selected, [1, 2])
        self.assertEqual(result.scores[0][2], -math.inf)
        self.assertEqual(result.scores[1][1], -math.inf)

    def test_nonfinite_action_is_invalidated_locally(self) -> None:
        backend = ReferenceBackend()
        features = [[[math.nan], [2.0]]]
        result = backend.frontier_and_score(
            [[1, 1]],
            [[0, 0]],
            [[[0, 0], [0, 0]]],
            features,
            [1.0],
        )
        self.assertEqual(result.selected, [1])
        self.assertEqual(result.scores[0][0], -math.inf)

    def test_all_invalid_scores_return_no_selection(self) -> None:
        backend = ReferenceBackend()
        self.assertEqual(backend.select([[-math.inf, math.nan]]), [-1])

    def test_ties_choose_smallest_index(self) -> None:
        self.assertEqual(ReferenceBackend().select([[3.0, 3.0, 2.0]]), [0])

    def test_shape_errors_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            ReferenceBackend().hard_frontier_mask([[1, 1]], [[0]], [[[0, 0], [0, 0]]])


class BackendSelectionTests(unittest.TestCase):
    def test_reference_is_always_available(self) -> None:
        backend = get_backend("reference", strict=True)
        self.assertEqual(backend.name, "reference")
        self.assertTrue(backend.doctor().available)

    def test_doctor_is_machine_readable(self) -> None:
        report = doctor_dict()
        self.assertEqual(set(report), {"reference", "numpy", "triton"})
        for value in report.values():
            self.assertIn("available", value)
            self.assertIn("capabilities", value)

    def test_unknown_backend_name_rejected(self) -> None:
        with self.assertRaises(ValueError):
            get_backend("quantum")  # type: ignore[arg-type]

    def test_explicit_optional_backend_falls_back_or_fails_strictly(self) -> None:
        with patch.object(NumPyBackend, "is_available", return_value=False):
            self.assertEqual(get_backend("numpy").name, "reference")
            with self.assertRaises(RuntimeError):
                get_backend("numpy", strict=True)

    def test_doctor_report_is_json_serializable(self) -> None:
        encoded = json.dumps(doctor_dict(), sort_keys=True)
        self.assertIn('"reference"', encoded)

    def test_importing_backends_does_not_eagerly_import_optional_stacks(self) -> None:
        source = (
            "import sys; import metamorpher.backends; "
            "bad=[x for x in ('numpy','torch','triton') if x in sys.modules]; "
            "sys.exit(','.join(bad)) if bad else None"
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(_support.SRC_ROOT)
        completed = subprocess.run(
            [sys.executable, "-c", source],
            text=True,
            capture_output=True,
            env=environment,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)


@unittest.skipUnless(NumPyBackend().is_available(), "NumPy optional backend unavailable")
class NumPyParityTests(unittest.TestCase):
    def test_frontier_score_and_selection_match_reference(self) -> None:
        reference = ReferenceBackend().frontier_and_score(
            PENDING, COMPLETED, PREREQUISITES, FEATURES, WEIGHTS
        )
        accelerated = NumPyBackend().frontier_and_score(
            PENDING, COMPLETED, PREREQUISITES, FEATURES, WEIGHTS
        )
        self.assertEqual(nested(accelerated.frontier), reference.frontier)
        self.assertEqual(nested(accelerated.selected), reference.selected)
        actual_scores = nested(accelerated.scores)
        for expected_row, actual_row in zip(reference.scores, actual_scores):
            for expected, actual in zip(expected_row, actual_row):
                if expected == -math.inf:
                    self.assertEqual(actual, -math.inf)
                else:
                    self.assertAlmostEqual(actual, expected, places=10)

    def test_nonfinite_action_handling_matches_reference(self) -> None:
        frontier = [[1, 1, 0]]
        features = [[[math.nan], [2.0], [999.0]]]
        expected = ReferenceBackend().frontier_and_score(
            frontier, [[0, 0, 0]], [[[0, 0, 0]] * 3], features, [1.0]
        )
        actual = NumPyBackend().frontier_and_score(
            frontier, [[0, 0, 0]], [[[0, 0, 0]] * 3], features, [1.0]
        )
        self.assertEqual(nested(actual.selected), expected.selected)
        self.assertEqual(nested(actual.frontier), expected.frontier)

    def test_seeded_randomized_parity(self) -> None:
        rng = random.Random(20260828)
        reference_backend = ReferenceBackend()
        numpy_backend = NumPyBackend()
        for _ in range(100):
            batch = rng.randint(1, 4)
            actions = rng.randint(1, 12)
            feature_count = rng.randint(1, 7)
            pending = [[rng.choice((0, 1)) for _ in range(actions)] for _ in range(batch)]
            completed = [[rng.choice((0, 1)) for _ in range(actions)] for _ in range(batch)]
            prerequisites = [
                [
                    [rng.choice((0, 0, 0, 1)) for _ in range(actions)]
                    for _ in range(actions)
                ]
                for _ in range(batch)
            ]
            features = [
                [
                    [rng.uniform(-5.0, 5.0) for _ in range(feature_count)]
                    for _ in range(actions)
                ]
                for _ in range(batch)
            ]
            weights = [rng.uniform(-2.0, 2.0) for _ in range(feature_count)]
            expected = reference_backend.frontier_and_score(
                pending, completed, prerequisites, features, weights
            )
            actual = numpy_backend.frontier_and_score(
                pending, completed, prerequisites, features, weights
            )
            self.assertEqual(nested(actual.frontier), expected.frontier)
            self.assertEqual(nested(actual.selected), expected.selected)
            for expected_row, actual_row in zip(expected.scores, nested(actual.scores)):
                for expected_score, actual_score in zip(expected_row, actual_row):
                    if expected_score == -math.inf:
                        self.assertEqual(actual_score, -math.inf)
                    else:
                        self.assertAlmostEqual(actual_score, expected_score, places=9)


@unittest.skipUnless(TritonBackend().is_available(), "CUDA/Triton optional backend unavailable")
class TritonParityTests(unittest.TestCase):
    def test_frontier_score_and_selection_match_reference(self) -> None:
        reference = ReferenceBackend().frontier_and_score(
            PENDING, COMPLETED, PREREQUISITES, FEATURES, WEIGHTS
        )
        accelerated = TritonBackend().frontier_and_score(
            PENDING, COMPLETED, PREREQUISITES, FEATURES, WEIGHTS
        )
        self.assertEqual(nested(accelerated.frontier), reference.frontier)
        self.assertEqual(nested(accelerated.selected), reference.selected)
        actual_scores = nested(accelerated.scores)
        for expected_row, actual_row in zip(reference.scores, actual_scores):
            for expected, actual in zip(expected_row, actual_row):
                if expected == -math.inf:
                    self.assertEqual(actual, -math.inf)
                else:
                    self.assertAlmostEqual(actual, expected, places=5)


if __name__ == "__main__":
    unittest.main()
