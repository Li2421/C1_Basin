"""Zero-training unit tests for startup-complete pipeline invariants."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .pipeline import (_split_counts, assert_fresh_output,
                       assert_v3_prefix_preserved, deterministic_episode_split,
                       validate_sample_alignment)


class SplitTests(unittest.TestCase):
    def test_episode_split_is_deterministic_and_group_exclusive(self):
        episodes = [f"episode_{index:03d}" for index in range(20)]
        ratios = {"train": 0.70, "validation": 0.15, "test": 0.15}
        first = deterministic_episode_split(episodes, 20260925, ratios)
        second = deterministic_episode_split(list(reversed(episodes)), 20260925, ratios)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(episodes))
        self.assertEqual({split: list(first.values()).count(split) for split in ratios},
                         {"train": 14, "validation": 3, "test": 3})

    def test_small_valid_split_has_every_partition(self):
        counts = _split_counts(3, {"train": 0.70, "validation": 0.15, "test": 0.15})
        self.assertEqual(sum(counts.values()), 3)
        self.assertTrue(all(value == 1 for value in counts.values()))

    def test_existing_episode_membership_is_inherited(self):
        episodes = [f"episode_{index:03d}" for index in range(10)]
        ratios = {"train": 0.70, "validation": 0.15, "test": 0.15}
        result = deterministic_episode_split(
            episodes, 20260925, ratios,
            fixed={"episode_000": "test", "episode_001": "validation"})
        self.assertEqual(result["episode_000"], "test")
        self.assertEqual(result["episode_001"], "validation")


class PrefixTests(unittest.TestCase):
    def test_exact_numeric_and_string_prefix_passes(self):
        old = {
            "features": np.asarray([[1.0, -0.0], [2.0, 3.0]], dtype=np.float64),
            "state_id": np.asarray(["a", "b"]),
        }
        merged = {
            "features": np.concatenate([old["features"], np.asarray([[4.0, 5.0]])]),
            "state_id": np.asarray(["a", "b", "long_startup_state"]),
        }
        assert_v3_prefix_preserved(old, merged)

    def test_changed_numeric_prefix_fails(self):
        old = {"features": np.asarray([[1.0]], dtype=np.float64)}
        merged = {"features": np.asarray([[1.0 + 1e-15], [2.0]], dtype=np.float64)}
        with self.assertRaises(AssertionError):
            assert_v3_prefix_preserved(old, merged)


class IntegrityTests(unittest.TestCase):
    def test_sample_metadata_state_alignment(self):
        arrays = {
            "sample_id": np.asarray(["s0"]), "state_id": np.asarray(["z0"]),
            "split": np.asarray(["train"]), "category": np.asarray(["STARTUP"]),
            "state_index": np.asarray([3]),
        }
        states = [{"state_id": "z0", "split": "train", "category": "STARTUP",
                   "state_index": 3}]
        metadata = [{"sample_id": "s0", "state_id": "z0", "split": "train",
                     "category": "STARTUP", "state_index": 3}]
        self.assertTrue(validate_sample_alignment(arrays, states, metadata, "fixture")["passed"])
        arrays["split"][0] = "test"
        with self.assertRaises(ValueError):
            validate_sample_alignment(arrays, states, metadata, "fixture")

    def test_output_must_not_preexist(self):
        with TemporaryDirectory() as directory:
            absent = Path(directory) / "new_artifacts"
            assert_fresh_output(absent)
            absent.mkdir()
            with self.assertRaises(FileExistsError):
                assert_fresh_output(absent)


if __name__ == "__main__":
    unittest.main()
