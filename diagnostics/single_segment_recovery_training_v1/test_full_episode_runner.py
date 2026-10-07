"""CPU-only integrity tests for frozen full-loop policy evaluation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import build_full_loop_manifest as builder
import run_full_episode_system as runner


def write_head(path: Path, dimension: int, threshold: float = 0.0) -> None:
    np.savez(
        path,
        normalization_mean=np.zeros(dimension),
        normalization_scale=np.ones(dimension),
        layer_0_weight=np.zeros((dimension, 64)), layer_0_bias=np.ones(64),
        layer_1_weight=np.zeros((64, 64)), layer_1_bias=np.ones(64),
        layer_2_weight=np.zeros((64, 1)), layer_2_bias=np.zeros(1),
        threshold_logit=np.asarray(threshold), metadata_json=np.asarray("{}"),
    )


class FullLoopManifestTests(unittest.TestCase):
    def test_validation_threshold_pair_is_part_of_policy_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry, exit_head = root / "entry.npz", root / "exit.npz"
            write_head(entry, 214)
            write_head(exit_head, 217)
            with mock.patch.object(builder, "MANIFEST_DIR", root / "manifests"):
                first = builder.build_manifest(
                    system="ETA", split="validation", entry_checkpoint=entry,
                    exit_checkpoint=exit_head, entry_threshold_probability=0.25,
                    exit_threshold_probability=0.75, output=root / "first.json",
                )
                second = builder.build_manifest(
                    system="ETA", split="validation", entry_checkpoint=entry,
                    exit_checkpoint=exit_head, entry_threshold_probability=0.50,
                    exit_threshold_probability=0.75, output=root / "second.json",
                )
            self.assertNotEqual(first["policy_hash"], second["policy_hash"])
            self.assertEqual(first["source_count"], 20)
            self.assertTrue(first["threshold_pair_is_predeclared_validation_candidate"])
            self.assertFalse(first["threshold_pair_selected_by_this_run"])
            self.assertEqual(first["budget"]["new_safety_continuations_upper_bound"], 0)

    def test_final_test_is_locked_without_double_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry, exit_head = root / "entry.npz", root / "exit.npz"
            write_head(entry, 214)
            write_head(exit_head, 217)
            with self.assertRaises(RuntimeError):
                builder.build_manifest(
                    system="ETA", split="final_test", entry_checkpoint=entry,
                    exit_checkpoint=exit_head, entry_threshold_probability=0.5,
                    exit_threshold_probability=0.5, output=root / "locked.json",
                    allow_final_test=False,
                )

    def test_final_test_requires_frozen_validation_calibration_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entry, exit_head = root / "entry.npz", root / "exit.npz"
            write_head(entry, 214)
            write_head(exit_head, 217)
            with self.assertRaises(RuntimeError):
                builder.build_manifest(
                    system="ETA", split="final_test", entry_checkpoint=entry,
                    exit_checkpoint=exit_head, entry_threshold_probability=0.5,
                    exit_threshold_probability=0.5, output=root / "no_gate.json",
                    allow_final_test=True, final_gate=None,
                )

    def test_invalid_head_normalization_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.npz"
            write_head(path, 214)
            with np.load(path, allow_pickle=False) as archive:
                payload = {key: np.asarray(archive[key]) for key in archive.files}
            payload["normalization_scale"][0] = 0.0
            np.savez(path, **payload)
            with self.assertRaises(RuntimeError):
                builder.inspect_head(path, 214)

    def test_threshold_override_changes_decision_without_weight_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "head.npz"
            write_head(path, 214, threshold=100.0)
            record = {
                "path": str(path), "sha256": runner.sha256(path),
                "threshold_logit": -1.0,
            }
            head = runner.ThresholdedFrozenHead(record, 214)
            self.assertTrue(head(np.zeros(214)))
            self.assertAlmostEqual(head.threshold, -1.0)


class SafetyReuseTests(unittest.TestCase):
    def test_development_safety_is_reused_without_new_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_id = "source_0"
            path = root / "runs/safety_raw/validation" / f"{source_id}.json"
            path.parent.mkdir(parents=True)
            task = {
                "source_id": source_id, "root_source_id": source_id,
                "split": "validation", "rollout_id": 3, "flow_root_seed": 11,
                "initial_positions": [[-0.7, 0.0], [0.7, 0.0]],
            }
            row = {
                "record_complete": True, "execution_error": None,
                "source_manifest_sha256": "frozen", "source_id": source_id,
                "split": "validation", "rollout_id": 3, "flow_root_seed": 11,
                "initial_positions": task["initial_positions"], "outcome": "success",
                "success": True, "terminal_step": 10,
            }
            path.write_text(json.dumps(row))
            with mock.patch.object(runner, "HERE", root), mock.patch.object(
                runner, "run_safety", side_effect=AssertionError("must reuse")
            ):
                observed, provenance, created = runner.safety_reference(
                    task, {"source_manifest_sha256": "frozen"},
                    config=None, cbf=None, sample_action=None, episode_key=None,
                )
            self.assertEqual(observed["outcome"], "success")
            self.assertEqual(provenance, "FROZEN_SOURCE_COLLECTION_REUSE")
            self.assertFalse(created)

    def test_combined_row_matches_full_loop_analysis_schema(self) -> None:
        task = {
            "source_id": "s", "root_source_id": "s", "split": "validation",
            "episode_index": 0, "rollout_id": 0, "flow_root_seed": 1,
        }
        manifest = {
            "policy_hash": "policy", "system": "ETA", "content_sha256": "manifest",
            "policy": {
                "entry_head": {"threshold_probability": 0.25},
                "exit_head": {"threshold_probability": 0.75},
            },
        }
        learned = {
            "record_complete": True, "execution_error": None,
            "learned_outcome": "success", "learned_success": True,
            "terminal_step": 100, "jdef": 0.1, "entry_step": 10, "exit_step": 20,
            "recovery_transition_count": 10, "recovery_segment_count": 1,
            "agent_collision": 0, "wall_collision": 0, "invalid_action": 0,
            "nan_inf": 0, "projection_solver_failure": 0,
        }
        safety = {
            "record_complete": True, "execution_error": None,
            "outcome": "timeout", "success": False, "terminal_step": 850,
        }
        row = runner.combine_result(task, manifest, learned, safety, "REUSE")
        self.assertTrue(row["record_complete"])
        self.assertEqual(row["safety_outcome"], "timeout")
        self.assertTrue(row["learned_success"])
        self.assertEqual(row["recovery_segment_count"], 1)

    def test_final_safety_lock_refuses_duplicate_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_id = "final_0"
            cache = root / "runs/full_loop_safety/final_test" / f"{source_id}.json"
            cache.parent.mkdir(parents=True)
            cache.with_suffix(".json.lock").write_text("held")
            task = {
                "source_id": source_id, "root_source_id": source_id,
                "split": "final_test", "rollout_id": 0, "flow_root_seed": 1,
                "initial_positions": [[-0.7, 0.0], [0.7, 0.0]],
            }
            manifest = {
                "final_test_explicitly_unlocked": True,
                "source_manifest_sha256": "frozen",
                "final_test_manifest_sha256": "final",
            }
            with mock.patch.object(runner, "HERE", root), mock.patch.object(
                runner, "run_safety", side_effect=AssertionError("must not duplicate")
            ):
                with self.assertRaises(RuntimeError):
                    runner.safety_reference(
                        task, manifest, config=None, cbf=None,
                        sample_action=None, episode_key=None,
                    )


if __name__ == "__main__":
    unittest.main()
