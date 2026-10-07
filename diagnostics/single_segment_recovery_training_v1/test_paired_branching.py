"""CPU-only tests for paired branch manifests, hooks, and full state I/O."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import build_paired_branch_manifest as builder
from decision_learning import pair_branch_records
from decision_orchestration import build_head_dataset
from full_state_io import audit_round_trip, restore_full_history, save_full_history
from run_paired_branches import ConstantHead, FirstDecisionOverride


class PairedManifestTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.original = (builder.PROTOCOL, builder.SPLITS, builder.MANIFEST_DIR)
        builder.PROTOCOL = self.root / "protocol.json"
        builder.SPLITS = self.root / "splits.json"
        builder.MANIFEST_DIR = self.root / "manifests"
        builder.PROTOCOL.write_text(json.dumps({
            "branch_future_root_seed": 2026100201,
        }))
        split_payload = {"schema": "split", "status": "FROZEN"}
        split_payload["content_sha256"] = builder.canonical_hash(split_payload)
        builder.SPLITS.write_text(json.dumps(split_payload))

    def tearDown(self):
        builder.PROTOCOL, builder.SPLITS, builder.MANIFEST_DIR = self.original
        self.temporary.cleanup()

    def make_states(self, decision_kind="entry", status="COMPLETE_FROZEN"):
        rows = []
        for split, count in (("train", 20), ("validation", 10)):
            for index in range(count):
                state = self.root / f"{split}_{index}.npz"
                np.savez(state, marker=np.asarray(index))
                row = {
                    "state_id": f"{split}_{index}", "root_source_id": f"root_{split}_{index}",
                    "split": split, "state_file": str(state), "state_sha256": builder.sha256(state),
                    "absolute_step": 100 + index, "feature": np.zeros(214).tolist(),
                    "current_u_flow": np.zeros((2, 2)).tolist(),
                    "current_u_safe": np.zeros((2, 2)).tolist(),
                    "current_flow_realization_id": f"flow_{split}_{index}",
                    "current_flow_key_data": [0, index],
                }
                if decision_kind == "exit":
                    row.update({"eta_latched": [0.2, -0.1, 0.4], "entry_step": 90,
                                "recovery_transitions": 10 + index})
                rows.append(row)
        payload = {"schema": "states", "status": status,
                   "decision_kind": decision_kind, "states": rows}
        payload["content_sha256"] = builder.canonical_hash(payload)
        path = self.root / f"{decision_kind}_states.json"
        path.write_text(json.dumps(payload))
        return path

    def test_entry_bootstrap_has_32_strictly_paired_futures_and_no_entry_incumbent(self):
        state_path = self.make_states("entry")
        exit_checkpoint = self.root / "exit.npz"
        np.savez(exit_checkpoint, marker=np.asarray(1))
        result = builder.build_manifest(
            stage="entry_bootstrap_no_entry", state_manifest=state_path,
            output=self.root / "entry.json", exit_checkpoint=str(exit_checkpoint),
        )
        self.assertEqual(result["decision_count"], 24)
        self.assertEqual(result["task_count"], 24 * 32 * 2)
        self.assertEqual(result["selection"]["split_counts"], {"train": 16, "validation": 8})
        self.assertEqual(result["downstream_policy"]["entry"]["kind"], "NEVER_ENTER")
        self.assertEqual(result["downstream_policy"]["exit"]["kind"], "LEARNED_HEAD")
        grouped = {}
        for task in result["tasks"]:
            grouped.setdefault((task["decision_id"], task["future_stream_id"]), []).append(task)
        self.assertEqual(len(grouped), 24 * 32)
        for pair in grouped.values():
            self.assertEqual({row["action"] for row in pair}, {0, 1})
            for key in ("state_sha256", "absolute_step", "current_flow_realization_id",
                        "future_rollout_id", "future_rng_state_hash", "downstream_policy_hash"):
                self.assertEqual(pair[0][key], pair[1][key])

    def test_exit_bootstrap_encodes_continue_vs_exit_and_never_exit_incumbent(self):
        state_path = self.make_states("exit")
        result = builder.build_manifest(
            stage="exit_bootstrap_never_exit", state_manifest=state_path,
            output=self.root / "exit.json",
        )
        self.assertEqual(result["action_semantics"]["0"], "CONTINUE_NOW then frozen exit policy")
        self.assertEqual(result["action_semantics"]["1"], "EXIT_NOW then Safety forever")
        self.assertEqual(result["downstream_policy"]["exit"]["kind"], "NEVER_EXIT")
        self.assertTrue(all(task["eta_latched"] == [0.2, -0.1, 0.4] for task in result["tasks"]))

    def test_incomplete_collector_manifest_is_rejected(self):
        with self.assertRaises(RuntimeError):
            builder.build_manifest(
                stage="exit_bootstrap_never_exit",
                state_manifest=self.make_states("exit", status="COLLECTING"),
                output=self.root / "forbidden.json",
            )

    def test_paired_records_are_compatible_with_strict_label_pairer(self):
        shared = {
            "decision_id": "d", "future_stream_id": "f", "root_source_id": "root",
            "split": "train", "state_hash": "state", "absolute_step": 111,
            "mode": "RECOVERY", "current_flow_realization_id": "current",
            "future_rng_state_hash": "future", "eta_latched_hash": "eta",
            "downstream_policy_hash": "policy",
        }
        rows = [
            {**shared, "action": 0, "success": 0, "remaining_jdef": 0.2},
            {**shared, "action": 1, "success": 1, "remaining_jdef": 0.0},
        ]
        result = pair_branch_records(rows)
        self.assertEqual(result["d"][0].success0, 0)
        self.assertEqual(result["d"][0].success1, 1)
        corrupted = [rows[0], {**rows[1], "future_rng_state_hash": "different"}]
        with self.assertRaises(ValueError):
            pair_branch_records(corrupted)

    def test_manifest_and_runner_shaped_rows_build_complete_head_dataset(self):
        state_path = self.make_states("entry")
        exit_checkpoint = self.root / "exit.npz"
        np.savez(exit_checkpoint, marker=np.asarray(1))
        manifest = builder.build_manifest(
            stage="entry_bootstrap_no_entry", state_manifest=state_path,
            output=self.root / "entry_compatibility.json",
            exit_checkpoint=str(exit_checkpoint),
        )
        train_decision = next(row for row in manifest["decisions"] if row["split"] == "train")
        validation_decision = next(
            row for row in manifest["decisions"] if row["split"] == "validation"
        )
        chosen_ids = {train_decision["decision_id"], validation_decision["decision_id"]}
        results = []
        for task in manifest["tasks"]:
            if task["decision_id"] not in chosen_ids:
                continue
            results.append({
                "decision_id": task["decision_id"], "pass_id": task["pass_id"],
                "downstream_policy_hash": task["downstream_policy_hash"],
                "root_source_id": task["root_source_id"], "split": task["split"],
                "state_hash": task["state_sha256"],
                "absolute_step": task["absolute_step"], "mode": "SAFETY_BEFORE",
                "current_flow_realization_id": task["current_flow_realization_id"],
                "future_rng_state_hash": task["future_rng_state_hash"],
                "eta_latched_hash": "UNLATCHED_AT_DECISION",
                "future_stream_id": task["future_stream_id"], "action": task["action"],
                "success": int(task["action"] == 1),
                "remaining_jdef": 0.1 if task["action"] else 0.0,
            })
        dataset = build_head_dataset(
            [train_decision, validation_decision], results,
            head_kind="entry_eta", expected_futures_per_input=32,
        )
        self.assertEqual(dataset.train_features.shape, (1, 214))
        self.assertEqual(dataset.validation_features.shape, (1, 214))
        self.assertEqual(dataset.manifest["branch_rollouts"], 128)
        self.assertEqual(dataset.train_evidence[0].rescue, 32)


class HookAndStateIOTests(unittest.TestCase):
    def test_first_decision_override_then_incumbent(self):
        override = FirstDecisionOverride(1, ConstantHead(False))
        feature = np.zeros(214)
        self.assertTrue(override(feature))
        self.assertFalse(override(feature))
        self.assertEqual(override.calls, 2)

    def test_full_state_round_trip_preserves_actual_history_from_zero(self):
        from single_integrator.environment import Config, GiveWayEnv

        config = Config(max_steps=20)
        env = GiveWayEnv(config)
        env.reset(np.array([[-0.9, 0.01], [0.9, -0.01]]))
        for _ in range(7):
            env.step(np.array([[0.1, 0.0], [-0.1, 0.0]]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.npz"
            save_full_history(path, env)
            audit = audit_round_trip(path, env, config)
            restored = restore_full_history(path, config)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["history_start_step"], 0)
        self.assertEqual(audit["full_history_points"], 8)
        np.testing.assert_array_equal(restored.distance_history, env.distance_history)

    def test_strict_legacy_full_marker_accepts_collector_but_rejects_tail41(self):
        from single_integrator.environment import Config

        source = Path(
            "/home/zhihan/research/Basin_C1/diagnostics/single_segment_recovery_training_v1/"
            "states/safety_anchors/train/single_segment_v1_train_0000_a0_t0452.npz"
        )
        restored = restore_full_history(source, Config())
        self.assertEqual(restored.step_count, 452)
        self.assertEqual(len(restored.distance_history), 453)
        with tempfile.TemporaryDirectory() as directory, np.load(source, allow_pickle=False) as data:
            payload = {key: np.asarray(data[key]) for key in data.files if key != "complete_real_history"}
            payload["error_history"] = payload["error_history"][-41:]
            payload["history_start_step"] = np.asarray(restored.step_count - 40)
            tail = Path(directory) / "tail41.npz"
            np.savez_compressed(tail, **payload)
            with self.assertRaises(RuntimeError):
                restore_full_history(tail, Config())


if __name__ == "__main__":
    unittest.main(verbosity=2)
