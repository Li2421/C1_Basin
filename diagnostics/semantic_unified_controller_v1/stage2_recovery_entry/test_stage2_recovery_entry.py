"""CPU-only Stage-2 manifest, target, and trainer-preflight tests."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import build_mode_branch_manifest as builder
import finalize_mode_results as finalizer
from mode_common import FrozenModeHead, content_hash, sha256
from train_mode_head_grid import load_dataset


class ThreeWayEvidenceTests(unittest.TestCase):
    @staticmethod
    def rows(action: int, successes: int):
        return [{
            "future_stream_id": f"f{i:02d}", "success": i < successes,
            "remaining_jdef": float(action),
        } for i in range(32)]

    def test_recovery_requires_supported_advantage_over_both_alternatives(self):
        evidence = finalizer.summarize("d", {
            0: self.rows(0, 0), 1: self.rows(1, 0), 2: self.rows(2, 32),
        })
        self.assertEqual(evidence.target_action, 2)
        self.assertFalse(evidence.unresolved)

    def test_no_discordance_is_not_forced_to_safety_at_n32(self):
        evidence = finalizer.summarize("d", {
            0: self.rows(0, 32), 1: self.rows(1, 32), 2: self.rows(2, 32),
        })
        self.assertTrue(evidence.unresolved)
        self.assertIsNone(evidence.target_action)


class ManifestTests(unittest.TestCase):
    def test_replaceable_generic_manifest_and_three_matched_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct = root / "direct.npz"; np.savez(direct, x=np.asarray(1))
            eta = root / "eta.npz"; np.savez(eta, x=np.asarray(1))
            local = root / "local.npz"; np.savez(local, x=np.asarray(1))
            states = []
            for split, count in (("train", 45), ("validation", 25)):
                for index in range(count):
                    state = root / f"{split}_{index}.npz"; np.savez(state, x=np.asarray(index))
                    states.append({
                        "state_id": f"{split}_{index}", "root_source_id": f"root_{split}_{index}",
                        "split": split, "state_file": str(state), "state_sha256": sha256(state),
                        "absolute_step": 100, "feature": np.zeros(214).tolist(),
                        "current_u_flow": np.zeros((2, 2)).tolist(),
                        "current_u_safe": np.zeros((2, 2)).tolist(),
                        "current_flow_realization_id": f"flow_{split}_{index}",
                        "current_flow_key_data": [0, index], "selection_outcome_blind": True,
                    })
            payload = {"status": "COMPLETE_FROZEN", "decision_kind": "normal", "states": states}
            payload["content_sha256"] = content_hash(payload)
            source = root / "generic.json"; source.write_text(json.dumps(payload))
            original = (
                builder.DIRECT_CHECKPOINT, builder.DIRECT_SHA256, builder.ETA_CHECKPOINT,
                builder.ETA_SHA256, builder.LOCAL_HEAD, builder.prior_branch_budget,
            )
            builder.DIRECT_CHECKPOINT, builder.DIRECT_SHA256 = direct, sha256(direct)
            builder.ETA_CHECKPOINT, builder.ETA_SHA256 = eta, sha256(eta)
            builder.LOCAL_HEAD = local
            builder.prior_branch_budget = lambda output: (0, 0, [])
            try:
                manifest = builder.build("mode_bootstrap_local_downstream", root / "out.json", source)
            finally:
                (builder.DIRECT_CHECKPOINT, builder.DIRECT_SHA256, builder.ETA_CHECKPOINT,
                 builder.ETA_SHA256, builder.LOCAL_HEAD, builder.prior_branch_budget) = original
            self.assertEqual(manifest["decision_count"], 60)
            self.assertEqual(manifest["selection"]["split_counts"], {"train": 40, "validation": 20})
            self.assertEqual(manifest["task_count"], 60*32*3)
            grouped = {}
            for task in manifest["tasks"]:
                grouped.setdefault((task["decision_id"], task["future_stream_id"]), set()).add(task["action"])
            self.assertTrue(all(actions == {0, 1, 2} for actions in grouped.values()))

    def test_finalizer_snapshot_is_trainer_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); result_dir = root / "results"; result_dir.mkdir()
            decisions, tasks = [], []
            for split, suffix in (("train", "t"), ("validation", "v")):
                decision = {
                    "decision_id": f"d{suffix}", "pass_id": "test", "root_source_id": f"root{suffix}",
                    "split": split, "state_id": f"state{suffix}", "state_file": "/unused",
                    "state_sha256": f"hash{suffix}", "absolute_step": 100,
                    "feature": np.zeros(214).tolist(), "current_u_flow": np.zeros((2,2)).tolist(),
                    "current_u_safe": np.zeros((2,2)).tolist(), "current_flow_realization_id": f"flow{suffix}",
                    "current_flow_key_data": [0, 1], "downstream_policy_hash": "downstream",
                }
                decisions.append(decision)
                for future in range(32):
                    for action, name in finalizer.ACTION_NAMES.items():
                        index = len(tasks); stream = f"d{suffix}-f{future}"
                        tasks.append({
                            **decision, "task_index": index, "task_id": f"{stream}-a{action}",
                            "action": action, "action_name": name, "future_index": future,
                            "future_rollout_id": future, "future_stream_id": stream,
                            "future_rng_state_hash": f"rng-{suffix}-{future}",
                        })
            manifest = {
                "schema": "semantic_mode_paired_branch_manifest_v1", "status": "FROZEN_READY_FOR_EXECUTION",
                "stage": "test", "pass_id": "test", "matched_future_semantics": {"count_per_decision": 32},
                "decisions": decisions, "tasks": tasks,
            }
            manifest["content_sha256"] = content_hash(manifest)
            manifest_path = root / "manifest.json"; manifest_path.write_text(json.dumps(manifest))
            for task in tasks:
                action = int(task["action"]); success = action == 2
                transitions = 1
                row = {
                    "schema": "semantic_mode_paired_branch_result_v1", "record_complete": True,
                    **{key: task[key] for key in (
                        "task_id", "task_index", "decision_id", "pass_id", "action", "action_name",
                        "root_source_id", "split", "state_id", "absolute_step", "current_flow_realization_id",
                        "future_stream_id", "future_rollout_id", "future_rng_state_hash", "downstream_policy_hash",
                    )},
                    "state_hash": task["state_sha256"], "mode": "NORMAL",
                    "success": success, "outcome": "success" if success else "timeout",
                    "deadlock": False, "timeout": not success, "collision": False, "other_failure": False,
                    "remaining_jdef": float(action), "runtime_seconds": .01,
                    "physical_transition_count": transitions, "remaining_physical_steps": transitions,
                    "terminal_global_step": 101, "flow_sample_count": transitions,
                    "local_head_query_count": transitions if action in (0,1) else 0,
                    "direct_query_count": action if action in (0,1) else 0,
                    "local_action_count": action if action in (0,1) else 0,
                    "eta_query_count": 1 if action == 2 else 0,
                    "recovery_transition_count": transitions if action == 2 else 0,
                    "eta_latched": [0.,0.,0.] if action == 2 else None,
                    "eta_latched_hash": "eta" if action == 2 else "NOT_APPLICABLE",
                    "invalid_actions": 0, "nan_inf_events": 0, "projection_failures": 0,
                    "current_replay_max_abs": 0., "execution_error": None,
                    "manifest_content_sha256": manifest["content_sha256"],
                }
                (result_dir / f"task_{task['task_index']:05d}.json").write_text(json.dumps(row))
            frozen = finalizer.finalize(manifest_path, result_dir, root / "finalized")
            self.assertEqual(frozen["resolved_train"], 1)
            self.assertEqual(frozen["resolved_validation"], 1)
            loaded, data = load_dataset(root / "finalized/mode_dataset_manifest.json")
            self.assertEqual(data["features"].shape, (2, 214))
            self.assertEqual(loaded["input_dimension"], 214)


class InferenceTests(unittest.TestCase):
    def test_frozen_mode_head_uses_argmax(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mode.npz"
            np.savez(
                path, normalization_mean=np.zeros(214), normalization_scale=np.ones(214),
                layer_0_weight=np.zeros((214,64)), layer_0_bias=np.zeros(64),
                layer_1_weight=np.zeros((64,64)), layer_1_bias=np.zeros(64),
                layer_2_weight=np.zeros((64,3)), layer_2_bias=np.asarray([0., 1., -1.]),
            )
            self.assertEqual(FrozenModeHead(path)(np.zeros(214)), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
