"""CPU-only semantic and manifest tests; no policy/environment rollout."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import build_local_branch_manifest as builder
import finalize_local_results as finalizer
from train_local_head_grid import preflight
from local_common import (
    ConstantLocalHead, FirstLocalOverride, StateDrivenLocalMachine, content_hash, sha256,
)


class FakeContext:
    def __init__(self, step: int):
        self.step = step
        self.feature = np.full(214, step, dtype=np.float64)
        self.u_flow = np.full((2, 2), 0.2)
        self.u_safe = np.full((2, 2), 0.2)
        self.first_projection_retry = False


class FakeEnv:
    def __init__(self, horizon: int = 5):
        self.done = False
        self.step_count = 0
        self.horizon = horizon


class FakeKernel:
    def __init__(self, horizon: int = 5):
        self.env = FakeEnv(horizon)
        self.prepare_calls = self.execute_calls = self.second_calls = 0

    def prepare(self, *, need_feature: bool):
        self.prepare_calls += 1
        if not need_feature: raise AssertionError("local decisions always require h")
        return FakeContext(self.env.step_count)

    def second_projection(self, context, correction):
        self.second_calls += 1
        return context.u_safe + correction, False

    def execute(self, action):
        self.execute_calls += 1
        self.env.step_count += 1
        self.env.done = self.env.step_count >= self.env.horizon
        return self.env.done, {"termination": "timeout" if self.env.done else None}


class FakeDirect:
    def __init__(self): self.calls = 0
    def __call__(self, batch):
        self.calls += 1
        return np.full((1, 4), 0.1)


class LocalMachineTests(unittest.TestCase):
    def test_each_step_is_a_fresh_state_decision_without_clock(self):
        decisions = iter((False, True, True, False))
        direct = FakeDirect(); kernel = FakeKernel(4)
        machine = StateDrivenLocalMachine(
            kernel=kernel, local_head=lambda feature: next(decisions), direct_model=direct,
        )
        rows = machine.run_to_terminal()
        self.assertEqual([row.decision for row in rows], ["SAFETY", "LOCAL", "LOCAL", "SAFETY"])
        self.assertEqual(machine.head_query_count, 4)
        self.assertEqual(machine.direct_query_count, 2)
        self.assertEqual(direct.calls, 2)
        self.assertEqual(kernel.second_calls, 2)

    def test_first_action_override_only_changes_current_action(self):
        override = FirstLocalOverride(1, ConstantLocalHead(False))
        feature = np.zeros(214)
        self.assertTrue(override(feature))
        self.assertFalse(override(feature))
        self.assertFalse(override(feature))


class ManifestTests(unittest.TestCase):
    def test_bootstrap_is_strictly_paired_and_source_grouped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct = root / "direct.npz"; np.savez(direct, marker=np.asarray(1))
            states = []
            for split, count in (("train", 20), ("validation", 10)):
                for index in range(count):
                    state_file = root / f"{split}_{index}.npz"; np.savez(state_file, marker=np.asarray(index))
                    states.append({
                        "state_id": f"{split}_{index}", "root_source_id": f"root_{split}_{index}",
                        "split": split, "state_file": str(state_file), "state_sha256": sha256(state_file),
                        "absolute_step": 100, "feature": np.zeros(214).tolist(),
                        "current_u_flow": np.zeros((2, 2)).tolist(),
                        "current_u_safe": np.zeros((2, 2)).tolist(),
                        "current_flow_realization_id": f"flow_{split}_{index}",
                        "current_flow_key_data": [0, index], "selection_outcome_blind": True,
                    })
            payload = {
                "status": "COMPLETE_FROZEN", "decision_kind": "entry", "states": states,
                "source_manifest_sha256": "frozen-source-split",
            }
            payload["content_sha256"] = content_hash(payload)
            meta = root / "states.json"; meta.write_text(json.dumps(payload))
            original = (builder.SOURCE_META, builder.DIRECT_CHECKPOINT, builder.DIRECT_SHA256, builder.MANIFEST_DIR)
            builder.SOURCE_META, builder.DIRECT_CHECKPOINT = meta, direct
            builder.DIRECT_SHA256, builder.MANIFEST_DIR = sha256(direct), root / "manifests"
            try:
                manifest = builder.build("local_bootstrap_safety", root / "out.json")
            finally:
                builder.SOURCE_META, builder.DIRECT_CHECKPOINT, builder.DIRECT_SHA256, builder.MANIFEST_DIR = original
            self.assertEqual(manifest["decision_count"], 24)
            self.assertEqual(manifest["task_count"], 24*32*2)
            self.assertEqual(manifest["selection"]["split_counts"], {"train": 16, "validation": 8})
            grouped = {}
            for task in manifest["tasks"]:
                grouped.setdefault((task["decision_id"], task["future_stream_id"]), []).append(task)
            self.assertEqual(len(grouped), 24*32)
            self.assertTrue(all({row["action"] for row in pair} == {0, 1} for pair in grouped.values()))
            self.assertEqual(manifest["downstream_policy"]["kind"], "SAFETY_ONLY")

    def test_finalizer_and_shared_head_trainer_preflight_are_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); result_dir = root / "results"; result_dir.mkdir()
            decisions, tasks = [], []
            for split, offset in (("train", 0), ("validation", 1)):
                decision = {
                    "decision_id": f"d{offset}", "pass_id": "test_pass",
                    "root_source_id": f"root{offset}", "split": split,
                    "state_id": f"state{offset}", "state_file": "/unused",
                    "state_sha256": f"hash{offset}", "absolute_step": 100,
                    "feature": np.zeros(214).tolist(), "current_u_flow": np.zeros((2,2)).tolist(),
                    "current_u_safe": np.zeros((2,2)).tolist(),
                    "current_flow_realization_id": f"flow{offset}",
                    "current_flow_key_data": [0, offset], "downstream_policy_hash": "downstream",
                }
                decisions.append(decision)
                for future in range(32):
                    for action in (0, 1):
                        index = len(tasks); stream = f"d{offset}-f{future}"
                        tasks.append({
                            **decision, "task_index": index, "task_id": f"{stream}-a{action}",
                            "action": action, "future_index": future, "future_rollout_id": future+100*offset,
                            "future_stream_id": stream, "future_rng_state_hash": f"rng{offset}-{future}",
                        })
            manifest = {
                "schema": "semantic_local_paired_branch_manifest_v1",
                "status": "FROZEN_READY_FOR_EXECUTION", "stage": "test_pass", "pass_id": "test_pass",
                "decision_kind": "local", "matched_future_semantics": {"count_per_decision": 32},
                "decision_count": 2, "task_count": len(tasks), "decisions": decisions, "tasks": tasks,
                "budget": {"maximum_physical_steps": len(tasks)},
            }
            manifest["content_sha256"] = content_hash(manifest)
            manifest_path = root / "manifest.json"; manifest_path.write_text(json.dumps(manifest))
            for task in tasks:
                action = int(task["action"]); outcome = "success" if action else "timeout"
                row = {
                    "schema": "semantic_local_paired_branch_result_v1", "record_complete": True,
                    **{key: task[key] for key in (
                        "task_id", "task_index", "decision_id", "pass_id", "action", "root_source_id",
                        "split", "state_id", "absolute_step", "current_flow_realization_id",
                        "future_stream_id", "future_rollout_id", "future_rng_state_hash", "downstream_policy_hash",
                    )},
                    "state_hash": task["state_sha256"], "mode": "NORMAL", "outcome": outcome,
                    "success": outcome == "success", "deadlock": False, "timeout": outcome == "timeout",
                    "collision": False, "other_failure": False, "remaining_jdef": .1*action,
                    "runtime_seconds": .01, "physical_transition_count": 1, "remaining_physical_steps": 1,
                    "terminal_global_step": 101, "flow_sample_count": 1, "head_query_count": 1,
                    "direct_query_count": action, "local_action_count": action,
                    "first_decision_override_calls": 1, "invalid_actions": 0, "nan_inf_events": 0,
                    "projection_failures": 0, "execution_error": None,
                    "manifest_content_sha256": manifest["content_sha256"],
                    "eta_latched_hash": "NOT_APPLICABLE",
                }
                (result_dir / f"task_{task['task_index']:05d}.json").write_text(json.dumps(row))
            frozen = finalizer.finalize(manifest_path, result_dir, root / "finalized")
            self.assertEqual(frozen["head_kind"], "entry_g")
            dataset, summary = preflight(root / "finalized/decision_dataset_manifest.json", root / "training")
            self.assertEqual(dataset.input_dimension, 214)
            self.assertEqual(summary["branch_rollouts"], 128)


if __name__ == "__main__":
    unittest.main(verbosity=2)
