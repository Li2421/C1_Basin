"""CPU-only tests for fail-closed result finalization and training preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from finalize_paired_branch_results import canonical_hash, finalize
from merge_finalized_head_datasets import merge_finalized_datasets
from train_decision_head_grid import preflight


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AggregationTests(unittest.TestCase):
    def make_fixture(
        self, root: Path, *, stage: str = "synthetic", decision_prefix: str = "d",
        downstream_policy_hash: str = "policy",
    ):
        result_dir = root / "results"
        result_dir.mkdir(parents=True)
        decisions = []
        tasks = []
        for decision_index, split in enumerate(("train", "validation")):
            decision_id = f"{decision_prefix}{decision_index}"
            decision = {
                "pass_id": stage, "decision_id": decision_id,
                "decision_kind": "exit", "root_source_id": f"root-{split}",
                "split": split, "state_id": f"state-{split}",
                "state_file": "/unused", "state_sha256": f"hash-{split}",
                "absolute_step": 100, "feature": [0.0] * 214,
                "current_u_flow": [[0.0, 0.0], [0.0, 0.0]],
                "current_u_safe": [[0.0, 0.0], [0.0, 0.0]],
                "current_flow_realization_id": f"flow-{split}",
                "current_flow_key_data": [1, 2], "eta_latched": [0.1, 0.2, 0.3],
                "entry_step": 90, "recovery_transitions": 10,
                "downstream_policy_hash": downstream_policy_hash,
            }
            decisions.append(decision)
            for future in range(2):
                for action in (0, 1):
                    index = len(tasks)
                    tasks.append({
                        "task_index": index, "task_id": f"{decision_id}-f{future}-a{action}",
                        **decision, "action": action, "future_index": future,
                        "future_rollout_id": decision_index * 10 + future,
                        "future_stream_id": f"{decision_id}-f{future}",
                        "future_rng_state_hash": f"rng-{decision_index}-{future}",
                    })
        manifest = {
            "schema": "single_segment_paired_branch_manifest_v1",
            "status": "FROZEN_READY_FOR_EXECUTION", "stage": stage,
            "pass_id": stage, "decision_kind": "exit", "system": "ETA",
            "source_split_manifest_sha256": "shared-source-split",
            "matched_future_semantics": {"count_per_decision": 2},
            "downstream_policy_hash": downstream_policy_hash,
            "budget": {"maximum_physical_steps": 6000},
            "decision_count": 2, "task_count": len(tasks),
            "decisions": decisions, "tasks": tasks,
        }
        manifest["content_sha256"] = canonical_hash(manifest)
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        eta_hash = canonical_hash([0.1, 0.2, 0.3])
        for task in tasks:
            action = int(task["action"])
            transitions = 10 if action else 750
            row = {
                "schema": "single_segment_paired_branch_result_v1", "record_complete": True,
                "pass_id": stage, "task_id": task["task_id"],
                "task_index": task["task_index"], "decision_id": task["decision_id"],
                "decision_kind": "exit", "action": action,
                "root_source_id": task["root_source_id"], "split": task["split"],
                "state_id": task["state_id"], "state_hash": task["state_sha256"],
                "absolute_step": 100, "mode": "RECOVERY",
                "current_flow_realization_id": task["current_flow_realization_id"],
                "future_stream_id": task["future_stream_id"],
                "future_rollout_id": task["future_rollout_id"],
                "future_rng_state_hash": task["future_rng_state_hash"],
                "eta_latched_hash": eta_hash,
                "downstream_policy_hash": downstream_policy_hash,
                "success": bool(action), "outcome": "success" if action else "timeout",
                "deadlock": False, "timeout": not bool(action), "collision": False,
                "other_failure": False, "remaining_jdef": 0.1 * action,
                "terminal_global_step": 100 + transitions,
                "remaining_physical_steps": transitions,
                "entry_step": 90, "exit_step": 100 if action else None,
                "recovery_transitions": 10, "returned_to_safety": bool(action),
                "eta_query_count": 0, "flow_sample_count": transitions,
                "physical_transition_count": transitions, "first_decision_override_calls": 1,
                "first_projection_retry_count": 0, "second_projection_retry_count": 0,
                "agent_collision_events": 0, "wall_collision_events": 0,
                "invalid_actions": 0, "nan_inf_events": 0, "projection_failures": 0,
                "execution_error": None, "manifest_content_sha256": manifest["content_sha256"],
                "runtime_seconds": 0.01,
            }
            (result_dir / f"task_{task['task_index']:05d}.json").write_text(json.dumps(row))
        return manifest_path, result_dir

    def test_finalize_and_training_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, results = self.make_fixture(root)
            output = root / "finalized"
            finalized = finalize(manifest, results, output, require_task_count=8)
            self.assertEqual(finalized["result_count"], 8)
            self.assertTrue(finalized["all_results_complete"])
            self.assertEqual(finalized["head_dataset_manifest"]["branch_rollouts"], 8)
            dataset, summary = preflight(
                output / "decision_dataset_manifest.json", root / "training"
            )
            self.assertEqual(dataset.input_dimension, 217)
            self.assertEqual(summary["status"], "PREFLIGHT_PASS_TRAINING_NOT_LAUNCHED")

    def test_missing_or_errored_result_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, results = self.make_fixture(root)
            (results / "task_00007.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "coverage mismatch"):
                finalize(manifest, results, root / "out", require_task_count=8)
            manifest, results = self.make_fixture(root / "second")
            path = results / "task_00000.json"
            row = json.loads(path.read_text())
            row["record_complete"] = False
            row["execution_error"] = {"type": "Synthetic"}
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(RuntimeError, "incomplete/errored"):
                finalize(manifest, results, root / "out2", require_task_count=8)

    def test_multi_pass_merge_retains_lineage_and_is_training_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_manifest, first_results = self.make_fixture(
                root / "first", stage="exit_bootstrap_never_exit",
                decision_prefix="bootstrap-", downstream_policy_hash="never-exit",
            )
            second_manifest, second_results = self.make_fixture(
                root / "second", stage="exit_second_pass",
                decision_prefix="second-", downstream_policy_hash="learned-exit-v1",
            )
            first = root / "first-finalized"
            second = root / "second-finalized"
            finalize(first_manifest, first_results, first, require_task_count=8)
            finalize(second_manifest, second_results, second, require_task_count=8)
            merged_dir = root / "merged"
            merged = merge_finalized_datasets(
                [first / "decision_dataset_manifest.json", second / "decision_dataset_manifest.json"],
                merged_dir,
            )
            self.assertEqual(merged["pass_ids"], ["exit_bootstrap_never_exit", "exit_second_pass"])
            self.assertEqual(merged["decision_count"], 4)
            self.assertEqual(merged["result_count"], 16)
            self.assertEqual(
                merged["head_dataset_manifest"]["downstream_policy_lineage_counts"],
                {"never-exit": 2, "learned-exit-v1": 2},
            )
            dataset, summary = preflight(
                merged_dir / "decision_dataset_manifest.json", root / "training"
            )
            self.assertEqual(dataset.train_features.shape, (2, 217))
            self.assertEqual(dataset.validation_features.shape, (2, 217))
            self.assertEqual(summary["pass_ids"], ["exit_bootstrap_never_exit", "exit_second_pass"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
