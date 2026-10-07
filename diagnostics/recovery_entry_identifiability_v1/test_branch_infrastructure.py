"""CPU-only tests for paired branch freezing, validation, and aggregation."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import build_branch_manifest as builder
import finalize_paired_branches as finalizer
from branch_common import atomic_json, canonical_hash, file_hash
from run_paired_branches import EXIT_THRESHOLD_LOGIT, ForcedEntry


class BranchInfrastructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def queried_states(self, count: int = 3) -> Path:
        rows = []
        for index in range(count):
            state_file = self.root / f"state_{index}.npz"
            np.savez(state_file, marker=np.asarray(index))
            rows.append({
                "state_id": f"state_{index:03d}", "root_source_id": f"root_{index:03d}",
                "development_cohort": "development", "state_file": str(state_file),
                "state_sha256": file_hash(state_file), "absolute_step": 100 + index,
                "feature": np.zeros(214).tolist(),
                "current_u_flow": np.zeros((2, 2)).tolist(),
                "current_u_safe": np.zeros((2, 2)).tolist(),
                "current_flow_key_data": [index, index + 1],
                "current_flow_realization_id": f"current_{index}",
                "selection_outcome_blind": True, "location_independent": True,
                "terminal_relative_independent": True,
            })
        payload = {
            "schema": "recovery_entry_queried_state_manifest_v1",
            "status": "COMPLETE_FROZEN", "states": rows,
        }
        payload["content_sha256"] = canonical_hash(payload)
        path = self.root / "queried_states.json"
        atomic_json(path, payload)
        return path

    def initial_manifest(self, count: int = 3) -> tuple[Path, dict]:
        path = self.root / "branch_seed_manifest.json"
        manifest = builder.build_initial(self.queried_states(count), path)
        atomic_json(path, manifest)
        return path, manifest

    def test_initial_manifest_has_strict_16_stream_pairs_and_authoritative_exit_override(self) -> None:
        _, manifest = self.initial_manifest()
        self.assertEqual(manifest["decision_count"], 3)
        self.assertEqual(manifest["task_count"], 3 * 16 * 2)
        self.assertEqual(manifest["recovery_policy"]["exit_head"]["threshold_logit"],
                         EXIT_THRESHOLD_LOGIT)
        self.assertTrue(manifest["recovery_policy"]["exit_head"]["checkpoint_local_threshold_ignored"])
        grouped: dict[tuple[str, str], list[dict]] = {}
        for task in manifest["tasks"]:
            grouped.setdefault((task["state_id"], task["future_stream_id"]), []).append(task)
        self.assertEqual(len(grouped), 3 * 16)
        for pair in grouped.values():
            self.assertEqual({row["branch"] for row in pair}, {"N", "R"})
            for key in ("current_flow_key_data", "future_rollout_id", "future_rng_state_hash",
                        "current_flow_realization_id", "state_sha256", "absolute_step"):
                self.assertEqual(pair[0][key], pair[1][key])
        self.assertLessEqual(manifest["task_count"], 12_000)
        self.assertEqual(manifest["adaptive_confirmation"]["future_indices"], [16, 31])

    def test_confirmation_is_separate_one_shot_block_selected_by_frozen_ci_columns(self) -> None:
        initial_path, initial = self.initial_manifest()
        uncertainty = self.root / "label_uncertainty.csv"
        with uncertainty.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "state_id", "escalation_eligible", "delta_q_ci90_width"
            ))
            writer.writeheader()
            writer.writerow({"state_id": "state_000", "escalation_eligible": True,
                             "delta_q_ci90_width": 0.4})
            writer.writerow({"state_id": "state_001", "escalation_eligible": False,
                             "delta_q_ci90_width": 0.8})
            writer.writerow({"state_id": "state_002", "escalation_eligible": True,
                             "delta_q_ci90_width": 0.6})
        finalization = {
            "schema": "recovery_entry_branch_finalization_manifest_v1", "status": "COMPLETE",
            "waves": ["initial"], "all_results_complete": True, "all_pairs_matched": True,
            "source_manifests": [{"content_sha256": initial["content_sha256"]}],
            "outputs": {"label_uncertainty": {
                "path": str(uncertainty.resolve()), "sha256": file_hash(uncertainty),
            }},
        }
        finalization["content_sha256"] = canonical_hash(finalization)
        atomic_json(self.root / "branch_finalization_manifest.json", finalization)
        output = self.root / "confirmation.json"
        confirmation = builder.build_confirmation(initial_path, uncertainty, output)
        self.assertEqual(confirmation["decision_count"], 2)
        self.assertEqual(confirmation["task_count"], 2 * 16 * 2)
        self.assertEqual({row["future_index"] for row in confirmation["tasks"]}, set(range(16, 32)))
        self.assertEqual(confirmation["parent_initial_manifest"]["content_sha256"],
                         initial["content_sha256"])
        self.assertLessEqual(confirmation["budget"]["global_reserved_after_this_wave"], 12_000)

    def test_forced_entry_is_single_use(self) -> None:
        head = ForcedEntry()
        self.assertTrue(head(np.zeros(214)))
        with self.assertRaises(RuntimeError):
            head(np.zeros(214))

    def test_paired_state_summary_uses_success_not_failure_surrogates(self) -> None:
        paired = []
        for index in range(16):
            if index < 4:
                success_n, success_r = 0, 1
            elif index < 6:
                success_n, success_r = 1, 0
            elif index < 11:
                success_n = success_r = 1
            else:
                success_n = success_r = 0
            paired.append({
                "state_id": "s", "root_source_id": "root", "absolute_step": 100,
                "state_sha256": "state", "future_index": index,
                "success_n": success_n, "success_r": success_r,
                "rescue": int(not success_n and success_r),
                "break": int(success_n and not success_r),
                "both_success": int(success_n and success_r),
                "both_failure": int(not success_n and not success_r),
            })
        rows, uncertainty = finalizer.summarize_states(paired, confirmation_closed=False)
        row = rows[0]
        self.assertEqual(row["q_n"], 7 / 16)
        self.assertEqual(row["q_r"], 9 / 16)
        self.assertEqual(row["delta_q"], 2 / 16)
        self.assertEqual(row["rescue"], 4 / 16)
        self.assertEqual(row["break"], 2 / 16)
        self.assertEqual(row["delta_identity_error"], 0.0)
        self.assertTrue(uncertainty[0]["escalation_eligible"])

    @staticmethod
    def result_for(task: dict, manifest: dict) -> dict:
        is_r = task["branch"] == "R"
        # Deterministic mixture: future 0 is a rescue; all other pairs succeed.
        success = is_r or int(task["future_index"]) != 0
        outcome = "success" if success else "deadlock"
        return {
            "schema": "recovery_entry_paired_branch_result_v1", "record_complete": True,
            "execution_error": None, "manifest_content_sha256": manifest["content_sha256"],
            "wave": manifest["wave"], "task_id": task["task_id"],
            "task_index": task["task_index"], "state_id": task["state_id"],
            "root_source_id": task["root_source_id"], "state_sha256": task["state_sha256"],
            "absolute_step": task["absolute_step"], "branch": task["branch"],
            "future_index": task["future_index"], "future_stream_id": task["future_stream_id"],
            "future_rollout_id": task["future_rollout_id"],
            "future_rng_state_hash": task["future_rng_state_hash"],
            "current_flow_realization_id": task["current_flow_realization_id"],
            "current_replay_max_abs": {
                "current_u_flow": 0.0, "current_u_safe": 0.0,
                "current_feature": 0.0, "current_flow_key": 0.0,
            },
            "success": success, "outcome": outcome, "deadlock": not success,
            "timeout": False, "collision": False, "other_failure": False,
            "terminal_global_step": task["absolute_step"] + 1,
            "remaining_physical_steps": 1, "flow_sample_count": 1,
            "physical_transition_count": 1,
            "entry_step": task["absolute_step"] if is_r else None,
            "exit_step": None, "recovery_transitions": 1 if is_r else 0,
            "returned_to_safety": False, "entry_query_count": 1 if is_r else 0,
            "exit_query_count": 0, "eta_query_count": 1 if is_r else 0,
            "eta_latched_hash": "eta" if is_r else None,
            "first_step_exit_queried": False,
            "invalid_actions": 0, "nan_inf_events": 0, "projection_failures": 0,
        }

    def test_complete_finalization_checks_every_task_and_writes_required_tables(self) -> None:
        manifest_path, manifest = self.initial_manifest(count=1)
        result_dir = self.root / "results"
        result_dir.mkdir()
        for task in manifest["tasks"]:
            atomic_json(result_dir / f"task_{task['task_index']:05d}.json",
                        self.result_for(task, manifest))
        output = self.root / "finalized"
        summary = finalizer.finalize([manifest_path], [result_dir], output)
        self.assertEqual(summary["branch_continuation_count"], 32)
        self.assertEqual(summary["matched_pair_count"], 16)
        self.assertEqual(summary["queried_state_count"], 1)
        self.assertTrue(summary["all_pairs_matched"])
        for name in ("paired_branch_outcomes.csv", "statewise_recovery_advantage.csv",
                     "label_uncertainty.csv", "advantage_distribution.json"):
            self.assertTrue((output / name).is_file())
        distribution = json.loads((output / "advantage_distribution.json").read_text())
        self.assertEqual(distribution["trivial_and_hindsight_values"]["always_safety"], 15 / 16)
        self.assertEqual(distribution["trivial_and_hindsight_values"]["always_recovery"], 1.0)

    def test_safety_result_with_recovery_memory_is_rejected(self) -> None:
        _, manifest = self.initial_manifest(count=1)
        task = next(row for row in manifest["tasks"] if row["branch"] == "N")
        result = self.result_for(task, manifest)
        result["recovery_transitions"] = 1
        with self.assertRaises(RuntimeError):
            finalizer.validate_result(result, task, manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
