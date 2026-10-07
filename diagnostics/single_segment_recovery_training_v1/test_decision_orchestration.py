"""CPU tests for split-safe head orchestration and full-loop metrics."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np

from decision_orchestration import (
    analyze_full_loop_policy,
    build_head_dataset,
    calibration_break_budget_audit,
    choose_validation_threshold,
    materialize_threshold_variant,
    run_training_grid,
    select_policy_by_full_loop_validation,
    target_lineage_id,
    validation_threshold_table,
)


def make_decision(
    decision_id: str,
    split: str,
    source: str,
    policy_hash: str,
    pass_id: str,
    *,
    dimension: int = 214,
):
    return {
        "decision_id": decision_id,
        "pass_id": pass_id,
        "downstream_policy_hash": policy_hash,
        "root_source_id": source,
        "split": split,
        "state_hash": f"state-{decision_id}",
        "absolute_step": 100,
        "mode": "SAFETY_BEFORE",
        "feature": np.linspace(0, 1, dimension).tolist(),
    }


def make_branches(decision, *, futures=4, rescue=True):
    rows = []
    for future in range(futures):
        success0, success1 = ((0, 1) if rescue else (1, 0))
        common = {
            key: decision[key]
            for key in (
                "decision_id", "pass_id", "downstream_policy_hash", "root_source_id",
                "split", "state_hash", "absolute_step", "mode",
            )
        }
        common.update(
            {
                "future_stream_id": f"future-{future}",
                "future_rng_state_hash": f"rng-{future}",
                "current_flow_realization_id": "current-flow",
                "remaining_jdef": 0.2,
            }
        )
        rows.extend(
            [
                {**common, "action": 0, "success": success0},
                {**common, "action": 1, "success": success1, "remaining_jdef": 0.1},
            ]
        )
    return rows


class DatasetTests(unittest.TestCase):
    def test_policy_pass_lineage_is_retained(self):
        decisions = [
            make_decision("same", "train", "train-root", "old-policy", "bootstrap"),
            make_decision("same", "train", "train-root", "new-policy", "iteration1"),
            make_decision("v", "validation", "validation-root", "new-policy", "iteration1"),
        ]
        branches = [row for decision in decisions for row in make_branches(decision)]
        dataset = build_head_dataset(
            decisions, branches, head_kind="entry_eta", expected_futures_per_input=4
        )
        self.assertEqual(dataset.train_features.shape, (2, 214))
        self.assertEqual(dataset.validation_features.shape, (1, 214))
        self.assertEqual(len(dataset.manifest["downstream_policy_lineage_counts"]), 2)
        self.assertTrue(dataset.manifest["lineage_retained_not_merged"])
        self.assertNotEqual(target_lineage_id(decisions[0]), target_lineage_id(decisions[1]))

    def test_exit_eta_appends_latched_eta(self):
        train = make_decision("t", "train", "t-root", "p", "x")
        validation = make_decision("v", "validation", "v-root", "p", "x")
        train["eta_latched"] = [0.1, 0.2, 0.3]
        validation["eta_latched"] = [0.4, 0.5, 0.6]
        decisions = [train, validation]
        branches = [row for decision in decisions for row in make_branches(decision)]
        dataset = build_head_dataset(
            decisions, branches, head_kind="exit_eta", expected_futures_per_input=4
        )
        self.assertEqual(dataset.train_features.shape, (1, 217))
        np.testing.assert_allclose(dataset.train_features[0, -3:], [0.1, 0.2, 0.3])

    def test_split_leak_and_incomplete_pairs_fail_closed(self):
        decision = make_decision("d", "train", "shared", "p", "x")
        validation = make_decision("v", "validation", "shared", "p", "x")
        rows = make_branches(decision) + make_branches(validation)
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            build_head_dataset(
                [decision, validation], rows, head_kind="entry_g", expected_futures_per_input=4
            )
        valid = make_decision("v", "validation", "other", "p", "x")
        incomplete = make_branches(decision) + make_branches(valid)
        incomplete.pop()
        with self.assertRaisesRegex(ValueError, "missing matched"):
            build_head_dataset(
                [decision, valid], incomplete, head_kind="entry_g", expected_futures_per_input=4
            )

    def test_calibration_and_final_labels_are_rejected(self):
        train = make_decision("t", "train", "t", "p", "x")
        final = make_decision("f", "final_test", "f", "p", "x")
        branches = make_branches(train) + make_branches(final)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            build_head_dataset(
                [train, final], branches, head_kind="entry_eta", expected_futures_per_input=4
            )


class TrainingTests(unittest.TestCase):
    def test_validation_threshold_prefers_rescue_without_break(self):
        decisions = [
            make_decision("t", "train", "tr", "p", "x"),
            make_decision("v1", "validation", "v1r", "p", "x"),
            make_decision("v2", "validation", "v2r", "p", "x"),
        ]
        branches = []
        branches += make_branches(decisions[0], rescue=True)
        branches += make_branches(decisions[1], rescue=True)
        branches += make_branches(decisions[2], rescue=False)
        dataset = build_head_dataset(
            decisions, branches, head_kind="entry_eta", expected_futures_per_input=4
        )
        table = validation_threshold_table(
            np.asarray([0.9, 0.1]), dataset.validation_evidence, lambda_break=3.0
        )
        chosen = choose_validation_threshold(table)
        self.assertEqual([row["threshold"] for row in table], [0.25, 0.5, 0.75])
        self.assertEqual(chosen["event_count"], 1)
        self.assertGreater(chosen["estimated_net_success"], 0)

    def test_tiny_training_grid_writes_hashed_manifest(self):
        decisions = [
            make_decision("t1", "train", "tr1", "p", "x"),
            make_decision("t2", "train", "tr2", "p", "x"),
            make_decision("v1", "validation", "vr1", "p", "x"),
            make_decision("v2", "validation", "vr2", "p", "x"),
        ]
        branches = []
        for index, decision in enumerate(decisions):
            branches += make_branches(decision, rescue=index % 2 == 0)
        dataset = build_head_dataset(
            decisions, branches, head_kind="entry_eta", expected_futures_per_input=4
        )
        with tempfile.TemporaryDirectory() as directory:
            result = run_training_grid(
                dataset, directory, seeds=(17,), lambda_break_family=(1.0,), epochs=2,
                allow_test_configuration=True,
            )
            self.assertEqual(result["candidate_count"], 1)
            provisional = result["provisional_branch_evidence_choice"]
            self.assertTrue(Path(provisional["checkpoint_path"]).exists())
            self.assertEqual(len(provisional["policy_sha256"]), 64)
            self.assertTrue(result["requires_full_episode_validation"])
            self.assertIsNone(result["final_selected_policy"])
            # Deployment loader must consume the exact checkpoint contract.
            from run_paired_branches import FrozenHead

            frozen = FrozenHead(Path(provisional["checkpoint_path"]))
            self.assertEqual(frozen.mean.shape, (214,))
            self.assertTrue(np.isfinite(frozen.threshold))

            variant_path = Path(directory) / "threshold_075.npz"
            variant = materialize_threshold_variant(
                provisional["checkpoint_path"], variant_path,
                threshold_probability=0.75,
            )
            variant_head = FrozenHead(variant_path)
            self.assertAlmostEqual(variant_head.threshold, np.log(3.0))
            self.assertEqual(len(variant["checkpoint_sha256"]), 64)
            self.assertFalse(variant["weights_changed"])
            with self.assertRaises(FileExistsError):
                materialize_threshold_variant(
                    provisional["checkpoint_path"], variant_path,
                    threshold_probability=0.75,
                )


class FullLoopTests(unittest.TestCase):
    def rows(self, split="validation"):
        outcomes = [
            ("success", "success", None, None, 0),
            ("timeout", "success", 100, 120, 20),
            ("success", "timeout", 200, 202, 2),
            ("strict_deadlock", "strict_deadlock", None, None, 0),
        ]
        rows = []
        for index, (safety, learned, entry, exit_step, recovery_count) in enumerate(outcomes):
            rows.append(
                {
                    "root_source_id": f"r{index}", "split": split, "policy_hash": "policy",
                    "record_complete": True, "execution_error": None,
                    "safety_outcome": safety, "learned_outcome": learned,
                    "entry_step": entry, "exit_step": exit_step,
                    "recovery_transition_count": recovery_count,
                    "recovery_segment_count": int(entry is not None),
                    "terminal_step": 300, "jdef": 0.1 * index,
                    "agent_collision": 0, "wall_collision": 0, "invalid_action": 0,
                    "nan_inf": 0, "projection_solver_failure": 0,
                    "entry_threshold_probability": 0.25,
                    "exit_threshold_probability": 0.75,
                }
            )
        return rows

    def test_source_level_rescue_break_and_segment_metrics(self):
        metrics = analyze_full_loop_policy(
            self.rows(), split="validation", policy_hash="policy"
        )
        self.assertEqual(metrics["source_episode_count"], 4)
        self.assertEqual((metrics["both_success"], metrics["rescue"], metrics["break"], metrics["both_fail"]), (1, 1, 1, 1))
        self.assertEqual(metrics["timeout_rescues"], 1)
        self.assertEqual(metrics["strict_deadlock_rescues"], 0)
        self.assertEqual(metrics["entry_threshold_probability"], 0.25)
        self.assertEqual(metrics["exit_threshold_probability"], 0.75)
        self.assertEqual(metrics["segment_statistics"]["successful_finite_segment_then_safety_count"], 1)
        self.assertEqual(metrics["segment_statistics"]["returned_to_safety_then_failed_count"], 1)
        self.assertTrue(metrics["hard_safety"]["intact"])

    def test_final_test_is_locked_and_calibration_not_certification(self):
        with self.assertRaisesRegex(ValueError, "locked"):
            analyze_full_loop_policy(
                self.rows("final_test"), split="final_test", policy_hash="policy"
            )
        calibration = analyze_full_loop_policy(
            self.rows("calibration"), split="calibration", policy_hash="policy"
        )
        audit = calibration_break_budget_audit(calibration)
        self.assertFalse(audit["certification_claimed"])
        self.assertTrue(audit["insufficient_support"])

    def test_final_selection_uses_full_loop_validation(self):
        first_rows = self.rows()
        for row in first_rows:
            row["policy_hash"] = "a"
        first = analyze_full_loop_policy(first_rows, split="validation", policy_hash="a")
        second_rows = self.rows()
        second_rows[2]["learned_outcome"] = "success"
        for row in second_rows:
            row["policy_hash"] = "b"
        second = analyze_full_loop_policy(second_rows, split="validation", policy_hash="b")
        selected = select_policy_by_full_loop_validation([first, second])
        self.assertEqual(selected["selected_policy_hash"], "b")
        self.assertEqual(selected["selection_basis"], "complete source-level full-episode outcomes")
        self.assertFalse(selected["break_budget_used_as_selection_filter"])

        # Exact-Q ties follow the frozen global ordering, not local head loss
        # or failure-type rescue counts.
        low_break = dict(first)
        low_break.update({
            "policy_hash": "low-break", "q_learned": 0.5, "break_rate": 0.01,
            "entry_threshold_probability": 0.25,
            "exit_threshold_probability": 0.75,
        })
        high_break = dict(first)
        high_break.update({
            "policy_hash": "high-break", "q_learned": 0.5, "break_rate": 0.2,
            "entry_threshold_probability": 0.75,
            "exit_threshold_probability": 0.25,
        })
        tied = select_policy_by_full_loop_validation([high_break, low_break])
        self.assertEqual(tied["selected_policy_hash"], "low-break")
        calibration = analyze_full_loop_policy(
            self.rows("calibration"), split="calibration", policy_hash="policy"
        )
        with self.assertRaisesRegex(ValueError, "validation only"):
            select_policy_by_full_loop_validation([calibration])


if __name__ == "__main__":
    unittest.main(verbosity=2)
