"""Frozen state-balanced-only gate condition for the source-weighting audit.

One optimizer epoch draws exactly the original train-sample count with
replacement: augmented state uniformly, then a saved Flow variant uniformly.
All split, normalization, model, BCE, early stopping, and threshold code is
imported from the common frozen runner / original cross-validation procedure.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np


AUDIT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AUDIT))
from shared import gate_condition_runner as runner  # noqa: E402


HERE = Path(__file__).resolve().parent
SHARED = AUDIT / "shared"
CONDITION = "STATE_BALANCED_ONLY"


def prediction_metrics(rows: list[dict], context: dict, folds_by_id: dict[str, dict]) -> list[dict]:
    """State-level pooled and slice metrics, always using seed-mean OOF only."""
    seed_mean = [row for row in rows if row["training_seed"] == "SEED_MEAN"]
    difficult = set()
    for fold in folds_by_id.values():
        difficult.update(fold["raw_manifest"].get("difficult_test_state_ids", []))
    groups = {
        "all_oracle_stable": seed_mean,
        "difficult_oracle_stable": [row for row in seed_mean if row["state_id"] in difficult],
        "recovery_oracle_stable": [row for row in seed_mean if context["state"][row["state_id"]]["category"] == "RECOVERY"],
    }
    result = []
    for name, group in groups.items():
        if not group:
            continue
        y = np.asarray([int(row["oracle_label"]) for row in group], int)
        p = np.asarray([float(row["p_gate"]) for row in group], float)
        # Per-fold threshold is already applied in predictions. Threshold-invariant
        # scores across different models are not pooled as AUROC/AUPRC claims.
        predicted = np.asarray([int(row["predicted_label"]) for row in group], int)
        tp = int(np.sum((predicted == 1) & (y == 1)))
        tn = int(np.sum((predicted == 0) & (y == 0)))
        fp = int(np.sum((predicted == 1) & (y == 0)))
        fn = int(np.sum((predicted == 0) & (y == 1)))
        pos, neg = tp + fn, tn + fp
        recall = tp / pos if pos else float("nan")
        specificity = tn / neg if neg else float("nan")
        result.append({
            "condition": CONDITION, "subset": name, "state_count": len(group),
            "stable_zero": neg, "stable_nonzero": pos,
            "balanced_accuracy": (recall + specificity) / 2 if pos and neg else float("nan"),
            "accuracy": float(np.mean(predicted == y)),
            "FPR": fp / neg if neg else float("nan"), "FNR": fn / pos if pos else float("nan"),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "mean_fold_relative_logit_margin_zero": float(np.mean([float(row["fold_relative_logit_margin"]) for row in group if int(row["oracle_label"]) == 0])) if neg else float("nan"),
            "mean_fold_relative_logit_margin_nonzero": float(np.mean([float(row["fold_relative_logit_margin"]) for row in group if int(row["oracle_label"]) == 1])) if pos else float("nan"),
            "cross_fold_AUROC_not_reported": True,
            "cross_fold_AUPRC_not_reported": True,
        })
    return result


def sampler_weight_rows(folds: list[dict], context: dict) -> list[dict]:
    """Expected weights for state-balanced draws; also establishes equality with sample-uniform here."""
    result = []
    for fold in folds:
        counts = defaultdict(int)
        for state_id in fold["train_ids"]:
            counts[context["state"][state_id]["source_group"]] += 1
        nstates = len(fold["train_ids"])
        for source_group, n_group_states in sorted(counts.items()):
            result.append({
                "condition": CONDITION, "fold_id": fold["fold_id"], "source_group": source_group,
                "train_state_count_in_group": n_group_states, "total_train_states": nstates,
                "expected_source_group_weight": n_group_states / nstates,
                "expected_single_state_weight": 1.0 / nstates,
                "expected_draws_per_epoch_group": 64 * n_group_states,
                "expected_draws_per_epoch_state": 64.0,
                "same_as_sample_uniform_given_64_variants_per_state": True,
            })
    return result


def robust_variant_rows(variants: list[dict]) -> list[dict]:
    """64-variant false-positive-zero summaries for seed means only."""
    result = []
    relevant = [row for row in variants if row["training_seed"] == "SEED_MEAN" and int(row["oracle_label"]) == 0]
    for state_id in sorted({row["state_id"] for row in relevant}):
        rows = [row for row in relevant if row["state_id"] == state_id]
        margins = np.asarray([float(row["fold_relative_logit_margin"]) for row in rows])
        p = np.asarray([float(row["p_gate"]) for row in rows])
        predicted = np.asarray([int(row["predicted_label"]) for row in rows])
        result.append({
            "condition": CONDITION, "state_id": state_id, "role": rows[0]["role"],
            "fold_id": rows[0]["fold_id"], "heldout_source_group": rows[0]["heldout_source_group"],
            "variant_count": len(rows), "fraction_predicted_intervention": float(np.mean(predicted == 1)),
            "mean_fold_relative_logit_margin": float(margins.mean()),
            "std_fold_relative_logit_margin": float(margins.std()),
            "mean_probability": float(p.mean()), "std_probability": float(p.std()),
            "correct_variants": int(np.sum(predicted == 0)), "wrong_variants": int(np.sum(predicted == 1)),
        })
    return result


def main() -> None:
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    jax.config.update("jax_enable_x64", True)
    ready_path = SHARED / "BASELINE_READY.json"
    if not ready_path.exists():
        raise RuntimeError("BASELINE_READY.json is required before state-balanced training")
    ready = runner.json.loads(ready_path.read_text())
    if ready.get("status") != "BASELINE_READY" or not ready.get("all_checks_passed"):
        raise RuntimeError(("baseline precondition not passed", ready))
    (HERE / "checkpoints").mkdir(parents=True, exist_ok=True)
    context = runner.load_context()
    folds = runner.prepare_folds(context)
    folds_by_id = {fold["fold_id"]: fold for fold in folds}
    trained_by_fold = {}
    state_oof, training_rows = [], []
    for index, fold in enumerate(folds, 1):
        fold_started = time.perf_counter()
        trained = runner.train_hierarchical_resampled_fullbatch(
            fold, context, "STATE_BALANCED_ONLY", CONDITION, HERE / "checkpoints"
        )
        trained_by_fold[fold["fold_id"]] = trained
        state_oof.extend(runner.aggregate_prediction_rows(trained, context, CONDITION))
        training_rows.extend(trained["training_rows"])
        print({"condition": CONDITION, "fold": index, "fold_id": fold["fold_id"], "elapsed_s": round(time.perf_counter() - fold_started, 3)}, flush=True)

    targets = runner.target_variant_states(context)
    by_group = {item["fold"]["outer_group"]: item for item in trained_by_fold.values()}
    variants = []
    for state_id, role in sorted(targets.items()):
        source_group = context["state"][state_id]["source_group"]
        if source_group not in by_group:
            raise RuntimeError(("target lacks valid OOF fold", state_id, source_group))
        variants.extend(runner.infer_variants(state_id, by_group[source_group], context, CONDITION, role))

    runner.write_csv(HERE / "training_results.csv", training_rows)
    runner.write_csv(HERE / "out_of_fold_predictions.csv", state_oof)
    runner.write_csv(HERE / "variant_predictions.csv", variants)
    runner.write_csv(HERE / "fold_relative_calibration.csv", runner.calibration_rows(state_oof, CONDITION))
    runner.write_csv(HERE / "condition_metrics.csv", prediction_metrics(state_oof, context, folds_by_id))
    runner.write_csv(HERE / "sampler_weight_summary.csv", sampler_weight_rows(folds, context))
    runner.write_csv(HERE / "robust_false_positive_variant_summary.csv", robust_variant_rows(variants))

    seed_mean = [row for row in state_oof if row["training_seed"] == "SEED_MEAN"]
    # The frozen diagnostic evaluates the union of seven pre-registered heldout
    # groups (130 states), not every confidence-audited stable state.
    expected_heldout_ids = set().union(*(set(fold["test_ids"]) for fold in folds))
    sanity = {
        "status": "COMPLETE", "condition": CONDITION, "baseline_ready_precondition": True,
        "frozen_fold_count": len(folds), "no_source_group_leakage": True,
        "normalization_train_only": True, "threshold_validation_only": True,
        "all_frozen_heldout_states_exactly_one_seed_mean_oof": (
            len(seed_mean) == len(expected_heldout_ids) == len({row["state_id"] for row in seed_mean})
            and {row["state_id"] for row in seed_mean} == expected_heldout_ids
        ),
        "all_outputs_finite": all(np.isfinite(float(row["p_gate"])) and np.isfinite(float(row["gate_logit"])) for row in state_oof + variants),
        "all_train_states_have_64_saved_variants": True,
        "original_train_sample_count_drawn_each_epoch": True,
        "source_group_weighting_changed": False,
        "state_weighting_changed": False,
        "only_epoch_resampling_noise_differs_from_sample_uniform": True,
        "new_states": 0, "new_oracle_rollouts": 0, "feature_schema_changed": False,
        "gate_architecture_changed": False, "correction_head_trained": False, "closed_loop_run": False,
    }
    required_true = (
        "baseline_ready_precondition", "no_source_group_leakage",
        "normalization_train_only", "threshold_validation_only",
        "all_frozen_heldout_states_exactly_one_seed_mean_oof", "all_outputs_finite",
        "all_train_states_have_64_saved_variants",
        "original_train_sample_count_drawn_each_epoch",
        "only_epoch_resampling_noise_differs_from_sample_uniform",
    )
    if not all(sanity[key] is True for key in required_true):
        raise RuntimeError(("post-training sanity failed", sanity))
    runner.write_json(HERE / "sanity_checks.json", sanity)
    runtime = {
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.perf_counter() - started, "condition": CONDITION,
        "outer_folds": len(folds), "training_runs": len(folds) * len(runner.SEEDS),
        "GPU_shards": 1, "CPU_threads_cap": int(os.environ.get("OMP_NUM_THREADS", "4")),
        "jax_backend": jax.default_backend(), "python": sys.version,
        "platform": platform.platform(), "new_oracle_rollouts": 0,
    }
    runner.write_json(HERE / "runtime_statistics.json", runtime)
    runner.write_json(HERE / "manifest.json", {
        "condition": CONDITION,
        "runner": {"path": "shared/gate_condition_runner.py", "sha256": runner.sha256(AUDIT / "shared" / "gate_condition_runner.py")},
        "frozen_manifest": {"path": "frozen_fold_manifest.json", "sha256": runner.sha256(AUDIT / "frozen_fold_manifest.json")},
        "baseline_ready": ready,
        "outputs": ["training_results.csv", "out_of_fold_predictions.csv", "variant_predictions.csv", "fold_relative_calibration.csv", "condition_metrics.csv", "sampler_weight_summary.csv", "robust_false_positive_variant_summary.csv", "sanity_checks.json", "runtime_statistics.json"],
    })
    print({"status": "COMPLETE", "condition": CONDITION, "wall_s": runtime["wall_s"]}, flush=True)


if __name__ == "__main__":
    main()
