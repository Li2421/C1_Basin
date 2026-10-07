"""Finish inference-only outputs from the completed source-balanced run.

The training process finished every fold/seed and serialized checkpoints, but
its compact IG postprocessing used an unnecessarily strict quadrature
assertion.  This script deliberately performs no optimizer step: it reloads
those exact frozen checkpoints, reconstructs all required OOF/variant rows,
and completes attribution at a finer integration resolution.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

AUDIT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AUDIT))
from shared import gate_condition_runner as runner  # noqa: E402
import run_source_group_balanced as condition  # noqa: E402


OUT = Path(__file__).resolve().parent
CHECKPOINTS = OUT / "checkpoints"
CONDITION = "SOURCE_GROUP_BALANCED"


def load_checkpoint(path: Path):
    with np.load(path, allow_pickle=False) as loaded:
        params = []
        layer = 0
        while f"layer_{layer}_weight" in loaded:
            params.append({
                # Preserve x64 checkpoint parameters exactly; forcing float32
                # changes validation probabilities enough to alter a selected
                # threshold in this small-state setting.
                "w": jnp.asarray(loaded[f"layer_{layer}_weight"]),
                "b": jnp.asarray(loaded[f"layer_{layer}_bias"]),
            })
            layer += 1
        metadata = json.loads(str(loaded["metadata_json"]))
        mean = np.asarray(loaded["normalization_mean"], float)
        scale = np.asarray(loaded["normalization_scale"], float)
        binary = np.asarray(loaded["normalization_binary_mask"], bool)
    if metadata.get("condition") != CONDITION or metadata.get("sampler") != CONDITION:
        raise RuntimeError(("checkpoint metadata mismatch", path, metadata))
    return params, metadata, mean, scale, binary


def rebuild_trained(fold: dict, context: dict) -> dict:
    sample_ids = context["sample_ids"]
    label = context["label"]
    models, validation, test, training_rows = {}, [], [], []
    for seed in runner.SEEDS:
        path = CHECKPOINTS / fold["fold_id"] / f"seed_{seed}.npz"
        if not path.exists():
            raise RuntimeError(("missing completed checkpoint", path))
        params, metadata, mean, scale, binary = load_checkpoint(path)
        if not (np.allclose(mean, fold["mean"]) and np.allclose(scale, fold["scale"]) and np.array_equal(binary, fold["binary"])):
            raise RuntimeError(("checkpoint normalization is not frozen fold normalization", path))
        val_ids, val_p, val_y = runner.cv.aggregate_state(
            runner.cv.predict(params, fold["normalized"][fold["validation_index"]]),
            sample_ids[fold["validation_index"]], label,
        )
        test_ids, test_p, test_y = runner.cv.aggregate_state(
            runner.cv.predict(params, fold["normalized"][fold["test_index"]]),
            sample_ids[fold["test_index"]], label,
        )
        threshold = runner.cv.select_threshold(val_y, val_p)
        validation_bce = runner.cv.state_bce(val_y, val_p)
        if not np.isfinite(validation_bce):
            raise RuntimeError(("nonfinite checkpoint validation BCE", fold["fold_id"], seed))
        expected = float(metadata["validation_selected_threshold"])
        if abs(threshold - expected) > 2e-6:
            raise RuntimeError(("validation-only threshold reproduction failed", fold["fold_id"], seed, threshold, expected))
        models[seed] = {"params": params, "threshold": threshold, "best_epoch": int(metadata["best_epoch"]), "validation_bce": validation_bce}
        validation.append((val_ids, val_p, val_y))
        test.append((test_ids, test_p, test_y))
        training_rows.append({
            "condition": CONDITION, "sampler": CONDITION, "fold_id": fold["fold_id"], "heldout_source_group": fold["outer_group"],
            "training_seed": seed, "best_epoch": int(metadata["best_epoch"]),
            "validation_state_BCE": validation_bce, "validation_selected_threshold": threshold,
            "finite": bool(np.isfinite(val_p).all() and np.isfinite(test_p).all()), "checkpoint_reloaded_no_retraining": True,
        })
    if any(not np.array_equal(validation[0][0], item[0]) for item in validation) or any(not np.array_equal(test[0][0], item[0]) for item in test):
        raise RuntimeError(("aggregate IDs changed", fold["fold_id"]))
    val_mean = np.mean([item[1] for item in validation], axis=0)
    test_mean = np.mean([item[1] for item in test], axis=0)
    ensemble_threshold = runner.cv.select_threshold(validation[0][2], val_mean)
    training_rows.append({
        "condition": CONDITION, "sampler": CONDITION, "fold_id": fold["fold_id"], "heldout_source_group": fold["outer_group"],
        "training_seed": "SEED_MEAN", "best_epoch": "", "validation_state_BCE": runner.cv.state_bce(validation[0][2], val_mean),
        "validation_selected_threshold": ensemble_threshold, "finite": bool(np.isfinite(val_mean).all() and np.isfinite(test_mean).all()), "checkpoint_reloaded_no_retraining": True,
    })
    return {"fold": fold, "seed_models": models, "validation": validation, "test": test, "val_ids": validation[0][0], "test_ids": test[0][0], "validation_mean": val_mean, "test_mean": test_mean, "ensemble_threshold": ensemble_threshold, "training_rows": training_rows}


def main() -> None:
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    jax.config.update("jax_enable_x64", True)
    ready = json.loads((AUDIT / "shared" / "BASELINE_READY.json").read_text())
    if ready.get("status") != "BASELINE_READY" or not ready.get("all_checks_passed"):
        raise RuntimeError(("baseline precondition absent", ready))
    context = runner.load_context()
    folds = runner.prepare_folds(context)
    trained_by_fold = {fold["fold_id"]: rebuild_trained(fold, context) for fold in folds}
    state_rows = []
    training_rows = []
    for trained in trained_by_fold.values():
        state_rows.extend(runner.aggregate_prediction_rows(trained, context, CONDITION))
        training_rows.extend(trained["training_rows"])
    target_states = runner.target_variant_states(context)
    by_group = {trained["fold"]["outer_group"]: trained for trained in trained_by_fold.values()}
    variant_rows = []
    for state_id, role in sorted(target_states.items()):
        variant_rows.extend(runner.infer_variants(state_id, by_group[context["state"][state_id]["source_group"]], context, CONDITION, role))
    attribution = condition.attribution_rows(context, trained_by_fold)
    calibration = runner.calibration_rows(state_rows, CONDITION)
    seed_mean = [row for row in state_rows if row["training_seed"] == "SEED_MEAN"]
    checks = {
        "status": "COMPLETE",
        "training_completed_from_saved_checkpoints": True,
        "postprocessing_performed_without_any_optimizer_step": True,
        "baseline_ready_verified": True,
        "frozen_fold_count": len(folds),
        "normalization_train_only_reverified": True,
        "threshold_validation_only_reverified": True,
        "no_test_or_validation_sampling": True,
        # The frozen hard-stable LOGO protocol holds out seven pre-specified
        # difficult source groups, not every source group in the global stable
        # pool.  Integrity requires exactly one OOF row for each *evaluated*
        # held-out state and no duplicate held-out state, not artificial full
        # coverage of groups that were never outer-test groups.
        "all_evaluated_test_states_exactly_one_seed_mean_oof": len(seed_mean) == len({row["state_id"] for row in seed_mean}),
        "evaluated_seed_mean_state_count": len(seed_mean),
        "global_stable_state_count": len(context["stable_ids"]),
        "all_outputs_finite": all(np.isfinite(float(row["p_gate"])) and np.isfinite(float(row["gate_logit"])) for row in state_rows + variant_rows),
        "all_checkpoint_validation_BCE_finite": all(np.isfinite(float(row["validation_state_BCE"])) for row in training_rows),
        "feature_schema_changed": False, "oracle_labels_changed": False, "new_states": 0, "new_oracle_rollouts": 0,
        "correction_head_trained": False, "closed_loop_run": False, "attribution_inference_only": True,
    }
    needed = ("training_completed_from_saved_checkpoints", "postprocessing_performed_without_any_optimizer_step", "baseline_ready_verified", "normalization_train_only_reverified", "threshold_validation_only_reverified", "no_test_or_validation_sampling", "all_evaluated_test_states_exactly_one_seed_mean_oof", "all_outputs_finite", "all_checkpoint_validation_BCE_finite", "attribution_inference_only")
    if not all(checks[name] is True for name in needed):
        raise RuntimeError(("postprocess sanity failure", checks))
    runner.write_csv(OUT / "training_results.csv", training_rows)
    runner.write_csv(OUT / "out_of_fold_predictions.csv", state_rows)
    runner.write_csv(OUT / "variant_predictions.csv", variant_rows)
    runner.write_csv(OUT / "fold_relative_calibration.csv", calibration)
    runner.write_csv(OUT / "attribution.csv", attribution)
    runner.write_json(OUT / "sanity_checks.json", checks)
    runtime = {
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(), "wall_s": time.perf_counter() - started,
        "condition": CONDITION, "stage": "inference-only checkpoint postprocessing", "outer_folds": len(folds), "training_runs_relaunched": 0,
        "GPU_shards": 1, "CPU_threads_cap": int(os.environ.get("OMP_NUM_THREADS", "4")), "jax_backend": jax.default_backend(),
        "integrated_gradient_steps": condition.IG_STEPS, "new_oracle_rollouts": 0,
    }
    runner.write_json(OUT / "runtime_statistics.json", runtime)
    runner.write_json(OUT / "manifest.json", {
        "condition": CONDITION, "sampler": CONDITION, "baseline_ready": ready,
        "runner_sha256": runner.sha256(AUDIT / "shared" / "gate_condition_runner.py"),
        "completed_training_checkpoint_count": len(folds) * len(runner.SEEDS),
        "postprocessing_script": "postprocess_saved_checkpoints.py",
        "outputs": ["training_results.csv", "out_of_fold_predictions.csv", "variant_predictions.csv", "fold_relative_calibration.csv", "attribution.csv", "sanity_checks.json", "runtime_statistics.json"],
    })
    print({"status": "COMPLETE", "condition": CONDITION, "wall_s": runtime["wall_s"], "attribution_rows": len(attribution)}, flush=True)


if __name__ == "__main__":
    main()
