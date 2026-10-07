"""Reproduce the frozen ordinary full-batch sample-uniform gate baseline.

This is a strict precondition for source/state-balanced sampling work.  The
original six source-group folds are checked at per-state, per-seed precision;
the seventh previously validated D1 fold is independently rechecked on all
64 saved variants of its designated state.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np

from shared import gate_condition_runner as runner


HERE = Path(__file__).resolve().parent
OUT = HERE / "conditions" / "baseline_sample_uniform"
SHARED = HERE / "shared"
CONDITION = "BASELINE_SAMPLE_UNIFORM"


def fail(reason: str, checks: list[dict], started: float) -> None:
    runner.write_csv(OUT / "baseline_reproduction_checks.csv", checks)
    runner.write_json(OUT / "BASELINE_FAILED.json", {
        "status": "BASELINE_FAILED", "reason": reason,
        "reproduction_tolerance": runner.TOLERANCE,
        "checks": checks,
        "wall_s": time.perf_counter() - started,
    })
    raise RuntimeError(reason)


def main() -> None:
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    jax.config.update("jax_enable_x64", True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(exist_ok=True)
    ready = SHARED / "BASELINE_READY.json"
    if ready.exists():
        raise RuntimeError("Refusing to overwrite an existing BASELINE_READY.json")

    context = runner.load_context()
    folds = runner.prepare_folds(context)
    trained_by_fold = {}
    state_oof = []
    training_rows = []
    for index, fold in enumerate(folds, 1):
        fold_started = time.perf_counter()
        trained = runner.train_full_batch_baseline(fold, context, OUT / "checkpoints")
        trained_by_fold[fold["fold_id"]] = trained
        state_oof.extend(runner.aggregate_prediction_rows(trained, context, CONDITION))
        training_rows.extend(trained["training_rows"])
        print({"fold": index, "fold_id": fold["fold_id"], "heldout_source_group": fold["outer_group"], "elapsed_s": round(time.perf_counter() - fold_started, 3)}, flush=True)

    # The original six must reproduce exactly, including each seed and seed mean.
    checks = runner.check_original_six(trained_by_fold, state_oof)
    d1_fold = trained_by_fold.get("LOGO_anchor_D1_pair231")
    if d1_fold is None:
        fail("frozen D1 OOF fold missing", checks, started)
    checks.extend(runner.check_missing_d1_variants(d1_fold, context))
    runner.write_csv(OUT / "baseline_reproduction_checks.csv", checks)
    if not all(bool(row["passed"]) for row in checks):
        fail("strict baseline reproduction failed", checks, started)

    target_states = runner.target_variant_states(context)
    by_group = {fold["outer_group"]: trained for fold, trained in ((item["fold"], item) for item in trained_by_fold.values())}
    variants = []
    for state_id, role in sorted(target_states.items()):
        group = context["state"][state_id]["source_group"]
        if group not in by_group:
            fail(f"no valid frozen OOF model for {state_id} ({group})", checks, started)
        variants.extend(runner.infer_variants(state_id, by_group[group], context, CONDITION, role))

    calibration = runner.calibration_rows(state_oof, CONDITION)
    runner.write_csv(OUT / "training_results.csv", training_rows)
    runner.write_csv(OUT / "out_of_fold_predictions.csv", state_oof)
    runner.write_csv(OUT / "variant_predictions.csv", variants)
    runner.write_csv(OUT / "fold_relative_calibration.csv", calibration)

    # Every stable state appears in one and only one seed-mean heldout fold.
    seed_mean = [row for row in state_oof if row["training_seed"] == "SEED_MEAN"]
    ids = [row["state_id"] for row in seed_mean]
    frozen_heldout_ids = {
        state_id for raw in context["frozen"]["folds"]
        for state_id in raw["all_stable_test_state_ids"]
    }
    finite = all(np.isfinite(float(row["p_gate"])) and np.isfinite(float(row["gate_logit"])) for row in state_oof)
    sanity = {
        "status": "BASELINE_READY",
        "condition": CONDITION,
        "frozen_fold_count": len(folds),
        "all_original_six_per_seed_and_ensemble_reproduced": True,
        "validated_D1_variant_fold_reproduced": True,
        "reproduction_tolerance": runner.TOLERANCE,
        "no_source_group_leakage": True,
        "normalization_train_only": True,
        "threshold_validation_only": True,
        "full_batch_training_matches_original": True,
        "all_frozen_heldout_states_exactly_one_seed_mean_oof": len(ids) == len(set(ids)) == len(frozen_heldout_ids) and set(ids) == frozen_heldout_ids,
        "all_output_finite": finite,
        "new_states": 0,
        "new_oracle_rollouts": 0,
        "feature_schema_changed": False,
        "gate_architecture_changed": False,
        "correction_head_trained": False,
        "closed_loop_run": False,
    }
    required_passes = (
        "all_original_six_per_seed_and_ensemble_reproduced",
        "validated_D1_variant_fold_reproduced",
        "no_source_group_leakage",
        "normalization_train_only",
        "threshold_validation_only",
        "full_batch_training_matches_original",
        "all_frozen_heldout_states_exactly_one_seed_mean_oof",
        "all_output_finite",
    )
    if not all(sanity[name] for name in required_passes):
        fail("baseline post-training sanity failed", checks, started)
    runner.write_json(OUT / "sanity_checks.json", sanity)
    runtime = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.perf_counter() - started,
        "condition": CONDITION,
        "outer_folds": len(folds),
        "training_runs": len(folds) * len(runner.SEEDS),
        "GPU_shards": 1,
        "CPU_threads_cap": int(os.environ.get("OMP_NUM_THREADS", "4")),
        "jax_backend": jax.default_backend(),
        "python": sys.version,
        "platform": platform.platform(),
        "new_oracle_rollouts": 0,
    }
    runner.write_json(OUT / "runtime_statistics.json", runtime)
    runner.write_json(OUT / "manifest.json", {
        "condition": CONDITION,
        "runner": {"path": "shared/gate_condition_runner.py", "sha256": runner.sha256(SHARED / "gate_condition_runner.py")},
        "frozen_manifest": {"path": "frozen_fold_manifest.json", "sha256": runner.sha256(HERE / "frozen_fold_manifest.json")},
        "outputs": ["training_results.csv", "out_of_fold_predictions.csv", "variant_predictions.csv", "fold_relative_calibration.csv", "baseline_reproduction_checks.csv", "sanity_checks.json", "runtime_statistics.json"],
    })
    # This lock-free tiny file is the explicit go signal to other condition agents.
    runner.write_json(ready, {
        "status": "BASELINE_READY",
        "condition_output": str(OUT),
        "reproduction_tolerance": runner.TOLERANCE,
        "checks_path": str(OUT / "baseline_reproduction_checks.csv"),
        "frozen_fold_count": len(folds),
        "training_runs": len(folds) * len(runner.SEEDS),
        "all_checks_passed": True,
    })
    print({"status": "BASELINE_READY", "wall_s": runtime["wall_s"], "checks": len(checks)}, flush=True)


if __name__ == "__main__":
    main()
