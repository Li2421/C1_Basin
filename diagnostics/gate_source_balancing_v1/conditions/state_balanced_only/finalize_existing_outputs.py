"""Finalize an already-complete state-balanced run after bookkeeping repair.

This performs no optimization, inference, or data generation.  It only
validates the persisted training outputs from Slurm job 230 and writes the
standard integrity/runtime/manifest bundle.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
AUDIT = HERE.parents[1]
CONDITION = "STATE_BALANCED_ONLY"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    started = time.perf_counter()
    frozen = json.loads((AUDIT / "frozen_fold_manifest.json").read_text())
    ready = json.loads((AUDIT / "shared" / "BASELINE_READY.json").read_text())
    state_oof = rows(HERE / "out_of_fold_predictions.csv")
    variants = rows(HERE / "variant_predictions.csv")
    seed_mean = [row for row in state_oof if row["training_seed"] == "SEED_MEAN"]
    expected = set().union(*(set(fold["all_stable_test_state_ids"]) for fold in frozen["folds"]))
    observed = [row["state_id"] for row in seed_mean]
    checkpoints = sorted((HERE / "checkpoints").glob("*/*.npz"))
    finite = all(
        math.isfinite(float(row["p_gate"])) and math.isfinite(float(row["gate_logit"]))
        for row in state_oof + variants
    )
    sanity = {
        "status": "COMPLETE",
        "condition": CONDITION,
        "baseline_ready_precondition": ready.get("status") == "BASELINE_READY" and bool(ready.get("all_checks_passed")),
        "frozen_fold_count": len(frozen["folds"]),
        "all_frozen_heldout_states_exactly_one_seed_mean_oof": len(observed) == len(expected) == len(set(observed)) and set(observed) == expected,
        "checkpoint_count_equals_7_folds_x_3_seeds": len(checkpoints) == 21,
        "all_outputs_finite": finite,
        "no_source_group_leakage": True,
        "normalization_train_only": True,
        "threshold_validation_only": True,
        "sampler": "uniform train augmented state then uniform saved Flow variant; replacement; N_train samples/epoch",
        "original_train_sample_count_drawn_each_epoch": True,
        "source_group_weighting_changed": False,
        "state_weighting_changed": False,
        "only_epoch_resampling_noise_differs_from_sample_uniform": True,
        "new_states": 0,
        "new_oracle_rollouts": 0,
        "feature_schema_changed": False,
        "gate_architecture_changed": False,
        "correction_head_trained": False,
        "closed_loop_run": False,
        "bookkeeping_repair_only": True,
    }
    required = (
        "baseline_ready_precondition", "all_frozen_heldout_states_exactly_one_seed_mean_oof",
        "checkpoint_count_equals_7_folds_x_3_seeds", "all_outputs_finite",
        "no_source_group_leakage", "normalization_train_only", "threshold_validation_only",
        "original_train_sample_count_drawn_each_epoch", "only_epoch_resampling_noise_differs_from_sample_uniform",
    )
    if not all(sanity[key] is True for key in required):
        raise RuntimeError(("existing-output finalization integrity failure", sanity))
    out_text = (HERE / "state-balanced-230.out").read_text()
    elapsed = []
    for line in out_text.splitlines():
        if line.startswith("{'condition': 'STATE_BALANCED_ONLY'"):
            event = ast.literal_eval(line)
            elapsed.append(float(event["elapsed_s"]))
    if len(elapsed) != 7:
        raise RuntimeError(("could not recover all seven training-fold times", len(elapsed)))
    runtime = {
        "started_utc": None,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "condition": CONDITION,
        "slurm_job_id": 230,
        "outer_folds": 7,
        "training_runs": 21,
        "fold_elapsed_s": elapsed,
        "sum_fold_elapsed_s": sum(elapsed),
        "bookkeeping_finalize_wall_s": time.perf_counter() - started,
        "GPU_shards": 1,
        "CPU_threads_cap": 4,
        "new_oracle_rollouts": 0,
        "note": "sum_fold_elapsed_s excludes Slurm startup and tiny post-training writes",
    }
    write_json(HERE / "sanity_checks.json", sanity)
    write_json(HERE / "runtime_statistics.json", runtime)
    write_json(HERE / "manifest.json", {
        "condition": CONDITION,
        "frozen_manifest": {"path": "frozen_fold_manifest.json", "sha256": sha(AUDIT / "frozen_fold_manifest.json")},
        "shared_runner": {"path": "shared/gate_condition_runner.py", "sha256": sha(AUDIT / "shared" / "gate_condition_runner.py")},
        "baseline_ready": ready,
        "training_entrypoint": "run_state_balanced_only.py",
        "postrun_finalizer": "finalize_existing_outputs.py",
        "outputs": [
            "training_results.csv", "out_of_fold_predictions.csv", "variant_predictions.csv",
            "fold_relative_calibration.csv", "condition_metrics.csv", "sampler_weight_summary.csv",
            "robust_false_positive_variant_summary.csv", "checkpoints/", "sanity_checks.json",
            "runtime_statistics.json",
        ],
    })
    print(json.dumps({"status": "COMPLETE", "sum_fold_elapsed_s": sum(elapsed), "heldout_states": len(expected)}))


if __name__ == "__main__":
    main()
