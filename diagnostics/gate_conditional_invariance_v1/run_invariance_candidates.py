"""Sequential, validation-only candidate runner for conditional invariance.

Candidate task outputs consist only of training histories, checkpoints, and
validation state predictions.  The eta/lambda selector consequently cannot
inspect outer-test outcomes before it commits one lambda per frozen LOGO fold.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT))
import conditional_invariance_gate_runner as inv  # noqa: E402


CONDITION = "CONDITIONAL_SOURCE_INVARIANCE"


def lambda_tag(value: float) -> str:
    return f"lambda_{value:.2f}".replace(".", "p")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def task_directory(fold_id: str, value: float, seed: int) -> Path:
    return OUT / "candidate_runs" / lambda_tag(value) / fold_id / f"seed_{seed}"


def complete(path: Path) -> bool:
    marker = path / "DONE.json"
    if not marker.exists():
        return False
    detail = json.loads(marker.read_text())
    return detail.get("status") == "COMPLETE" and all((path / detail[key]).exists() for key in ("checkpoint", "validation_predictions", "history"))


def run_one(fold: dict, context: dict, value: float, seed: int) -> dict:
    destination = task_directory(fold["fold_id"], value, seed)
    if complete(destination):
        return {"fold_id": fold["fold_id"], "lambda_invariance": value, "seed": seed, "status": "SKIPPED_COMPLETE"}
    destination.mkdir(parents=True, exist_ok=True)
    mode = "source_balanced_bce" if value == 0.0 else "conditional_source_invariance_bce"
    started = time.perf_counter()
    result = inv.train_one_seed(
        fold, context, loss_mode=mode, seed=seed, lambda_invariance=value,
        checkpoint_path=destination / "checkpoint.npz", keep_history=True,
    )
    ids, probabilities, labels = result["validation"]
    validation = [{
        "fold_id": fold["fold_id"], "heldout_source_group": fold["outer_group"],
        "lambda_invariance": value, "training_seed": seed, "state_id": str(state_id),
        "oracle_label": int(label), "p_gate": float(probability),
    } for state_id, probability, label in zip(ids, probabilities, labels)]
    write_csv(destination / "validation_predictions.csv", validation, ["fold_id", "heldout_source_group", "lambda_invariance", "training_seed", "state_id", "oracle_label", "p_gate"])
    inv.base.write_csv(destination / "training_history.csv", result["history"])
    detail = {
        "status": "COMPLETE", "fold_id": fold["fold_id"], "heldout_source_group": fold["outer_group"],
        "lambda_invariance": value, "training_seed": seed, "loss_mode": mode,
        "best_epoch": result["best_epoch"], "epochs_ran": result["epochs_ran"],
        "validation_state_BCE": result["validation_bce"], "validation_selected_threshold": result["threshold"],
        "checkpoint": "checkpoint.npz", "validation_predictions": "validation_predictions.csv", "history": "training_history.csv",
        "eligible_training_source_groups_by_class": result["layout"]["eligible_group_count_by_class"].tolist(),
        "outer_test_used_for_lambda_selection": False, "finite": bool(np.isfinite(probabilities).all()),
        "elapsed_s": time.perf_counter() - started,
    }
    inv.base.write_json(destination / "DONE.json", detail)
    return {"fold_id": fold["fold_id"], "lambda_invariance": value, "seed": seed, **detail}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    ready = json.loads((OUT / "INVARIANCE_BASELINE_READY.json").read_text())
    if ready.get("status") != "INVARIANCE_BASELINE_READY" or not ready.get("all_checks_passed"):
        raise RuntimeError(("invariance baseline not ready", ready))
    jax.config.update("jax_enable_x64", True)
    context, folds = inv.load_context_and_folds()
    started = time.perf_counter()
    rows = []
    # lambda=0 is materialized from the already hash-verified frozen
    # source-balanced checkpoints by ``materialize_lambda0_baseline.py``.
    # Never retrain it here: the positive-lambda grid is the only unfinished
    # optimization work in this controlled experiment.
    tasks = [(value, fold, seed) for value in inv.LAMBDA_GRID if value > 0.0 for fold in folds for seed in inv.SEEDS]
    for ordinal, (value, fold, seed) in enumerate(tasks, 1):
        row = run_one(fold, context, value, seed)
        rows.append(row)
        print({"task": ordinal, "task_count": len(tasks), **row}, flush=True)
    inv.base.write_csv(OUT / "candidate_runs" / "candidate_training_results.csv", rows)
    inv.base.write_json(OUT / "candidate_runs" / "candidate_runtime_statistics.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(), "wall_s": time.perf_counter() - started,
        "candidate_training_runs": len(tasks), "GPU_shards": 1, "CPU_threads_cap": int(os.environ.get("OMP_NUM_THREADS", "4")),
        "lambda_grid": list(inv.LAMBDA_GRID), "new_states": 0, "new_oracle_rollouts": 0,
    })


if __name__ == "__main__":
    main()
