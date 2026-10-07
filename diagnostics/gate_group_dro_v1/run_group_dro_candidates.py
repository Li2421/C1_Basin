"""Run the pre-registered Group-DRO eta candidates without outer-test selection.

Each worker owns a disjoint deterministic subset of
``(eta_q, frozen LOGO fold, seed)`` tasks.  It records validation-only
outputs for hyperparameter selection and an exact checkpoint plus compact
training-group history for the subsequently selected candidate.  Although the
underlying frozen trainer computes its test prediction at the end in order to
preserve its return contract, this worker neither serializes nor reads that
prediction: selection is based only on saved validation-state predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(OUT))
import group_dro_gate_runner as dro  # noqa: E402


ETAS = (0.01, 0.05, 0.1)
CONDITION = "GROUP_DRO_BCE"


def eta_tag(eta: float) -> str:
    return f"eta_{eta:.2f}".replace(".", "p")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_history(path: Path, rows: list[dict]) -> None:
    """Compact, lossless numeric history; selected histories are expanded later."""
    if not rows:
        raise RuntimeError("Group-DRO unexpectedly emitted no group history")
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "epoch": np.asarray([int(row["epoch"]) for row in rows], np.int32),
        "group": np.asarray([str(row["training_source_group"]) for row in rows]),
        "q_g": np.asarray([float(row["q_g"]) for row in rows], np.float32),
        "L_g": np.asarray([float(row["L_g"]) for row in rows], np.float32),
        "objective": np.asarray([float(row["objective"]) for row in rows], np.float32),
        "q_entropy": np.asarray([float(row["q_entropy"]) for row in rows], np.float32),
        "q_effective_group_count": np.asarray([float(row["q_effective_group_count"]) for row in rows], np.float32),
        "validation_state_BCE": np.asarray([
            np.nan if row["validation_state_BCE"] is None else float(row["validation_state_BCE"])
            for row in rows
        ], np.float32),
    }
    np.savez_compressed(path, **values)


def task_dir(fold_id: str, eta: float, seed: int) -> Path:
    return OUT / "candidate_runs" / eta_tag(eta) / fold_id / f"seed_{seed}"


def task_complete(path: Path) -> bool:
    marker = path / "DONE.json"
    if not marker.exists():
        return False
    info = json.loads(marker.read_text())
    required = ("checkpoint", "validation_predictions", "history")
    return info.get("status") == "COMPLETE" and all((path / info[name]).exists() for name in required)


def run_task(fold: dict, context: dict, eta: float, seed: int) -> dict:
    destination = task_dir(fold["fold_id"], eta, seed)
    if task_complete(destination):
        return {"fold_id": fold["fold_id"], "eta_q": eta, "seed": seed, "status": "SKIPPED_COMPLETE"}
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    result = dro.train_one_seed(
        fold,
        context,
        loss_mode="group_dro_bce",
        eta_q=eta,
        seed=seed,
        checkpoint_path=destination / "checkpoint.npz",
        keep_group_history=True,
    )
    ids, probabilities, labels = result["validation"]
    validation_rows = [
        {
            "fold_id": fold["fold_id"],
            "heldout_source_group": fold["outer_group"],
            "eta_q": eta,
            "training_seed": seed,
            "state_id": str(state_id),
            "oracle_label": int(label),
            "p_gate": float(probability),
        }
        for state_id, probability, label in zip(ids, probabilities, labels)
    ]
    write_csv(
        destination / "validation_predictions.csv",
        validation_rows,
        ["fold_id", "heldout_source_group", "eta_q", "training_seed", "state_id", "oracle_label", "p_gate"],
    )
    save_history(destination / "group_history.npz", result["group_history"])
    summary = {
        "status": "COMPLETE",
        "fold_id": fold["fold_id"],
        "heldout_source_group": fold["outer_group"],
        "eta_q": eta,
        "training_seed": seed,
        "best_epoch": result["best_epoch"],
        "epochs_ran": result["epochs_ran"],
        "validation_state_BCE": result["validation_bce"],
        "validation_selected_threshold": result["threshold"],
        "checkpoint": "checkpoint.npz",
        "validation_predictions": "validation_predictions.csv",
        "history": "group_history.npz",
        "training_source_group_count": len(result["group_layout"]["groups"]),
        "group_weights_use_only_training_groups": True,
        "outer_test_used_for_eta_selection": False,
        "finite": bool(np.isfinite(probabilities).all()),
        "elapsed_s": time.perf_counter() - started,
    }
    dro.base.write_json(destination / "DONE.json", summary)
    return {"fold_id": fold["fold_id"], "eta_q": eta, "seed": seed, "status": "COMPLETE", **summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if args.workers not in (1, 2) or not 0 <= args.worker < args.workers:
        raise ValueError("workers must be 1 or 2, and worker must be in range")
    ready = json.loads((OUT / "BASELINE_DRO_READY.json").read_text())
    if ready.get("status") != "BASELINE_DRO_READY" or not ready.get("all_checks_passed"):
        raise RuntimeError(("baseline readiness check failed", ready))
    jax.config.update("jax_enable_x64", True)
    context, folds = dro.load_context_and_folds()
    tasks = [(eta, fold, seed) for eta in ETAS for fold in folds for seed in dro.SEEDS]
    assigned = [task for index, task in enumerate(tasks) if index % args.workers == args.worker]
    started = time.perf_counter()
    log_rows = []
    for ordinal, (eta, fold, seed) in enumerate(assigned, 1):
        row = run_task(fold, context, eta, seed)
        log_rows.append(row)
        print({"worker": args.worker, "task": ordinal, "task_count": len(assigned), **row}, flush=True)
    dro.base.write_csv(
        OUT / "candidate_runs" / f"worker_{args.worker}_results.csv",
        log_rows,
        ["fold_id", "eta_q", "seed", "training_seed", "status", "heldout_source_group", "best_epoch", "epochs_ran", "validation_state_BCE", "validation_selected_threshold", "checkpoint", "validation_predictions", "history", "training_source_group_count", "group_weights_use_only_training_groups", "outer_test_used_for_eta_selection", "finite", "elapsed_s"],
    )
    dro.base.write_json(OUT / "candidate_runs" / f"worker_{args.worker}_runtime.json", {
        "worker": args.worker,
        "workers": args.workers,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.perf_counter() - started,
        "task_count": len(assigned),
        "GPU_shards": 1,
        "CPU_threads_cap": int(os.environ.get("OMP_NUM_THREADS", "4")),
        "jax_backend": jax.default_backend(),
        "platform": platform.platform(),
        "new_states": 0,
        "new_oracle_rollouts": 0,
    })


if __name__ == "__main__":
    main()
