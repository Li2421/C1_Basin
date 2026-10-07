"""Exact one-fold lambda=0 source-balanced reproduction integrity gate."""

from __future__ import annotations

import csv
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import conditional_invariance_gate_runner as runner  # noqa: E402


FOLD_ID = "LOGO_00_anchor_D2_pair228"
SEED = 17
TOLERANCE = 2e-6
OUT = HERE / "lambda0_reproduction"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    jax.config.update("jax_enable_x64", True)
    OUT.mkdir(parents=True, exist_ok=True)
    context, folds = runner.load_context_and_folds()
    fold = next(value for value in folds if value["fold_id"] == FOLD_ID)
    result = runner.train_one_seed(
        fold, context, loss_mode="source_balanced_bce", lambda_invariance=0.0, seed=SEED,
        checkpoint_path=OUT / "checkpoint.npz", keep_history=False,
    )
    current = {row["state_id"]: row for row in runner.seed_prediction_rows(result, "LAMBDA0_SOURCE_BALANCED_REPRODUCTION")}
    saved = read_csv(ROOT / "diagnostics" / "gate_source_balancing_v1" / "conditions" / "source_group_balanced" / "out_of_fold_predictions.csv")
    reference = {row["state_id"]: row for row in saved if row["fold_id"] == FOLD_ID and str(row["training_seed"]) == str(SEED)}
    if set(current) != set(reference):
        raise RuntimeError(("state identity mismatch", len(current), len(reference)))
    p_diff = max(abs(float(current[key]["p_gate"]) - float(reference[key]["p_gate"])) for key in current)
    t_diff = max(abs(float(current[key]["validation_selected_threshold"]) - float(reference[key]["validation_selected_threshold"])) for key in current)
    layout = runner.class_cell_layout(fold, context)
    outer = fold["outer_group"]
    leakage_free = (outer not in fold["train_groups"] and outer not in fold["validation_groups"] and
                    not any(context["state"][state_id]["source_group"] == outer for state_id in fold["train_ids"] + fold["validation_ids"]))
    no_test_group_in_layout = outer not in set(layout["groups"])
    row = {
        "fold_id": FOLD_ID, "seed": SEED, "lambda_invariance": 0.0,
        "reference_condition": "SOURCE_GROUP_BALANCED", "test_state_count": len(current),
        "max_abs_probability_difference": p_diff, "max_abs_threshold_difference": t_diff,
        "tolerance": TOLERANCE, "outer_group_excluded_from_train_validation": leakage_free,
        "outer_group_excluded_from_invariance_layout": no_test_group_in_layout,
        "eligible_training_source_cells_class_0": int(layout["eligible_group_count_by_class"][0]),
        "eligible_training_source_cells_class_1": int(layout["eligible_group_count_by_class"][1]),
        "passed": bool(p_diff <= TOLERANCE and t_diff <= TOLERANCE and leakage_free and no_test_group_in_layout),
    }
    runner.base.write_csv(OUT / "lambda0_reproduction_checks.csv", [row])
    ready = {
        "status": "INVARIANCE_BASELINE_READY" if row["passed"] else "INVARIANCE_BASELINE_BLOCKED",
        "all_checks_passed": bool(row["passed"]), "fold_id": FOLD_ID, "seed": SEED,
        "lambda_invariance": 0.0, "reproduction_tolerance": TOLERANCE,
        "max_abs_probability_difference": p_diff, "max_abs_threshold_difference": t_diff,
        "outer_test_group_excluded_from_train_validation_normalization_threshold_and_regularizer": bool(leakage_free and no_test_group_in_layout),
        "eligibility_counts_training_source_groups_by_class": layout["eligible_group_count_by_class"].tolist(),
        "new_states": 0, "new_oracle_rollouts": 0,
    }
    runner.base.write_json(HERE / "INVARIANCE_BASELINE_READY.json", ready)
    runner.base.write_json(OUT / "runtime_statistics.json", {
        "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.perf_counter() - started, "jax_backend": jax.default_backend(),
        "GPU_shards": 0, "CPU_threads_cap": int(os.environ.get("OMP_NUM_THREADS", "4")),
        "platform": platform.platform(), "new_oracle_rollouts": 0,
    })
    if not row["passed"]:
        raise RuntimeError(("lambda=0 source-balanced reproduction failed", row))
    print(json.dumps(ready, sort_keys=True))


if __name__ == "__main__":
    main()
