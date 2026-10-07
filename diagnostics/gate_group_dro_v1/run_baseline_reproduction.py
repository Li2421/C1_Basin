"""One-fold/one-seed source-balanced BCE reproduction gate for Group-DRO code."""

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
import numpy as np

THIS = Path(__file__).resolve().parent
ROOT = THIS.parents[1]
sys.path.insert(0, str(THIS))
import group_dro_gate_runner as dro  # noqa: E402


FOLD_ID = "LOGO_00_anchor_D2_pair228"
SEED = 17
TOLERANCE = 2e-6
OUT = THIS / "baseline_reproduction"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    started = time.perf_counter()
    jax.config.update("jax_enable_x64", True)
    OUT.mkdir(parents=True, exist_ok=True)
    context, folds = dro.load_context_and_folds()
    fold = next(item for item in folds if item["fold_id"] == FOLD_ID)
    result = dro.train_one_seed(
        fold, context, loss_mode="source_balanced_bce", seed=SEED,
        checkpoint_path=OUT / "checkpoint.npz", keep_group_history=False,
    )
    current = {row["state_id"]: row for row in dro.seed_prediction_rows(result, context, "SOURCE_GROUP_BALANCED_BCE_REPRODUCTION")}
    saved = read_csv(ROOT / "diagnostics" / "gate_source_balancing_v1" / "conditions" / "source_group_balanced" / "out_of_fold_predictions.csv")
    reference = {
        row["state_id"]: row for row in saved
        if row["fold_id"] == FOLD_ID and str(row["training_seed"]) == str(SEED)
    }
    if set(current) != set(reference):
        raise RuntimeError(("state identity mismatch", len(current), len(reference)))
    p_diff = max(abs(float(current[state_id]["p_gate"]) - float(reference[state_id]["p_gate"])) for state_id in current)
    t_diff = max(abs(float(current[state_id]["validation_selected_threshold"]) - float(reference[state_id]["validation_selected_threshold"])) for state_id in current)
    # Exact fold leakage audit is performed independently of any output metric.
    outer = fold["outer_group"]
    leakage_free = (outer not in fold["train_groups"] and outer not in fold["validation_groups"] and
                    not any(context["state"][state_id]["source_group"] == outer for state_id in fold["train_ids"] + fold["validation_ids"]))
    rows = [{
        "fold_id": FOLD_ID,
        "seed": SEED,
        "reference_condition": "SOURCE_GROUP_BALANCED",
        "reproduced_loss_mode": "source_balanced_bce",
        "test_state_count": len(current),
        "max_abs_probability_difference": p_diff,
        "max_abs_threshold_difference": t_diff,
        "tolerance": TOLERANCE,
        "leakage_free": leakage_free,
        "passed": bool(p_diff <= TOLERANCE and t_diff <= TOLERANCE and leakage_free),
    }]
    dro.base.write_csv(OUT / "baseline_reproduction_checks.csv", rows)
    passed = bool(rows[0]["passed"])
    ready = {
        "status": "BASELINE_DRO_READY" if passed else "BASELINE_DRO_BLOCKED",
        "all_checks_passed": passed,
        "fold_id": FOLD_ID,
        "seed": SEED,
        "reproduction_tolerance": TOLERANCE,
        "max_abs_probability_difference": p_diff,
        "max_abs_threshold_difference": t_diff,
        "outer_group_excluded_from_train_validation_normalization_and_threshold": leakage_free,
        "frozen_fold_count": len(folds),
        "loss_code": "group_dro_gate_runner.py; BASELINE_BCE_PRESERVED markers retained",
        "new_states": 0,
        "new_oracle_rollouts": 0,
    }
    dro.base.write_json(THIS / "BASELINE_DRO_READY.json", ready)
    dro.base.write_json(OUT / "runtime_statistics.json", {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.perf_counter() - started,
        "jax_backend": jax.default_backend(),
        "GPU_shards": 0,
        "CPU_threads_cap": int(os.environ.get("OMP_NUM_THREADS", "4")),
        "platform": platform.platform(),
        "new_oracle_rollouts": 0,
    })
    if not passed:
        raise RuntimeError(("source-balanced reproduction failed", rows[0]))
    print(json.dumps(ready, sort_keys=True))


if __name__ == "__main__":
    main()
