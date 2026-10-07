"""Validation-only Pareto selection, then frozen offline diagnostic evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_strict_deadlock_coverage_retrain_v1"
DATASET = ROOT / "diagnostics/gphi_training_dataset_strict_deadlock_v1"
BASE = ROOT / "diagnostics/gphi_training_startup_complete_v1/artifacts"
OLD = ROOT / "diagnostics/gphi_startup_warm_pareto_v1/best_balanced_checkpoint.npz"
CAPACITY = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1"
SEEDS = (17, 23, 41)
V3_COUNT = 18816
BASE_COUNT = 26432

import sys
sys.path.insert(0, str(ROOT))
import diagnostics.gphi_pilot_training_v2.train_and_evaluate as core
from diagnostics.gphi_training_startup_complete_v1.pipeline import numpy_predict_checkpoint
from diagnostics.gphi_strict_deadlock_coverage_retrain_v1.prepare_dataset import (
    build_samples, load_capacity_states, target_error_rows,
)
from single_integrator.cbf import CBFConfig
from single_integrator.environment import Config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def nondominated(rows: list[dict]) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (
        row["startup_validation_state_grouped_mean_l2"],
        row["warm_validation_state_grouped_mean_l2"],
        row["epoch"], row["seed"],
    ))
    frontier = []
    best_warm = float("inf")
    for row in ordered:
        warm = row["warm_validation_state_grouped_mean_l2"]
        if warm < best_warm:
            frontier.append(dict(row))
            best_warm = warm
    return sorted(frontier, key=lambda row: (row["seed"], row["epoch"]))


def choose_balanced(rows: list[dict]) -> tuple[dict, list[dict]]:
    frontier = nondominated(rows)
    minimum_startup = min(row["startup_validation_state_grouped_mean_l2"] for row in rows)
    minimum_warm = min(row["warm_validation_state_grouped_mean_l2"] for row in rows)
    for row in frontier:
        row["startup_relative_regret"] = row["startup_validation_state_grouped_mean_l2"] / minimum_startup - 1.0
        row["warm_relative_regret"] = row["warm_validation_state_grouped_mean_l2"] / minimum_warm - 1.0
        row["max_relative_regret"] = max(row["startup_relative_regret"], row["warm_relative_regret"])
        row["sum_relative_regret"] = row["startup_relative_regret"] + row["warm_relative_regret"]
    selected = min(frontier, key=lambda row: (
        row["max_relative_regret"], row["sum_relative_regret"],
        row["all_validation_state_grouped_mean_l2"], row["epoch"], row["seed"],
    ))
    return selected, frontier


def cohort_rows(checkpoint: Path, arrays: dict[str, np.ndarray], split_name: str) -> list[dict]:
    indices = np.flatnonzero(arrays["split"] == split_name)
    if np.any(indices >= BASE_COUNT):
        raise RuntimeError("coverage additions leaked into validation/test")
    prediction = numpy_predict_checkpoint(checkpoint, arrays["features"][indices])
    startup = (indices >= V3_COUNT) & (indices < BASE_COUNT)
    rows = []
    for cohort, mask in (
        ("ALL", np.ones(len(indices), dtype=bool)),
        ("STARTUP", startup),
        ("WARM_V3", ~startup),
    ):
        metric = core.subset_metrics(
            prediction[mask], arrays["targets"][indices][mask], arrays["state_id"][indices][mask]
        )
        rows.append({"split": split_name, "cohort": cohort, **metric})
    return rows


def main() -> None:
    started = time.monotonic()
    rows = []
    runtimes = []
    for seed in SEEDS:
        progress = json.loads((HERE / f"seed{seed}/progress.json").read_text())
        if progress["status"] != "COMPLETED" or int(progress["epoch"]) != 1200:
            raise RuntimeError((seed, "incomplete training", progress))
        source_rows = read_csv(HERE / f"seed{seed}/epoch_metrics.csv")
        if len(source_rows) != 1200:
            raise RuntimeError((seed, len(source_rows)))
        shutil.copy2(HERE / f"seed{seed}/epoch_metrics.csv", HERE / f"epoch_metrics_seed{seed}.csv")
        for source in source_rows:
            parsed = {"seed": int(source["seed"]), "epoch": int(source["epoch"])}
            parsed.update({key: float(value) for key, value in source.items() if key not in ("seed", "epoch")})
            rows.append(parsed)
        runtimes.append(json.loads((HERE / f"seed{seed}/runtime.json").read_text()))

    selected, frontier = choose_balanced(rows)
    write_csv(HERE / "checkpoint_pareto.csv", frontier)
    checkpoint_source = HERE / f"seed{selected['seed']}/checkpoints/epoch_{selected['epoch']:04d}.npz"
    selected_checkpoint = HERE / "best_strict_deadlock_coverage_checkpoint.npz"
    shutil.copy2(checkpoint_source, selected_checkpoint)
    checkpoint_hash = sha256(selected_checkpoint)

    # Only after this copy and hash freeze may test or strict-deadlock target
    # diagnostics be evaluated.
    with np.load(DATASET / "samples.npz", allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]).copy() for key in source.files}
    evaluation_rows = []
    for model, checkpoint in (("OLD", OLD), ("NEW", selected_checkpoint)):
        for split_name in ("validation", "test"):
            for value in cohort_rows(checkpoint, arrays, split_name):
                evaluation_rows.append({"model": model, **value})
    write_csv(HERE / "startup_warm_retention.csv", evaluation_rows)

    capacity_manifest = json.loads((CAPACITY / "strict_deadlock_manifest.json").read_text())
    config = Config(**capacity_manifest["environment"])
    cbf = CBFConfig(**capacity_manifest["cbf"])
    states, robust_records, _ = load_capacity_states()
    historical = [row for row in states if row["benchmark"] == "historical"]
    fresh = [row for row in states if row["benchmark"] == "fresh_unseen"]
    _, samples_by_state = build_samples(states, robust_records, config, cbf)
    post_rows = target_error_rows(
        selected_checkpoint, historical, samples_by_state, config, cbf, "historical11"
    ) + target_error_rows(
        selected_checkpoint, fresh, samples_by_state, config, cbf, "fresh6_heldout"
    )
    write_csv(HERE / "post_retrain_target_error.csv", post_rows)

    selection_payload = {
        "selection_frozen_before_test_or_diagnostic_evaluation": True,
        "selection_rule": json.loads((HERE / "config.json").read_text())["selection_rule"],
        "test_used_for_selection": False,
        "strict_deadlock_diagnostics_used_for_selection": False,
        "selected_seed": selected["seed"],
        "selected_epoch": selected["epoch"],
        "selected_validation_metrics": selected,
        "checkpoint": str(selected_checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "source_checkpoint": str(checkpoint_source),
        "pareto_checkpoint_count": len(frontier),
    }
    write_json(HERE / "selected_checkpoint.json", selection_payload)
    old_test = {row["cohort"]: row for row in evaluation_rows if row["model"] == "OLD" and row["split"] == "test"}
    new_test = {row["cohort"]: row for row in evaluation_rows if row["model"] == "NEW" and row["split"] == "test"}
    report = f"""# Strict-deadlock coverage retraining

The model was trained from scratch on the immutable startup-complete dataset
plus exactly 11 historical earliest-robust strict-deadlock states (64 Flow
variants per state).  No state was duplicated or specially weighted.  The six
fresh-unseen strict-deadlock states were absent from training, normalization,
validation, and checkpoint selection.

## Validation-only checkpoint

- Seed: {selected['seed']}
- Epoch: {selected['epoch']}
- SHA256: `{checkpoint_hash}`
- Startup validation mean L2: {selected['startup_validation_state_grouped_mean_l2']:.8f}
- Warm validation mean L2: {selected['warm_validation_state_grouped_mean_l2']:.8f}
- Aggregate validation mean L2: {selected['all_validation_state_grouped_mean_l2']:.8f}

## Frozen matched test retention

| Cohort | Old mean L2 | New mean L2 |
|---|---:|---:|
| Startup | {old_test['STARTUP']['state_grouped_mean_l2']:.8f} | {new_test['STARTUP']['state_grouped_mean_l2']:.8f} |
| Warm V3 | {old_test['WARM_V3']['state_grouped_mean_l2']:.8f} | {new_test['WARM_V3']['state_grouped_mean_l2']:.8f} |
| All | {old_test['ALL']['state_grouped_mean_l2']:.8f} | {new_test['ALL']['state_grouped_mean_l2']:.8f} |

Closed-loop outcomes are deliberately not used anywhere in this selection.
"""
    (HERE / "training_report.md").write_text(report)
    write_json(HERE / "training_runtime.json", {
        "seed_runs": runtimes,
        "parallel_training_wall_seconds": max(row["wall_seconds"] for row in runtimes),
        "sum_seed_training_seconds": sum(row["wall_seconds"] for row in runtimes),
        "aggregation_offline_seconds": time.monotonic() - started,
        "gpu_shards": 3,
        "allocated_cpu_cores": 6,
        "allocated_memory_gb": 48,
    })
    print(json.dumps({
        "status": "PASS",
        "selected_seed": selected["seed"],
        "selected_epoch": selected["epoch"],
        "checkpoint_sha256": checkpoint_hash,
        "new_test_startup_mean_l2": new_test["STARTUP"]["state_grouped_mean_l2"],
        "new_test_warm_mean_l2": new_test["WARM_V3"]["state_grouped_mean_l2"],
    }, indent=2))


if __name__ == "__main__":
    main()
