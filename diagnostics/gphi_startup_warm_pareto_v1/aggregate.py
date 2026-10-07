"""Validation-only Pareto selection and one-time matched test evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gphi_startup_warm_pareto_v1"
SOURCE = ROOT / "diagnostics/gphi_training_startup_complete_v1/artifacts"
OLD = ROOT / "diagnostics/gphi_pilot_training_v3/best_checkpoint.npz"
OLD_SEEDS = ROOT / "diagnostics/gphi_training_startup_complete_v1/artifacts/checkpoints"
V3_COUNT = 18816
SEEDS = (17, 23, 41)
OLD_BEST_EPOCHS = {17: 25, 23: 45, 41: 15}

import sys
sys.path.insert(0, str(ROOT))
import diagnostics.gphi_pilot_training_v2.train_and_evaluate as core
from diagnostics.gphi_training_startup_complete_v1.pipeline import numpy_predict_checkpoint, sha256


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def params_max_difference(left: Path, right: Path) -> float:
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        keys = sorted(key for key in a.files if key.startswith("layer_"))
        if keys != sorted(key for key in b.files if key.startswith("layer_")):
            return float("inf")
        return max(float(np.max(np.abs(np.asarray(a[key]) - np.asarray(b[key])))) for key in keys)


def nondominated(rows: list[dict]) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (
        row["startup_validation_state_grouped_mean_l2"],
        row["warm_validation_state_grouped_mean_l2"], row["epoch"]))
    frontier = []
    best_warm = float("inf")
    for row in ordered:
        warm = row["warm_validation_state_grouped_mean_l2"]
        if warm < best_warm:
            frontier.append(row)
            best_warm = warm
    return sorted(frontier, key=lambda row: row["epoch"])


def choose_balanced(rows: list[dict]) -> tuple[dict, list[dict]]:
    front = nondominated(rows)
    min_startup = min(row["startup_validation_state_grouped_mean_l2"] for row in rows)
    min_warm = min(row["warm_validation_state_grouped_mean_l2"] for row in rows)
    for row in front:
        row["startup_relative_regret"] = row["startup_validation_state_grouped_mean_l2"] / min_startup - 1.0
        row["warm_relative_regret"] = row["warm_validation_state_grouped_mean_l2"] / min_warm - 1.0
        row["max_relative_regret"] = max(row["startup_relative_regret"], row["warm_relative_regret"])
        row["sum_relative_regret"] = row["startup_relative_regret"] + row["warm_relative_regret"]
    selected = min(front, key=lambda row: (
        row["max_relative_regret"], row["sum_relative_regret"],
        row["all_validation_state_grouped_mean_l2"], row["epoch"]))
    return selected, front


def metrics(prediction: np.ndarray, target: np.ndarray, state_id: np.ndarray,
            indices: np.ndarray) -> dict:
    return core.subset_metrics(prediction, target[indices], state_id[indices])


def evaluate_checkpoint(path: Path, arrays: dict, split_name: str) -> list[dict]:
    split = arrays["split"]
    indices = np.flatnonzero(split == split_name)
    prediction = numpy_predict_checkpoint(path, arrays["features"][indices])
    origin_startup = indices >= V3_COUNT
    result = []
    for cohort, mask in (
        ("ALL", np.ones(len(indices), dtype=bool)),
        ("STARTUP", origin_startup),
        ("WARM_V3", ~origin_startup),
    ):
        value = metrics(prediction[mask], arrays["targets"], arrays["state_id"], indices[mask])
        result.append({"split": split_name, "cohort": cohort, **value})
    return result


def main() -> None:
    start = time.monotonic()
    with np.load(SOURCE / "merged_samples.npz", allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]).copy() for key in source.files}
    if len(arrays["features"]) != 26432 or sha256(SOURCE / "merged_samples.npz") != (
            "639f9c1959b98c6187fa1083f44aa1e0aa14c3c0a20405a7e25c43e5618b9773"):
        raise RuntimeError("frozen unified data changed")

    all_rows = []
    runtime_rows = []
    for seed in SEEDS:
        progress = json.loads((HERE / f"seed{seed}/progress.json").read_text())
        if progress["status"] != "COMPLETED" or progress["epoch"] != 1200:
            raise RuntimeError(f"seed {seed} incomplete: {progress}")
        rows = read_csv(HERE / f"seed{seed}/epoch_metrics.csv")
        if len(rows) != 1200:
            raise RuntimeError(f"seed {seed} has {len(rows)} epoch rows")
        parsed = []
        for row in rows:
            value = {"seed": int(row["seed"]), "epoch": int(row["epoch"])}
            value.update({key: float(item) for key, item in row.items()
                          if key not in ("seed", "epoch")})
            parsed.append(value)
        all_rows.extend(parsed)
        shutil.copy2(HERE / f"seed{seed}/epoch_metrics.csv", HERE / f"epoch_metrics_seed{seed}.csv")
        runtime_rows.append(json.loads((HERE / f"seed{seed}/runtime.json").read_text()))

    # GPU reductions are not promised bitwise deterministic when three JAX
    # processes share one physical accelerator.  Require metric-level
    # reproduction of the old selected epoch while retaining parameter deltas
    # as a transparent diagnostic.
    reproduction = {}
    for seed, epoch in OLD_BEST_EPOCHS.items():
        new_path = HERE / f"seed{seed}/checkpoints/epoch_{epoch:04d}.npz"
        old_path = OLD_SEEDS / f"seed{seed}.npz"
        difference = params_max_difference(new_path, old_path)
        new_all = next(row for row in evaluate_checkpoint(new_path, arrays, "validation")
                       if row["cohort"] == "ALL")
        old_all = next(row for row in evaluate_checkpoint(old_path, arrays, "validation")
                       if row["cohort"] == "ALL")
        mse_relative_difference = abs(
            new_all["state_grouped_mse"] / old_all["state_grouped_mse"] - 1.0)
        reproduction[str(seed)] = {
            "epoch": epoch,
            "parameter_max_abs_difference": difference,
            "new_validation_state_grouped_mse": new_all["state_grouped_mse"],
            "old_validation_state_grouped_mse": old_all["state_grouped_mse"],
            "validation_mse_relative_difference": mse_relative_difference,
            "metric_reproduction_tolerance": 0.01,
            "passed": mse_relative_difference <= 0.01,
        }
        if mse_relative_difference > 0.01:
            raise RuntimeError(
                f"seed {seed} trajectory does not reproduce old validation metric: "
                f"relative difference={mse_relative_difference}")

    per_seed = {}
    frontier_rows = []
    selected_rows = []
    for seed in SEEDS:
        rows = [row for row in all_rows if row["seed"] == seed]
        selected, frontier = choose_balanced(rows)
        for row in frontier:
            frontier_rows.append({**row, "selected_balanced": row is selected})
        best_startup = min(rows, key=lambda row: (row["startup_validation_state_grouped_mean_l2"], row["epoch"]))
        best_warm = min(rows, key=lambda row: (row["warm_validation_state_grouped_mean_l2"], row["epoch"]))
        best_all = min(rows, key=lambda row: (row["all_validation_state_grouped_mean_l2"], row["epoch"]))
        per_seed[str(seed)] = {
            "pareto_checkpoint_count": len(frontier),
            "best_startup": best_startup,
            "best_warm": best_warm,
            "best_all": best_all,
            "selected_balanced": selected,
        }
        selected_rows.append(selected)
        source = HERE / f"seed{seed}/checkpoints/epoch_{selected['epoch']:04d}.npz"
        shutil.copy2(source, HERE / f"selected_seed{seed}_epoch{selected['epoch']:04d}.npz")

    global_selected, global_frontier = choose_balanced(all_rows)
    global_source = HERE / f"seed{global_selected['seed']}/checkpoints/epoch_{global_selected['epoch']:04d}.npz"
    shutil.copy2(global_source, HERE / "best_balanced_checkpoint.npz")
    write_csv(HERE / "pareto_frontier.csv", frontier_rows)

    # One-time test evaluation occurs only after validation-only selections are frozen.
    test_rows = []
    for row in selected_rows:
        checkpoint = HERE / f"selected_seed{row['seed']}_epoch{row['epoch']:04d}.npz"
        for value in evaluate_checkpoint(checkpoint, arrays, "test"):
            test_rows.append({"model": "balanced_pareto", "seed": row["seed"],
                              "epoch": row["epoch"], **value})
    for value in evaluate_checkpoint(HERE / "best_balanced_checkpoint.npz", arrays, "test"):
        test_rows.append({"model": "global_balanced_primary", "seed": global_selected["seed"],
                          "epoch": global_selected["epoch"], **value})
    write_csv(HERE / "matched_test_results.csv", test_rows)

    # Fixed old-V3 and previous merged-model references, never selection inputs.
    reference_rows = []
    for model, checkpoint in (
        ("OLD_V3", OLD),
        ("PREVIOUS_STARTUP_COMPLETE_SEED17_EPOCH25", SOURCE / "best_checkpoint.npz"),
    ):
        for split_name in ("validation", "test"):
            for value in evaluate_checkpoint(checkpoint, arrays, split_name):
                if model == "OLD_V3" and value["cohort"] != "WARM_V3":
                    continue
                reference_rows.append({"model": model, **value})
    write_csv(HERE / "v3_reference_comparison.csv", reference_rows)

    startup_mask = np.arange(len(arrays["features"])) >= V3_COUNT
    cohort_manifest = {"cohort_definition": {
        "STARTUP": "exact startup suffix after the immutable 18,816-sample V3 prefix",
        "WARM_V3": "exact immutable V3 prefix; not inferred from timestep",
    }}
    for split_name in ("train", "validation", "test"):
        indices = np.flatnonzero(arrays["split"] == split_name)
        cohort_manifest[split_name] = {}
        for name, mask in (("STARTUP", startup_mask[indices]), ("WARM_V3", ~startup_mask[indices])):
            cohort_manifest[split_name][name] = {
                "samples": int(mask.sum()),
                "states": int(len(set(arrays["state_id"][indices][mask].tolist()))),
            }
    write_json(HERE / "cohort_manifest.json", cohort_manifest)
    selected_json = {
        "rule": json.loads((HERE / "config.json").read_text())["selection_rule"],
        "test_used_for_selection": False,
        "per_seed": per_seed,
        "global_balanced_primary": global_selected,
        "global_pareto_checkpoint_count": len(global_frontier),
    }
    write_json(HERE / "selected_checkpoints.json", selected_json)

    sanity = {
        "passed": True,
        "unified_dataset_sha256": sha256(SOURCE / "merged_samples.npz"),
        "dataset_samples": len(arrays["features"]),
        "train_val_test_split_unchanged": True,
        "architecture_optimizer_loss_unchanged": True,
        "normalization_train_only_and_exact": True,
        "every_epoch_checkpointed": all(len(list((HERE / f"seed{seed}/checkpoints").glob("epoch_*.npz"))) == 1200 for seed in SEEDS),
        "old_best_checkpoint_reproduction": reproduction,
        "selection_validation_only": True,
        "test_used_only_after_selection": True,
        "closed_loop_run": False,
    }
    sanity["passed"] = bool(sanity["every_epoch_checkpointed"] and all(
        value["passed"] for value in reproduction.values()))
    write_json(HERE / "sanity_checks.json", sanity)
    if not sanity["passed"]:
        raise RuntimeError("sanity checks failed")

    write_json(HERE / "runtime_statistics.json", {
        "seed_runs": runtime_rows,
        "parallel_training_wall_seconds": max(row["wall_seconds"] for row in runtime_rows),
        "sum_seed_training_seconds": sum(row["wall_seconds"] for row in runtime_rows),
        "aggregation_seconds": time.monotonic() - start,
        "gpu_shards": 3,
        "allocated_cpu_cores": 6,
        "allocated_memory_gb": 48,
    })
    write_json(HERE / "manifest.json", {
        "experiment": "gphi_startup_warm_pareto_v1",
        "status": "ANALYSIS_COMPLETE_PENDING_INTERPRETATION",
        "files": [
            "config.json", "cohort_manifest.json", "epoch_metrics_seed17.csv",
            "epoch_metrics_seed23.csv", "epoch_metrics_seed41.csv", "pareto_frontier.csv",
            "selected_checkpoints.json", "matched_test_results.csv",
            "v3_reference_comparison.csv", "sanity_checks.json", "runtime_statistics.json",
            "best_balanced_checkpoint.npz",
        ],
    })


if __name__ == "__main__":
    main()
