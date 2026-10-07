"""Validation-only Pareto selection followed by frozen eta diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1"
DATASET = ROOT / "diagnostics/gphi_training_dataset_strict_deadlock_v1"
BASE = ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1"
CAPACITY = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1"
SEEDS = (17, 23, 41)
V3_COUNT = 18816
BASE_COUNT = 26432

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SYSROOT))
import diagnostics.gphi_pilot_training_v2.train_and_evaluate as core
from diagnostics.gphi_fixed_d_eta_predictor_v1.eta_model import FixedDEtaPredictor
from diagnostics.gphi_strict_deadlock_coverage_retrain_v1.prepare_dataset import (
    build_samples, load_capacity_states,
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
        row["warm_validation_state_grouped_mean_l2"], row["epoch"], row["seed"],
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


def cosine_rows(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(pred, axis=1) * np.linalg.norm(target, axis=1)
    values = np.full(len(pred), np.nan)
    valid = denom > 1e-12
    values[valid] = np.einsum("ij,ij->i", pred[valid], target[valid]) / denom[valid]
    return values


def metrics(pred: np.ndarray, raw: np.ndarray, target: np.ndarray,
            target_norm: np.ndarray, state_id: np.ndarray) -> dict:
    physical_l2 = np.linalg.norm(pred - target, axis=1)
    norm_l2 = np.linalg.norm(raw - target_norm, axis=1)
    cosines = cosine_rows(pred, target)
    state_values = []
    state_norm_values = []
    for source_id in dict.fromkeys(state_id.tolist()):
        mask = state_id == source_id
        state_values.append(float(physical_l2[mask].mean()))
        state_norm_values.append(float(norm_l2[mask].mean()))
    return {
        "samples": int(len(pred)),
        "states": int(len(state_values)),
        "state_grouped_physical_eta_l2_mean": float(np.mean(state_values)),
        "state_grouped_normalized_eta_l2_mean": float(np.mean(state_norm_values)),
        "physical_eta_l2_mean": float(physical_l2.mean()),
        "physical_eta_l2_median": float(np.median(physical_l2)),
        "coordinate_mae": np.abs(pred - target).mean(axis=0).tolist(),
        "cosine_mean_nonzero": float(np.nanmean(cosines)) if np.isfinite(cosines).any() else None,
        "predicted_eta_norm_mean": float(np.linalg.norm(pred, axis=1).mean()),
        "sample_clipping_fraction": float(np.any((raw < 0) | (raw > 1), axis=1).mean()),
        "coordinate_clipping_fraction": float(((raw < 0) | (raw > 1)).mean()),
    }


def state_prediction_rows(model: FixedDEtaPredictor, selected: list[dict], samples: dict,
                          cohort: str) -> list[dict]:
    rows = []
    for item in selected:
        state_samples = samples[item["state_id"]]
        features = np.stack([row["feature"] for row in state_samples])
        prediction, raw, _ = model.predict(features)
        target = np.asarray(item["eta"], dtype=np.float64)
        error = np.linalg.norm(prediction - target[None], axis=1)
        cos = cosine_rows(prediction, np.repeat(target[None], len(prediction), axis=0))
        row = {
            "cohort": cohort,
            "benchmark": item["benchmark"],
            "case_id": item["case_id"],
            "state_id": item["state_id"],
            "query_label": item["query"]["query_label"],
            "variants": len(prediction),
            "target_eta1": target[0], "target_eta2": target[1], "target_eta3": target[2],
            "prediction_eta1_mean": prediction[:, 0].mean(),
            "prediction_eta2_mean": prediction[:, 1].mean(),
            "prediction_eta3_mean": prediction[:, 2].mean(),
            "prediction_eta1_std": prediction[:, 0].std(),
            "prediction_eta2_std": prediction[:, 1].std(),
            "prediction_eta3_std": prediction[:, 2].std(),
            "physical_eta_l2_mean": error.mean(),
            "physical_eta_l2_median": np.median(error),
            "cosine_mean": np.nanmean(cos) if np.isfinite(cos).any() else None,
            "clipping_fraction": np.any((raw < 0) | (raw > 1), axis=1).mean(),
        }
        rows.append(row)
    return rows


def main() -> None:
    started = time.monotonic()
    all_rows = []
    runtimes = []
    for seed in SEEDS:
        progress = json.loads((HERE / f"seed{seed}/progress.json").read_text())
        if progress["status"] != "COMPLETED" or int(progress["epoch"]) != 1200:
            raise RuntimeError((seed, progress))
        shutil.copy2(HERE / f"seed{seed}/epoch_metrics.csv", HERE / f"epoch_metrics_seed{seed}.csv")
        seed_rows = read_csv(HERE / f"seed{seed}/epoch_metrics.csv")
        if len(seed_rows) != 1200:
            raise RuntimeError((seed, len(seed_rows)))
        for source in seed_rows:
            row = {"seed": int(source["seed"]), "epoch": int(source["epoch"])}
            row.update({key: float(value) for key, value in source.items() if key not in ("seed", "epoch")})
            all_rows.append(row)
        runtimes.append(json.loads((HERE / f"seed{seed}/runtime.json").read_text()))
    selected, frontier = choose_balanced(all_rows)
    write_csv(HERE / "checkpoint_pareto.csv", frontier)
    source_checkpoint = HERE / f"seed{selected['seed']}/checkpoints/epoch_{selected['epoch']:04d}.npz"
    checkpoint = HERE / "best_fixed_d_eta_checkpoint.npz"
    shutil.copy2(source_checkpoint, checkpoint)
    checkpoint_hash = sha256(checkpoint)
    selection = {
        "selection_frozen_before_test_or_strict_deadlock_evaluation": True,
        "test_used_for_selection": False,
        "strict_deadlock_outcomes_used_for_selection": False,
        "fresh6_used_for_selection": False,
        "selected_seed": selected["seed"],
        "selected_epoch": selected["epoch"],
        "selected_validation_metrics": selected,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "source_checkpoint": str(source_checkpoint),
        "pareto_checkpoint_count": len(frontier),
    }
    write_json(HERE / "selected_checkpoint.json", selection)

    # Only after selection is frozen do we inspect test or strict-deadlock cohorts.
    with np.load(DATASET / "samples.npz", allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]).copy() for key in source.files}
    with np.load(HERE / "eta_targets.npz", allow_pickle=False) as source:
        target_data = {key: np.asarray(source[key]).copy() for key in source.files}
    model = FixedDEtaPredictor(checkpoint)
    prediction, raw, _ = model.predict(arrays["features"])
    offline = {"selection": selection, "cohorts": {}}
    for split_name in ("validation", "test"):
        base_mask = arrays["split"] == split_name
        index = np.arange(len(base_mask))
        cohorts = {
            "ALL": base_mask,
            "STARTUP": base_mask & (index >= V3_COUNT) & (index < BASE_COUNT),
            "WARM_V3": base_mask & (index < V3_COUNT),
        }
        for cohort, mask in cohorts.items():
            offline["cohorts"][f"{split_name}_{cohort}"] = metrics(
                prediction[mask], raw[mask], target_data["eta_physical"][mask],
                target_data["eta_normalized"][mask], arrays["state_id"][mask],
            )
    zero_test = (arrays["split"] == "test") & np.all(np.isclose(target_data["eta_physical"], 0), axis=1)
    offline["heldout_zero_eta"] = {
        "samples": int(zero_test.sum()),
        "states": int(len(set(arrays["state_id"][zero_test].tolist()))),
        "predicted_eta_norm_mean": float(np.linalg.norm(prediction[zero_test], axis=1).mean()),
        "predicted_eta_norm_p95": float(np.quantile(np.linalg.norm(prediction[zero_test], axis=1), .95)),
    }

    capacity_manifest = json.loads((CAPACITY / "strict_deadlock_manifest.json").read_text())
    config = Config(**capacity_manifest["environment"])
    cbf = CBFConfig(**capacity_manifest["cbf"])
    states, robust_records, _ = load_capacity_states()
    historical = [row for row in states if row["benchmark"] == "historical"]
    fresh = [row for row in states if row["benchmark"] == "fresh_unseen"]
    _, samples_by_state = build_samples(states, robust_records, config, cbf)
    historical_rows = state_prediction_rows(model, historical, samples_by_state, "historical11_training_diagnostic")
    fresh_rows = state_prediction_rows(model, fresh, samples_by_state, "fresh6_heldout_diagnostic")
    write_csv(HERE / "historical11_eta_predictions.csv", historical_rows)
    write_csv(HERE / "fresh6_eta_predictions.csv", fresh_rows)
    offline["historical11"] = {
        "physical_eta_l2_mean_across_states": float(np.mean([row["physical_eta_l2_mean"] for row in historical_rows])),
        "clipping_fraction_mean": float(np.mean([row["clipping_fraction"] for row in historical_rows])),
    }
    offline["fresh6"] = {
        "physical_eta_l2_mean_across_states": float(np.mean([row["physical_eta_l2_mean"] for row in fresh_rows])),
        "clipping_fraction_mean": float(np.mean([row["clipping_fraction"] for row in fresh_rows])),
    }
    write_json(HERE / "offline_eta_metrics.json", offline)

    stability = []
    for state_id in dict.fromkeys(arrays["state_id"].tolist()):
        mask = arrays["state_id"] == state_id
        values = prediction[mask]
        pairwise = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
        stability.append({
            "cohort": "training_dataset",
            "state_id": state_id,
            "split": str(arrays["split"][np.flatnonzero(mask)[0]]),
            "variants": len(values),
            "eta1_mean": values[:, 0].mean(), "eta2_mean": values[:, 1].mean(), "eta3_mean": values[:, 2].mean(),
            "eta1_std": values[:, 0].std(), "eta2_std": values[:, 1].std(), "eta3_std": values[:, 2].std(),
            "max_pairwise_eta_distance": pairwise.max(),
        })
    for item in fresh:
        features = np.stack([row["feature"] for row in samples_by_state[item["state_id"]]])
        values, _, _ = model.predict(features)
        pairwise = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
        stability.append({
            "cohort": "fresh6_heldout", "state_id": item["state_id"], "split": "heldout",
            "variants": len(values), "eta1_mean": values[:, 0].mean(), "eta2_mean": values[:, 1].mean(),
            "eta3_mean": values[:, 2].mean(), "eta1_std": values[:, 0].std(),
            "eta2_std": values[:, 1].std(), "eta3_std": values[:, 2].std(),
            "max_pairwise_eta_distance": pairwise.max(),
        })
    write_csv(HERE / "eta_flow_stability.csv", stability)

    train_report = f"""# Fixed-D structured eta training\n\nA single 214 -> 128 -> 128 -> 3 SiLU network (44,419 parameters) was trained from scratch with normalized eta MSE on the exact 27,136-sample coverage dataset.  The direct-g comparator has 44,548 parameters.  No state, target, split, weighting, or optimizer setting was changed.\n\n## Validation-only selection\n\n- Seed: {selected['seed']}\n- Epoch: {selected['epoch']}\n- Checkpoint: `{checkpoint}`\n- SHA256: `{checkpoint_hash}`\n- Startup validation normalized-eta mean L2: {selected['startup_validation_state_grouped_mean_l2']:.8f}\n- Warm validation normalized-eta mean L2: {selected['warm_validation_state_grouped_mean_l2']:.8f}\n- Aggregate validation normalized-eta mean L2: {selected['all_validation_state_grouped_mean_l2']:.8f}\n\nThe checkpoint was frozen before test, historical-11, fresh-6, or closed-loop evaluation.\n"""
    (HERE / "training_report.md").write_text(train_report)
    write_json(HERE / "training_runtime.json", {
        "seed_runs": runtimes,
        "parallel_training_wall_seconds": max(row["wall_seconds"] for row in runtimes),
        "sum_seed_training_seconds": sum(row["wall_seconds"] for row in runtimes),
        "aggregation_offline_seconds": time.monotonic() - started,
        "gpu_shards": 3,
        "allocated_cpu_cores": 6,
        "allocated_memory_gb": 43,
    })
    print(json.dumps({
        "selected_seed": selected["seed"], "selected_epoch": selected["epoch"],
        "checkpoint_sha256": checkpoint_hash,
        "test": offline["cohorts"]["test_ALL"], "fresh6": offline["fresh6"],
    }, indent=2))


if __name__ == "__main__":
    main()
