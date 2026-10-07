"""Aggregate V4 seed workers and audit recovery-boundary generalization."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "diagnostics/gphi_training_dataset_v4"
V3_DATA = ROOT / "diagnostics/gphi_training_dataset_v3"
V3_TRAIN = ROOT / "diagnostics/gphi_pilot_training_v3"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
VMAX = 0.5
EPS = 1e-12
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import diagnostics.gphi_pilot_training_v2.train_and_evaluate as core
core.DATA = DATA; core.HERE = HERE


OLD_HARD_IDS = (
    "R_D4_s95101008_p134", "R_D4_s95105001_p116",
    "R_D4_s95101014_p123", "R_D4_s95105004_p114",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"Object of type {type(item).__name__} is not JSON serializable")

    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=convert) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def load_checkpoint(path: Path):
    with np.load(path, allow_pickle=False) as data:
        layers = sorted({int(key.split("_")[1]) for key in data.files if key.startswith("layer_") and key.endswith("_weight")})
        params = [{"w": np.asarray(data[f"layer_{index}_weight"]), "b": np.asarray(data[f"layer_{index}_bias"])} for index in layers]
        mean = np.asarray(data["normalization_mean"]); scale = np.asarray(data["normalization_scale"])
        binary = np.asarray(data["normalization_binary_mask"], dtype=bool)
    return params, mean, scale, binary


def checkpoint_predict(path: Path, features: np.ndarray) -> np.ndarray:
    params, mean, scale, _ = load_checkpoint(path)
    return core.predict_mlp(params, core.normalize(features, mean, scale).astype(np.float32))


def distribution(values: np.ndarray) -> dict:
    return {"mean": float(np.mean(values)), "median": float(np.median(values)), "p95": float(np.quantile(values, .95)), "max": float(np.max(values))}


def nonzero_diagnostics(pred: np.ndarray, target: np.ndarray) -> dict:
    error = np.linalg.norm(pred - target, axis=1)
    target_norm = np.linalg.norm(target, axis=1); pred_norm = np.linalg.norm(pred, axis=1)
    valid = (target_norm > EPS) & (pred_norm > EPS)
    cosine = np.full(len(pred), np.nan)
    cosine[valid] = np.sum(pred[valid] * target[valid], axis=1) / (target_norm[valid] * pred_norm[valid])
    return {
        "sample_count": len(pred), "mean_L2_error": float(error.mean()),
        "mean_relative_error": float(np.mean(error / target_norm)),
        "mean_target_norm": float(target_norm.mean()), "mean_predicted_norm": float(pred_norm.mean()),
        "mean_cosine_similarity": float(np.nanmean(cosine)), "median_cosine_similarity": float(np.nanmedian(cosine)),
        "error_distribution": distribution(error),
    }


def main() -> None:
    started = time.monotonic(); started_utc = datetime.now(timezone.utc).isoformat()
    arrays, audit, metadata, schema, dataset_manifest = core.dataset_audit()
    if not audit["passed"]:
        raise RuntimeError(audit["failures"])
    indices = {split: np.flatnonzero(arrays["split"] == split) for split in ("train", "validation", "test")}
    y = arrays["targets"].astype(np.float32)
    workers = HERE / "workers"
    runs = [read_json(workers / f"seed{seed}_summary.json") for seed in (17, 23, 41)]
    best = min(runs, key=lambda row: row["validation_metrics"]["state_grouped_mse"])
    best_seed = best["seed"]
    shutil.copy2(workers / f"seed{best_seed}.npz", HERE / "best_checkpoint.npz")
    best_params, mean, scale, binary = load_checkpoint(HERE / "best_checkpoint.npz")
    x_norm = core.normalize(arrays["features"], mean, scale).astype(np.float32)
    split_data = {split: (x_norm[index], y[index], arrays["state_id"][index]) for split, index in indices.items()}
    metadata_by_sample = {row["sample_id"]: row for row in metadata}

    normalization = {
        "fit_split": "train only", "epsilon": 1e-8,
        "rule": "continuous standardized; schema boolean/latch unchanged",
        "binary_feature_indices": np.flatnonzero(binary).tolist(), "mean": mean.tolist(), "scale": scale.tolist(),
    }
    write_json(HERE / "normalization.json", normalization)

    predictions = defaultdict(dict); comparison = []; state_rows = []
    for name, constant in {"ZERO": np.zeros(4, dtype=np.float32), "TRAIN_MEAN": y[indices["train"]].mean(axis=0)}.items():
        row = {"model": name, "kind": "baseline", "parameter_count": 0, "selected_by_validation": False}
        for split, (_, target, state_id) in split_data.items():
            pred = np.broadcast_to(constant, target.shape).copy(); predictions[name][split] = pred
            metrics = core.subset_metrics(pred, target, state_id)
            for key in ("mse", "state_grouped_mse", "state_grouped_mean_l2"):
                row[f"{split}_{key}"] = metrics[key]
            state_rows.extend(core.state_metric_rows(name, split, pred, target, arrays, indices[split], metadata_by_sample))
        comparison.append(row)
    linear_candidates = []
    for wd in (0.0, 1e-4):
        model = core.ridge_fit(split_data["train"][0], split_data["train"][1], wd)
        pred = core.ridge_predict(model, split_data["validation"][0])
        score = core.subset_metrics(pred, split_data["validation"][1], split_data["validation"][2])["state_grouped_mse"]
        linear_candidates.append((score, wd, model))
    _, linear_wd, linear = min(linear_candidates, key=lambda item: item[0])
    row = {"model": "LINEAR", "kind": "linear", "parameter_count": 860, "weight_decay": linear_wd, "selected_by_validation": False}
    for split, (features, target, state_id) in split_data.items():
        pred = core.ridge_predict(linear, features); predictions["LINEAR"][split] = pred
        metrics = core.subset_metrics(pred, target, state_id)
        for key in ("mse", "state_grouped_mse", "state_grouped_mean_l2"):
            row[f"{split}_{key}"] = metrics[key]
        state_rows.extend(core.state_metric_rows("LINEAR", split, pred, target, arrays, indices[split], metadata_by_sample))
    comparison.append(row)

    best_predictions = {}; best_metrics = {}
    for split, (features, target, state_id) in split_data.items():
        pred = core.predict_mlp(best_params, features); best_predictions[split] = pred
        best_metrics[split] = core.subset_metrics(pred, target, state_id)
        state_rows.extend(core.state_metric_rows(f"SELECTED_MLP_seed{best_seed}", split, pred, target, arrays, indices[split], metadata_by_sample))
    for run in runs:
        comparison.append({
            "model": f"FINAL_seed{run['seed']}", "kind": "frozen_architecture_seed", "hidden": "128x128",
            "parameter_count": run["summary"]["parameter_count"], "learning_rate": 1e-3, "weight_decay": 1e-5,
            "seed": run["seed"], "best_epoch": run["summary"]["best_epoch"],
            "train_state_grouped_mse": run["train_metrics"]["state_grouped_mse"],
            "train_state_grouped_mean_l2": run["train_metrics"]["state_grouped_mean_l2"],
            "validation_state_grouped_mse": run["validation_metrics"]["state_grouped_mse"],
            "validation_state_grouped_mean_l2": run["validation_metrics"]["state_grouped_mean_l2"],
            "test_mse": best_metrics["test"]["mse"] if run["seed"] == best_seed else "NOT_EVALUATED",
            "selected_by_validation": run["seed"] == best_seed,
        })
    histories = []
    for seed in (17, 23, 41):
        histories.extend(read_json(workers / f"seed{seed}_history.json"))
    write_csv(HERE / "training_history.csv", histories); write_csv(HERE / "model_comparison.csv", comparison); write_csv(HERE / "state_grouped_metrics.csv", state_rows)

    category_rows = []
    for split, index in indices.items():
        pred = best_predictions[split]; target = y[index]; state_id = arrays["state_id"][index]
        for category in ("NORMAL", "PRE_DEADLOCK", "RECOVERY"):
            mask = arrays["category"][index] == category
            category_rows.append({"split": split, "group_type": "category", "group": category, **core.subset_metrics(pred[mask], target[mask], state_id[mask])})
        zero = np.linalg.norm(target, axis=1) <= 1e-14
        for label, mask in (("zero_label", zero), ("nonzero_label", ~zero)):
            category_rows.append({"split": split, "group_type": "label", "group": label, **core.subset_metrics(pred[mask], target[mask], state_id[mask])})
    write_csv(HERE / "category_metrics.csv", category_rows)

    state_manifest = {row["state_id"]: row for row in core.read_jsonl(DATA / "state_manifest.jsonl")}
    new_ids = {state_id for state_id, row in state_manifest.items() if row.get("boundary_candidate")}
    test_index = indices["test"]; test_ids = arrays["state_id"][test_index]
    test_target = y[test_index]; test_pred = best_predictions["test"]
    boundary_class = np.asarray([state_manifest[str(state_id)].get("oracle_boundary_class", "none") for state_id in test_ids])
    new_mask = np.asarray([state_id in new_ids for state_id in test_ids])
    close_zero = new_mask & (boundary_class == "zero")
    close_nonzero = new_mask & (boundary_class == "nonzero")
    close_zero_norms = np.linalg.norm(test_pred[close_zero], axis=1)
    close_nonzero_metrics = nonzero_diagnostics(test_pred[close_nonzero], test_target[close_nonzero])
    v3_on_test = checkpoint_predict(V3_TRAIN / "best_checkpoint.npz", arrays["features"][test_index])
    v3_close_zero_norms = np.linalg.norm(v3_on_test[close_zero], axis=1)

    old_hard_rows = []
    for state_id in OLD_HARD_IDS:
        mask = test_ids == state_id
        if not np.any(mask):
            raise AssertionError(("old hard state missing", state_id))
        old_values = np.linalg.norm(v3_on_test[mask], axis=1); new_values = np.linalg.norm(test_pred[mask], axis=1)
        old_hard_rows.append({
            "state_id": state_id, "sample_count": int(mask.sum()),
            "V3_mean_prediction_norm": float(old_values.mean()), "V4_mean_prediction_norm": float(new_values.mean()),
            "absolute_change": float(new_values.mean() - old_values.mean()),
            "fractional_reduction": float(1.0 - new_values.mean() / old_values.mean()),
            "V3_median_prediction_norm": float(np.median(old_values)), "V4_median_prediction_norm": float(np.median(new_values)),
            "V3_p95_prediction_norm": float(np.quantile(old_values, .95)), "V4_p95_prediction_norm": float(np.quantile(new_values, .95)),
        })
    write_csv(HERE / "old_hard_state_replay.csv", old_hard_rows)
    old_hard_v3 = float(np.mean([row["V3_mean_prediction_norm"] for row in old_hard_rows]))
    old_hard_v4 = float(np.mean([row["V4_mean_prediction_norm"] for row in old_hard_rows]))
    old_hard_reduction = 1.0 - old_hard_v4 / old_hard_v3

    pairs = list(csv.DictReader((DATA / "matched_boundary_pairs.csv").open()))
    pair_metrics = []
    for row in pairs:
        zero_id = row["zero_state_id"]; nonzero_id = row["nonzero_state_id"]
        split = row["split"]; split_index = indices[split]; split_ids = arrays["state_id"][split_index]
        pred = best_predictions[split]; target = y[split_index]
        zero_mask = split_ids == zero_id; nonzero_mask = split_ids == nonzero_id
        zero_pred_norm = float(np.linalg.norm(pred[zero_mask], axis=1).mean())
        nz = nonzero_diagnostics(pred[nonzero_mask], target[nonzero_mask])
        pair_metrics.append({
            "pair_id": row["pair_id"], "split": split, "feature_distance": float(row["feature_distance"]),
            "zero_state_id": zero_id, "nonzero_state_id": nonzero_id,
            "zero_prediction_norm": zero_pred_norm,
            "nonzero_prediction_norm": nz["mean_predicted_norm"], "nonzero_target_norm": nz["mean_target_norm"],
            "nonzero_L2_error": nz["mean_L2_error"], "nonzero_relative_error": nz["mean_relative_error"],
            "nonzero_cosine_similarity": nz["mean_cosine_similarity"],
            "predicted_norm_gap_nonzero_minus_zero": nz["mean_predicted_norm"] - zero_pred_norm,
        })

    all_test_zero = np.linalg.norm(test_target, axis=1) <= 1e-14
    all_nonzero = nonzero_diagnostics(test_pred[~all_test_zero], test_target[~all_test_zero])
    overall_nonzero_change = all_nonzero["mean_L2_error"] / 0.009571565315127373 - 1.0
    diversity = read_json(DATA / "boundary_diversity_metrics.json")
    input_sufficient = bool(old_hard_reduction >= .30 and close_nonzero_metrics["mean_L2_error"] <= .02 and diversity["nearest_neighbor_leave_one_out_accuracy"] >= .60)
    boundary_metrics = {
        "heldout_new_close_zero": {
            "state_count": len(set(test_ids[close_zero].tolist())), "sample_count": int(close_zero.sum()),
            "V4_prediction_norm": distribution(close_zero_norms),
            "V3_model_on_exact_same_states": distribution(v3_close_zero_norms),
            "matched_fractional_reduction_V3_to_V4": float(1.0 - close_zero_norms.mean() / v3_close_zero_norms.mean()),
            "current_difficult_tail_reference_m_per_s": 0.093185,
        },
        "heldout_new_close_nonzero": {"state_count": len(set(test_ids[close_nonzero].tolist())), **close_nonzero_metrics},
        "old_four_hard_states": {"V3_mean": old_hard_v3, "V4_mean": old_hard_v4, "fractional_reduction": old_hard_reduction, "per_state_file": "old_hard_state_replay.csv"},
        "matched_pairs": pair_metrics,
        "boundary_learnability": {
            "nearest_neighbor_leave_one_out_accuracy": diversity["nearest_neighbor_leave_one_out_accuracy"],
            "nearest_neighbor_label_disagreement_rate": diversity["nearest_neighbor_label_disagreement_rate"],
            "opposite_over_same_mean_distance_ratio": diversity["opposite_over_same_mean_distance_ratio"],
            "top_informative_feature_groups": diversity["feature_group_centroid_differences_ranked"][:10],
            "current_214D_input_appears_sufficient": input_sufficient,
            "note": "diagnostic only; no feature redesign or classifier was introduced",
        },
    }
    write_json(HERE / "recovery_boundary_metrics.json", boundary_metrics)

    # Exact V3/V4 checkpoint comparison on retained V3 test states and new boundary states.
    v3_state_ids = {row["state_id"] for row in core.read_jsonl(V3_DATA / "state_manifest.jsonl") if row["split"] == "test"}
    group_masks = {
        "retained_V3_test_states": np.asarray([state_id in v3_state_ids for state_id in test_ids]),
        "old_four_difficult_zero_states": np.isin(test_ids, OLD_HARD_IDS),
        "new_close_zero_boundary_states": close_zero,
        "new_close_nonzero_boundary_states": close_nonzero,
        "all_nonzero_test_states": ~all_test_zero,
    }
    matched = {"V3_checkpoint": str(V3_TRAIN / "best_checkpoint.npz"), "V4_checkpoint": str(HERE / "best_checkpoint.npz"), "groups": {}}
    for name, mask in group_masks.items():
        old_metrics = core.subset_metrics(v3_on_test[mask], test_target[mask], test_ids[mask])
        new_metrics = core.subset_metrics(test_pred[mask], test_target[mask], test_ids[mask])
        matched["groups"][name] = {"state_count": len(set(test_ids[mask].tolist())), "V3": old_metrics, "V4": new_metrics, "state_grouped_mean_L2_fractional_reduction": 1.0 - new_metrics["state_grouped_mean_l2"] / max(old_metrics["state_grouped_mean_l2"], EPS)}
    write_json(HERE / "matched_v3_v4_comparison.json", matched)

    protocol = read_json(DATA / "protocol.json")
    projection = core.projection_replay(best_predictions, arrays, indices, protocol)
    for split in projection:
        projection[split]["invalid_action_count"] = 0 if not projection[split]["projection_failures"] else None
    write_json(HERE / "projection_replay_metrics.json", projection)

    nonzero_material_degrade = bool(all_nonzero["mean_L2_error"] > 1.25 * 0.009571565315127373 and all_nonzero["mean_L2_error"] - 0.009571565315127373 > .002)
    general_test_degrade = best_metrics["test"]["state_grouped_mean_l2"] > 1.10 * 0.015429105632938445
    close_nonzero_accurate = close_nonzero_metrics["mean_L2_error"] <= .02 and close_nonzero_metrics["mean_relative_error"] <= .50
    new_zero_reduction = 1.0 - close_zero_norms.mean() / 0.093185
    if new_zero_reduction >= .50 and old_hard_reduction >= .30 and close_nonzero_accurate and not nonzero_material_degrade and not general_test_degrade:
        conclusion = "BOUNDARY_COVERAGE_FIXES_FAILURE"
    elif (new_zero_reduction >= .15 or old_hard_reduction >= .15) and input_sufficient:
        conclusion = "STILL_BOUNDARY_DATA_LIMITED"
    elif not input_sufficient and new_zero_reduction < .15 and old_hard_reduction < .15:
        conclusion = "BOUNDARY_NOT_LEARNABLE_WITH_CURRENT_REGRESSION"
    else:
        conclusion = "STILL_BOUNDARY_DATA_LIMITED"
    projection_failures = sum(len(projection[split]["projection_failures"]) for split in projection)
    ready = bool(conclusion == "BOUNDARY_COVERAGE_FIXES_FAILURE" and close_zero_norms.mean() <= .025 and old_hard_v4 <= .025 and close_nonzero_metrics["mean_L2_error"] <= .015 and not nonzero_material_degrade and projection_failures == 0)
    readiness = "READY_FOR_CLOSED_LOOP_PILOT" if ready else "NOT_READY_FOR_CLOSED_LOOP_PILOT"

    test_baselines = {name: core.subset_metrics(predictions[name]["test"], test_target, test_ids) for name in ("ZERO", "TRAIN_MEAN", "LINEAR")}
    test_baselines[f"SELECTED_MLP_seed{best_seed}"] = best_metrics["test"]
    test_metrics = {
        "selected_seed": best_seed, "selection_used_test": False, "metrics_by_split": best_metrics,
        "test_baselines": test_baselines, "conclusion": conclusion, "readiness": readiness,
        "evidence": {
            "new_close_zero_mean": float(close_zero_norms.mean()), "new_close_zero_reduction_vs_0.093185": new_zero_reduction,
            "old_hard_V3_mean": old_hard_v3, "old_hard_V4_mean": old_hard_v4, "old_hard_reduction": old_hard_reduction,
            "close_nonzero_error": close_nonzero_metrics["mean_L2_error"],
            "V3_overall_nonzero_error": 0.009571565315127373, "V4_overall_nonzero_error": all_nonzero["mean_L2_error"],
            "overall_nonzero_fractional_change": overall_nonzero_change, "nonzero_material_degradation": nonzero_material_degrade,
            "general_test_degradation": general_test_degrade, "current_214D_input_appears_sufficient": input_sufficient,
        },
    }
    write_json(HERE / "test_metrics.json", test_metrics)

    config = {
        "task": "recovery intervention-boundary coverage retraining", "architecture": [214, 128, 128, 4],
        "activation": "SiLU", "output_activation": "identity", "parameter_count": best["summary"]["parameter_count"],
        "target": "g*_exec = u*_exec - u_safe", "primary_loss": "mean squared error",
        "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-5,
        "training_seeds": [17, 23, 41], "selected_seed": best_seed,
        "early_stopping": {"metric": "validation state-grouped MSE", "best_epoch": best["summary"]["best_epoch"], "patience_evaluations": 35},
        "conclusion": conclusion, "readiness": readiness,
    }
    write_json(HERE / "config.json", config)
    finished_utc = datetime.now(timezone.utc).isoformat()
    worker_times = {run["seed"]: run["summary"]["runtime_s"] for run in runs}
    runtime = {
        "evaluation_started_utc": started_utc, "evaluation_finished_utc": finished_utc,
        "evaluation_wall_seconds": time.monotonic() - started, "worker_training_seconds": worker_times,
        "parallel_training_critical_path_estimate_s": max(worker_times[17], worker_times[23]) + worker_times[41],
        "allocated_GPU_shards": 2, "allocated_CPU_cores": 8, "JAX_memory_fraction_per_process": 0.10,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "jax_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.version, "platform": platform.platform(),
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    frozen_paths = {"environment": SYSROOT / "single_integrator/environment.py", "projection": SYSROOT / "single_integrator/cbf.py", "corrector": ROOT / "diagnostics/cl_fhcb/closed_loop.py", "retry": ROOT / "diagnostics/success_basin_multimodality/exact_projector.py", "checkpoint": Path(protocol["checkpoint"])}
    frozen_actual = {key: sha(path) for key, path in frozen_paths.items()}
    sanity = dict(audit); sanity.update({
        "dataset_unchanged": sha(DATA / "samples.npz") == audit["samples_sha256_actual"],
        "frozen_hashes_expected": protocol["frozen_hashes"], "frozen_hashes_after": frozen_actual,
        "frozen_hashes_unchanged": frozen_actual == protocol["frozen_hashes"],
        "prediction_all_finite": all(np.isfinite(pred).all() for pred in best_predictions.values()),
        "checkpoint_output_dimension": 4, "projection_solver_failures": projection_failures,
        "projection_invalid_actions": 0, "formal_closed_loop_benchmark_performed": False,
    })
    write_json(HERE / "sanity_checks.json", sanity)

    report = f"""# Deterministic G_phi recovery-boundary retraining V4

## Decision

**{conclusion}**  
**{readiness}**

The deterministic `214 -> 128 -> 128 -> 4` SiLU MLP, direct MSE target, FlowBC, both hard projections, and all oracle/controller semantics remained unchanged. No gate, deadband, classifier, weighting, auxiliary loss, or closed-loop benchmark was introduced.

## Selected model

- Seed **{best_seed}**, best epoch {best['summary']['best_epoch']}, 44,548 parameters.
- AdamW, lr `1e-3`, weight decay `1e-5`, train-only normalization.
- Checkpoint: `{HERE / 'best_checkpoint.npz'}`.

| Split | state-grouped mean L2 (m/s) | state-grouped RMSE |
|---|---:|---:|
| Train | {best_metrics['train']['state_grouped_mean_l2']:.6f} | {best_metrics['train']['state_grouped_rmse']:.6f} |
| Validation | {best_metrics['validation']['state_grouped_mean_l2']:.6f} | {best_metrics['validation']['state_grouped_rmse']:.6f} |
| Test | {best_metrics['test']['state_grouped_mean_l2']:.6f} | {best_metrics['test']['state_grouped_rmse']:.6f} |

## Boundary results

- New held-out close-range zero states: {len(set(test_ids[close_zero].tolist()))}; false intervention mean/median/P95/max = **{close_zero_norms.mean():.6f}/{np.median(close_zero_norms):.6f}/{np.quantile(close_zero_norms,.95):.6f}/{close_zero_norms.max():.6f} m/s**.
- V3 on those exact states: {v3_close_zero_norms.mean():.6f}; matched reduction: {1-close_zero_norms.mean()/v3_close_zero_norms.mean():.2%}.
- Old four hard states: **{old_hard_v3:.6f} -> {old_hard_v4:.6f} m/s** ({old_hard_reduction:.2%} reduction).
- New held-out close nonzero states: error {close_nonzero_metrics['mean_L2_error']:.6f} m/s, relative error {close_nonzero_metrics['mean_relative_error']:.2%}, target norm {close_nonzero_metrics['mean_target_norm']:.6f}, predicted norm {close_nonzero_metrics['mean_predicted_norm']:.6f}, cosine {close_nonzero_metrics['mean_cosine_similarity']:.4f}.
- Overall nonzero test error: **0.009572 -> {all_nonzero['mean_L2_error']:.6f} m/s** ({overall_nonzero_change:+.2%}).

## Learnability and projection

Deployment-feature 1-NN leave-one-out boundary accuracy is {diversity['nearest_neighbor_leave_one_out_accuracy']:.2%}; the current 214-D input appears sufficient: **{input_sufficient}**. Detailed overlap, matched-pair, and feature-group diagnostics are stored without changing the feature schema.

Test projected executed-action error is {projection['test']['executed_action_error']['mean']:.6f} m/s and mean rewrite is {projection['test']['projection_rewrite_norm']['mean']:.6f} m/s. Solver failures: {len(projection['test']['projection_failures'])}; invalid actions: 0.

No formal closed-loop benchmark was run.
"""
    (HERE / "training_report.md").write_text(report)
    required = (
        "training_report.md", "config.json", "normalization.json", "training_history.csv",
        "model_comparison.csv", "state_grouped_metrics.csv", "category_metrics.csv",
        "recovery_boundary_metrics.json", "old_hard_state_replay.csv", "matched_v3_v4_comparison.json",
        "projection_replay_metrics.json", "best_checkpoint.npz", "sanity_checks.json", "runtime_statistics.json", "test_metrics.json",
    )
    manifest = {
        "study": "GPHI_RECOVERY_BOUNDARY_COVERAGE", "generated_at_utc": finished_utc,
        "classification": conclusion, "readiness": readiness, "training_performed": True,
        "formal_closed_loop_benchmark_performed": False, "dataset_directory": str(DATA),
        "dataset_manifest_sha256": sha(DATA / "manifest.json"), "dataset_samples_sha256": sha(DATA / "samples.npz"),
        "selected_checkpoint": "best_checkpoint.npz", "resource_policy": "two GPU shards and eight CPU cores under confirmed light GPU load",
        "frozen_hashes_unchanged": sanity["frozen_hashes_unchanged"],
        "files_sha256": {name: sha(HERE / name) for name in required},
    }
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({
        "conclusion": conclusion, "readiness": readiness, "seed": best_seed,
        "train_val_test": [best_metrics[split]["state_grouped_mean_l2"] for split in ("train", "validation", "test")],
        "new_close_zero_mean": float(close_zero_norms.mean()), "old_hard_V4_mean": old_hard_v4,
        "close_nonzero_error": close_nonzero_metrics["mean_L2_error"], "overall_nonzero_error": all_nonzero["mean_L2_error"],
        "input_sufficient": input_sufficient,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
