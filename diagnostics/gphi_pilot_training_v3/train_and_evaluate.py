"""Retrain the frozen deterministic G_phi on targeted RECOVERY-zero V3 data."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "diagnostics/gphi_training_dataset_v3"
V2_DATA = ROOT / "diagnostics/gphi_training_dataset_v2"
V2_TRAIN = ROOT / "diagnostics/gphi_pilot_training_v2"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
VMAX = 0.5
EPS = 1e-12
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import diagnostics.gphi_pilot_training_v2.train_and_evaluate as core

# The audited V2 helpers are formulation-generic; redirect only their I/O roots.
core.DATA = DATA
core.HERE = HERE


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path: Path):
    with np.load(path, allow_pickle=False) as data:
        layers = sorted({int(key.split("_")[1]) for key in data.files if key.startswith("layer_") and key.endswith("_weight")})
        params = [{"w": np.asarray(data[f"layer_{index}_weight"]), "b": np.asarray(data[f"layer_{index}_bias"])} for index in layers]
        mean = np.asarray(data["normalization_mean"])
        scale = np.asarray(data["normalization_scale"])
    return params, mean, scale


def prediction_from_checkpoint(path: Path, features: np.ndarray) -> np.ndarray:
    params, mean, scale = load_checkpoint(path)
    return core.predict_mlp(params, core.normalize(features, mean, scale).astype(np.float32))


def state_ids_in_order(rows: list[dict]) -> list[str]:
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["leakage_group"]].append(row["state_id"])
    for group in by_group:
        by_group[group].sort(key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    result = []
    while any(by_group.values()):
        for group in sorted(by_group):
            if by_group[group]:
                result.append(by_group[group].pop(0))
    return result


def group_metrics(pred, target, state_id):
    return core.subset_metrics(pred, target, state_id)


def norm_distribution(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)), "max": float(np.max(values)),
    }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    arrays, audit, metadata, schema, dataset_manifest = core.dataset_audit()
    if not audit["passed"]:
        raise SystemExit("dataset audit failed: " + "; ".join(audit["failures"]))
    write_json(HERE / "sanity_checks.json", audit)

    import jax
    import flax
    import optax

    print(json.dumps({
        "stage": "frozen deterministic G_phi V3 retraining",
        "samples": len(arrays["targets"]), "states": audit["unique_states"],
        "device": [str(x) for x in jax.devices()],
        "models": ["ZERO", "TRAIN_MEAN", "LINEAR", "MLP_128x128"],
        "selection": "validation state-grouped MSE only",
    }), flush=True)

    indices_by_split = {split: np.flatnonzero(arrays["split"] == split) for split in ("train", "validation", "test")}
    train_indices = indices_by_split["train"]
    mean, scale, binary = core.fit_normalization(arrays["features"][train_indices], schema)
    x_norm = core.normalize(arrays["features"], mean, scale).astype(np.float32)
    y = arrays["targets"].astype(np.float32)
    normalization = {
        "fit_split": "train only", "epsilon": 1e-8,
        "rule": "continuous: (x-mu)/max(sigma,eps); schema boolean/latch unchanged",
        "binary_feature_indices": np.flatnonzero(binary).tolist(),
        "constant_continuous_feature_indices": np.flatnonzero((scale == 1.0) & (~binary) & (arrays["features"][train_indices].std(axis=0) < 1e-8)).tolist(),
        "mean": mean.tolist(), "scale": scale.tolist(),
    }
    write_json(HERE / "normalization.json", normalization)
    split_data = {
        split: (x_norm[indices], y[indices], arrays["state_id"][indices])
        for split, indices in indices_by_split.items()
    }
    metadata_by_sample = {row["sample_id"]: row for row in metadata}

    predictions = defaultdict(dict)
    comparison = []
    state_rows = []
    for name, constant in {
        "ZERO": np.zeros(4, dtype=np.float32),
        "TRAIN_MEAN": y[train_indices].mean(axis=0),
    }.items():
        row = {"model": name, "kind": "baseline", "parameter_count": 0, "selected_by_validation": False}
        for split, (_, target, state_id) in split_data.items():
            prediction = np.broadcast_to(constant, target.shape).copy()
            predictions[name][split] = prediction
            metrics = group_metrics(prediction, target, state_id)
            for key in ("mse", "state_grouped_mse", "state_grouped_mean_l2"):
                row[f"{split}_{key}"] = metrics[key]
            state_rows.extend(core.state_metric_rows(name, split, prediction, target, arrays, indices_by_split[split], metadata_by_sample))
        comparison.append(row)

    linear_candidates = []
    for wd in (0.0, 1e-4):
        model = core.ridge_fit(split_data["train"][0], split_data["train"][1], wd)
        val_prediction = core.ridge_predict(model, split_data["validation"][0])
        score = group_metrics(val_prediction, split_data["validation"][1], split_data["validation"][2])["state_grouped_mse"]
        linear_candidates.append((score, wd, model))
    _, linear_wd, linear = min(linear_candidates, key=lambda row: row[0])
    linear_row = {"model": "LINEAR", "kind": "linear", "parameter_count": 860, "weight_decay": linear_wd, "selected_by_validation": False}
    for split, (features, target, state_id) in split_data.items():
        prediction = core.ridge_predict(linear, features)
        predictions["LINEAR"][split] = prediction
        metrics = group_metrics(prediction, target, state_id)
        for key in ("mse", "state_grouped_mse", "state_grouped_mean_l2"):
            linear_row[f"{split}_{key}"] = metrics[key]
        state_rows.extend(core.state_metric_rows("LINEAR", split, prediction, target, arrays, indices_by_split[split], metadata_by_sample))
    comparison.append(linear_row)

    base_config = {
        "hidden": [128, 128], "learning_rate": 1e-3, "weight_decay": 1e-5,
        "batch_size": 256, "max_epochs": 1200, "eval_interval": 5,
        "patience_evaluations": 35, "min_delta": 1e-9,
    }
    histories = []
    final_runs = []
    checkpoints = HERE / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    for seed in (17, 23, 41):
        config = {**base_config, "name": f"FINAL_128x128_s{seed}", "seed": seed}
        params, history, summary = core.train_mlp(*split_data["train"], *split_data["validation"], config)
        histories.extend(history)
        train_prediction = core.predict_mlp(params, split_data["train"][0])
        val_prediction = core.predict_mlp(params, split_data["validation"][0])
        train_metrics = group_metrics(train_prediction, split_data["train"][1], split_data["train"][2])
        val_metrics = group_metrics(val_prediction, split_data["validation"][1], split_data["validation"][2])
        core.save_checkpoint(checkpoints / f"final_seed{seed}.npz", params, config, mean, scale, binary)
        final_runs.append({
            "seed": seed, "params": params, "config": config, "summary": summary,
            "train_metrics": train_metrics, "validation_metrics": val_metrics,
        })
        print(json.dumps({"seed": seed, "best_epoch": summary["best_epoch"], "validation_mean_l2": val_metrics["state_grouped_mean_l2"], "runtime_s": summary["runtime_s"]}), flush=True)

    best = min(final_runs, key=lambda row: row["validation_metrics"]["state_grouped_mse"])
    best_name = f"SELECTED_MLP_seed{best['seed']}"
    best_predictions = {}
    best_metrics = {}
    for split, (features, target, state_id) in split_data.items():
        prediction = core.predict_mlp(best["params"], features)
        best_predictions[split] = prediction
        best_metrics[split] = group_metrics(prediction, target, state_id)
        state_rows.extend(core.state_metric_rows(best_name, split, prediction, target, arrays, indices_by_split[split], metadata_by_sample))
    core.save_checkpoint(HERE / "best_checkpoint.npz", best["params"], best["config"], mean, scale, binary)

    for run in final_runs:
        comparison.append({
            "model": f"FINAL_seed{run['seed']}", "kind": "frozen_architecture_seed",
            "hidden": "128x128", "parameter_count": run["summary"]["parameter_count"],
            "learning_rate": 1e-3, "weight_decay": 1e-5, "seed": run["seed"],
            "best_epoch": run["summary"]["best_epoch"],
            "train_mse": run["train_metrics"]["mse"],
            "train_state_grouped_mse": run["train_metrics"]["state_grouped_mse"],
            "train_state_grouped_mean_l2": run["train_metrics"]["state_grouped_mean_l2"],
            "validation_mse": run["validation_metrics"]["mse"],
            "validation_state_grouped_mse": run["validation_metrics"]["state_grouped_mse"],
            "validation_state_grouped_mean_l2": run["validation_metrics"]["state_grouped_mean_l2"],
            "test_mse": best_metrics["test"]["mse"] if run is best else "NOT_EVALUATED",
            "selected_by_validation": run is best,
        })
    write_csv(HERE / "training_history.csv", histories)
    write_csv(HERE / "model_comparison.csv", comparison)
    write_csv(HERE / "state_grouped_metrics.csv", state_rows)

    category_rows = []
    for split, indices in indices_by_split.items():
        pred, target, sid = best_predictions[split], y[indices], arrays["state_id"][indices]
        for category in ("NORMAL", "PRE_DEADLOCK", "RECOVERY"):
            mask = arrays["category"][indices] == category
            category_rows.append({"split": split, "group_type": "category", "group": category, **group_metrics(pred[mask], target[mask], sid[mask])})
        zero = np.linalg.norm(target, axis=1) <= 1e-14
        for label, mask in (("zero_label", zero), ("nonzero_label", ~zero)):
            row = {"split": split, "group_type": "label", "group": label, **group_metrics(pred[mask], target[mask], sid[mask])}
            if label == "zero_label":
                norms = np.linalg.norm(pred[mask], axis=1)
                row.update(false_intervention_mean_norm=float(norms.mean()), false_intervention_median_norm=float(np.median(norms)), false_intervention_p95_norm=float(np.quantile(norms, 0.95)), false_intervention_max_norm=float(norms.max()))
            category_rows.append(row)
    write_csv(HERE / "category_metrics.csv", category_rows)

    test_indices = indices_by_split["test"]
    test_pred = best_predictions["test"]
    test_target = y[test_indices]
    test_sid = arrays["state_id"][test_indices]
    test_category = arrays["category"][test_indices]
    zero_test = np.linalg.norm(test_target, axis=1) <= 1e-14
    recovery_zero_test = zero_test & (test_category == "RECOVERY")
    recovery_norms = np.linalg.norm(test_pred[recovery_zero_test], axis=1)
    recovery_ids = test_sid[recovery_zero_test]
    recovery_state_means = np.asarray([recovery_norms[recovery_ids == state_id].mean() for state_id in sorted(set(recovery_ids.tolist()))])

    state_manifest = {row["state_id"]: row for row in core.read_jsonl(DATA / "state_manifest.jsonl")}
    per_recovery_state = []
    for state_id in sorted(set(recovery_ids.tolist())):
        local = recovery_ids == state_id
        global_index = test_indices[recovery_zero_test][local][0]
        positions = arrays["positions"][global_index]
        row = state_manifest[state_id]
        per_recovery_state.append({
            "state_id": state_id, "source_group": row["leakage_group"],
            "new_v3_state": not row.get("retained_from_v2", True),
            "recovery_phase": row.get("recovery_phase", "unspecified"),
            "planned_phase_fraction": row.get("planned_phase_fraction"),
            "mean_false_intervention": float(recovery_norms[local].mean()),
            "inter_agent_distance": float(np.linalg.norm(positions[0] - positions[1])),
            "stuck_timer": float(row.get("stuck_timer", 0.0)),
            "candidate_active": int(row.get("candidate_since", -1)) >= 0,
            "ever_candidate_deadlock": bool(row.get("ever_candidate_deadlock", False)),
        })
    cluster_rows = []
    for name, selector in {
        "close_inter_agent_lt_0.55": lambda row: row["inter_agent_distance"] < 0.55,
        "candidate_active": lambda row: row["candidate_active"],
        "stuck_timer_positive": lambda row: row["stuck_timer"] > 0,
        "new_v3_states": lambda row: row["new_v3_state"],
        "retained_v2_states": lambda row: not row["new_v3_state"],
    }.items():
        selected = [row for row in per_recovery_state if selector(row)]
        cluster_rows.append({
            "cluster": name, "state_count": len(selected),
            "mean_state_false_intervention": float(np.mean([row["mean_false_intervention"] for row in selected])) if selected else None,
        })
    by_group = []
    for group in sorted({row["source_group"] for row in per_recovery_state}):
        selected = [row for row in per_recovery_state if row["source_group"] == group]
        by_group.append({"source_group": group, "state_count": len(selected), "mean_state_false_intervention": float(np.mean([row["mean_false_intervention"] for row in selected]))})
    recovery_metrics = {
        "split": "test", "state_count": len(set(recovery_ids.tolist())), "sample_count": int(recovery_zero_test.sum()),
        "semantics": "||G_phi(x)||_2 on held-out zero-label RECOVERY samples",
        "sample_distribution": norm_distribution(recovery_norms),
        "state_mean_distribution": norm_distribution(recovery_state_means),
        "V2_published_mean": 0.111549,
        "fractional_reduction_vs_V2_published": float(1.0 - recovery_norms.mean() / 0.111549),
        "per_state": sorted(per_recovery_state, key=lambda row: row["mean_false_intervention"], reverse=True),
        "failure_mode_clusters": cluster_rows,
        "source_group_clusters": by_group,
    }
    write_json(HERE / "recovery_zero_metrics.json", recovery_metrics)

    all_zero_norms = np.linalg.norm(test_pred[zero_test], axis=1)
    nonzero_errors = np.linalg.norm(test_pred[~zero_test] - test_target[~zero_test], axis=1)
    nonzero_norms = np.linalg.norm(test_target[~zero_test], axis=1)
    write_json(HERE / "zero_label_metrics.json", {"split": "test", "state_count": len(set(test_sid[zero_test].tolist())), "sample_count": int(zero_test.sum()), **norm_distribution(all_zero_norms), "mean_over_vmax": float(all_zero_norms.mean() / VMAX)})
    nonzero_metrics = {
        "split": "test", "state_count": len(set(test_sid[~zero_test].tolist())), "sample_count": int((~zero_test).sum()),
        "mean_l2_error": float(nonzero_errors.mean()), "median_l2_error": float(np.median(nonzero_errors)),
        "p95_l2_error": float(np.quantile(nonzero_errors, 0.95)), "max_l2_error": float(nonzero_errors.max()),
        "mean_relative_error": float(np.mean(nonzero_errors / nonzero_norms)), "mean_target_norm": float(nonzero_norms.mean()),
        "V2_published_mean_l2_error": 0.00779355,
        "fractional_change_vs_V2": float(nonzero_errors.mean() / 0.00779355 - 1.0),
    }
    write_json(HERE / "nonzero_label_metrics.json", nonzero_metrics)

    # Exact matched V2-test comparison. V2 memberships were deliberately kept.
    with np.load(V2_DATA / "samples.npz", allow_pickle=False) as source:
        v2_arrays = {key: np.asarray(source[key]) for key in source.files}
    v2_test_ids = set(v2_arrays["state_id"][v2_arrays["split"] == "test"].tolist())
    matched_mask = np.asarray([state_id in v2_test_ids for state_id in test_sid])
    v2_prediction = prediction_from_checkpoint(V2_TRAIN / "best_checkpoint.npz", arrays["features"][test_indices])
    matched_groups = {
        "all_V2_test_states": matched_mask,
        "zero_label_RECOVERY": matched_mask & recovery_zero_test,
        "all_zero_label": matched_mask & zero_test,
        "all_nonzero_label": matched_mask & (~zero_test),
    }
    matched = {"V2_checkpoint": str(V2_TRAIN / "best_checkpoint.npz"), "V3_checkpoint": str(HERE / "best_checkpoint.npz"), "groups": {}}
    for name, mask in matched_groups.items():
        target = test_target[mask]
        ids = test_sid[mask]
        old_metrics = group_metrics(v2_prediction[mask], target, ids)
        new_metrics = group_metrics(test_pred[mask], target, ids)
        matched["groups"][name] = {
            "state_count": len(set(ids.tolist())), "sample_count": int(mask.sum()),
            "V2": old_metrics, "V3": new_metrics,
            "mean_l2_absolute_change": new_metrics["state_grouped_mean_l2"] - old_metrics["state_grouped_mean_l2"],
            "mean_l2_fractional_reduction": 1.0 - new_metrics["state_grouped_mean_l2"] / max(old_metrics["state_grouped_mean_l2"], EPS),
        }
    write_json(HERE / "matched_v2_v3_comparison.json", matched)

    # Coverage-only scaling: keep all old V2 train states fixed and add 0/15/30/all new train states.
    new_train_rows = [row for row in state_manifest.values() if not row.get("retained_from_v2", True) and row["split"] == "train"]
    ordered_new = state_ids_in_order(new_train_rows)
    counts = sorted(set([0, min(15, len(ordered_new)), min(30, len(ordered_new)), len(ordered_new)]))
    v2_train_ids = {row["state_id"] for row in core.read_jsonl(V2_DATA / "state_manifest.jsonl") if row["split"] == "train"}
    val_indices = indices_by_split["validation"]
    val_recovery_zero = (arrays["category"][val_indices] == "RECOVERY") & (np.linalg.norm(y[val_indices], axis=1) <= 1e-14)
    scaling_rows = []
    seed23_full = next(run for run in final_runs if run["seed"] == 23)
    for count in counts:
        selected_ids = v2_train_ids | set(ordered_new[:count])
        subset_indices = train_indices[np.isin(arrays["state_id"][train_indices], list(selected_ids))]
        subset_mean, subset_scale, _ = core.fit_normalization(arrays["features"][subset_indices], schema)
        x_train = core.normalize(arrays["features"][subset_indices], subset_mean, subset_scale).astype(np.float32)
        x_val = core.normalize(arrays["features"][val_indices], subset_mean, subset_scale).astype(np.float32)
        config = {**base_config, "name": f"RECOVERY_ZERO_SCALING_plus{count}_seed23", "seed": 23}
        if count == 0:
            params, old_mean, old_scale = load_checkpoint(V2_TRAIN / "checkpoints/final_seed23.npz")
            # The fixed V2 training set and recipe imply identical train-only normalization.
            if not (np.allclose(subset_mean, old_mean, atol=1e-12, rtol=0) and np.allclose(subset_scale, old_scale, atol=1e-12, rtol=0)):
                raise AssertionError("V2 scaling-prefix normalization did not reproduce")
            summary = {"best_epoch": "reused_V2_seed23", "runtime_s": 0.0}
        elif count == len(ordered_new):
            params = seed23_full["params"]
            summary = seed23_full["summary"]
        else:
            params, history, summary = core.train_mlp(
                x_train, y[subset_indices], arrays["state_id"][subset_indices],
                x_val, y[val_indices], arrays["state_id"][val_indices], config)
            histories.extend(history)
        train_prediction = core.predict_mlp(params, x_train)
        val_prediction = core.predict_mlp(params, x_val)
        train_metrics = group_metrics(train_prediction, y[subset_indices], arrays["state_id"][subset_indices])
        val_metrics = group_metrics(val_prediction, y[val_indices], arrays["state_id"][val_indices])
        scaling_rows.append({
            "new_recovery_zero_training_states": count,
            "total_independent_training_states": len(selected_ids), "training_samples": len(subset_indices),
            "seed": 23, "best_epoch": summary["best_epoch"],
            "train_state_grouped_mean_l2": train_metrics["state_grouped_mean_l2"],
            "validation_state_grouped_mean_l2": val_metrics["state_grouped_mean_l2"],
            "validation_recovery_zero_state_count": len(set(arrays["state_id"][val_indices][val_recovery_zero].tolist())),
            "validation_recovery_zero_false_intervention_mean": float(np.linalg.norm(val_prediction[val_recovery_zero], axis=1).mean()),
        })
    write_csv(HERE / "scaling_analysis.csv", scaling_rows)
    write_csv(HERE / "training_history.csv", histories)

    protocol = json.loads((DATA / "protocol.json").read_text())
    projection = core.projection_replay(best_predictions, arrays, indices_by_split, protocol)
    write_json(HERE / "projection_replay_metrics.json", projection)

    test_comparison = {
        name: group_metrics(predictions[name]["test"], split_data["test"][1], split_data["test"][2])
        for name in ("ZERO", "TRAIN_MEAN", "LINEAR")
    }
    test_comparison[best_name] = best_metrics["test"]
    recovery_reduction = recovery_metrics["fractional_reduction_vs_V2_published"]
    matched_recovery = matched["groups"]["zero_label_RECOVERY"]
    matched_reduction = matched_recovery["mean_l2_fractional_reduction"]
    nonzero_degraded = bool(nonzero_metrics["mean_l2_error"] > 0.00979355 and nonzero_metrics["fractional_change_vs_V2"] > 0.20)
    scaling_values = [row["validation_recovery_zero_false_intervention_mean"] for row in scaling_rows]
    scaling_improvement = 1.0 - scaling_values[-1] / max(scaling_values[0], EPS)
    scaling_decrease_count = sum(b < a for a, b in zip(scaling_values, scaling_values[1:]))
    systematic_scaling = bool(scaling_improvement >= 0.20 and scaling_decrease_count >= 2)
    validation_gap = best_metrics["validation"]["state_grouped_mean_l2"] - best_metrics["train"]["state_grouped_mean_l2"]
    reasonable_gap = validation_gap <= 0.05
    if recovery_reduction >= 0.50 and matched_reduction >= 0.30 and not nonzero_degraded and reasonable_gap:
        conclusion = "RECOVERY_ZERO_COVERAGE_FIXES_FAILURE"
    elif recovery_reduction >= 0.20 and systematic_scaling:
        conclusion = "STILL_RECOVERY_DATA_LIMITED"
    else:
        conclusion = "COVERAGE_NOT_ENOUGH"
    projection_failures = len(projection["validation"]["projection_failures"]) + len(projection["test"]["projection_failures"])
    finite = all(np.isfinite(value).all() for value in best_predictions.values())
    ready = bool(
        conclusion == "RECOVERY_ZERO_COVERAGE_FIXES_FAILURE"
        and recovery_norms.mean() <= 0.025 and not nonzero_degraded
        and best_metrics["test"]["state_grouped_mean_l2"] <= 0.03
        and projection_failures == 0 and finite
    )
    readiness = "READY_FOR_CLOSED_LOOP_PILOT" if ready else "NOT_READY_FOR_CLOSED_LOOP_PILOT"
    test_metrics = {
        "selected_model": best_name, "selection_used_test": False,
        "metrics_by_split": best_metrics, "required_test_baseline_comparison": test_comparison,
        "recovery_zero_conclusion": conclusion, "ready_for_closed_loop_pilot": ready,
        "acceptance_evidence": {
            "V2_published_recovery_zero_mean": 0.111549,
            "V3_recovery_zero_mean": float(recovery_norms.mean()),
            "fractional_reduction": recovery_reduction,
            "matched_V2_to_V3_fractional_reduction": matched_reduction,
            "V2_published_nonzero_error": 0.00779355,
            "V3_nonzero_error": nonzero_metrics["mean_l2_error"],
            "nonzero_material_degradation": nonzero_degraded,
            "scaling_fractional_improvement": scaling_improvement,
            "scaling_decreasing_transitions": scaling_decrease_count,
            "systematic_scaling_improvement": systematic_scaling,
            "validation_train_mean_l2_gap": validation_gap,
            "projection_failures": projection_failures,
            "finite_predictions": finite,
        },
    }
    write_json(HERE / "test_metrics.json", test_metrics)

    config_output = {
        "task": "targeted recovery-zero state-coverage retraining",
        "input_dimension": 214, "output_dimension": 4,
        "target": "g*_exec = u*_exec - u_safe", "primary_loss": "mean squared error to g*_exec",
        "architecture": [214, 128, 128, 4], "activation": "SiLU", "output_activation": "identity",
        "optimizer": "AdamW", "learning_rate": 1e-3, "weight_decay": 1e-5,
        "selected_training_seed": best["seed"], "training_seeds": [17, 23, 41],
        "batch_size": 256, "early_stopping": {"metric": "validation state-grouped MSE", "best_epoch": best["summary"]["best_epoch"], "patience_evaluations": 35, "eval_interval_epochs": 5},
        "model_selection": "validation only", "parameter_count": best["summary"]["parameter_count"],
        "conclusion": conclusion, "readiness": readiness,
    }
    write_json(HERE / "config.json", config_output)
    (HERE / "model_summary.txt").write_text(
        "Deterministic G_phi\nArchitecture: 214 -> 128 -> 128 -> 4\nHidden activation: SiLU; output: linear\n"
        f"Parameters: {best['summary']['parameter_count']}\nPrimary loss: MSE to g*_exec\nSelected seed: {best['seed']}\n")

    finished_utc = datetime.now(timezone.utc).isoformat()
    runtime = {
        "started_utc": started_utc, "finished_utc": finished_utc,
        "wall_clock_seconds": time.monotonic() - started,
        "three_seed_training_seconds_sum": float(sum(run["summary"]["runtime_s"] for run in final_runs)),
        "additional_scaling_run_count": 2,
        "jax_devices": [str(device) for device in jax.devices()],
        "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "xla_memory_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "cpu_count_visible": os.cpu_count(), "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "python": sys.version, "platform": platform.platform(),
        "jax": jax.__version__, "flax": flax.__version__, "optax": optax.__version__,
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    frozen_paths = {
        "environment": SYSROOT / "single_integrator/environment.py",
        "projection": SYSROOT / "single_integrator/cbf.py",
        "corrector": ROOT / "diagnostics/cl_fhcb/closed_loop.py",
        "retry": ROOT / "diagnostics/success_basin_multimodality/exact_projector.py",
        "checkpoint": Path(protocol["checkpoint"]),
    }
    frozen_actual = {key: sha(path) for key, path in frozen_paths.items()}
    sanity = dict(audit)
    sanity.update({
        "dataset_unchanged": sha(DATA / "samples.npz") == audit["samples_sha256_actual"],
        "frozen_hashes_expected": protocol["frozen_hashes"], "frozen_hashes_after": frozen_actual,
        "frozen_hashes_unchanged": frozen_actual == protocol["frozen_hashes"],
        "prediction_all_finite": finite, "checkpoint_exists": (HERE / "best_checkpoint.npz").is_file(),
        "test_not_used_for_selection": True, "large_closed_loop_benchmark_performed": False,
    })
    write_json(HERE / "sanity_checks.json", sanity)

    report = f"""# Deterministic G_phi targeted RECOVERY-zero retraining v3

## Diagnosis

**{conclusion}**  
Closed-loop pilot status: **{readiness}**.

The frozen `214 -> 128 -> 128 -> 4` SiLU model was retrained with direct MSE to `g*_exec`; no gate, auxiliary loss, reweighting, architecture change, controller change, or closed-loop benchmark was introduced.

## Selected model

- Seed: **{best['seed']}** (chosen by validation state-grouped MSE from 17/23/41).
- Parameters: **{best['summary']['parameter_count']}**.
- AdamW: lr `1e-3`, weight decay `1e-5`; train-only normalization.
- Checkpoint: `{HERE / 'best_checkpoint.npz'}`.

| Split | state-grouped mean L2 (m/s) | state-grouped RMSE |
|---|---:|---:|
| Train | {best_metrics['train']['state_grouped_mean_l2']:.6f} | {best_metrics['train']['state_grouped_rmse']:.6f} |
| Validation | {best_metrics['validation']['state_grouped_mean_l2']:.6f} | {best_metrics['validation']['state_grouped_rmse']:.6f} |
| Test | {best_metrics['test']['state_grouped_mean_l2']:.6f} | {best_metrics['test']['state_grouped_rmse']:.6f} |

## Primary RECOVERY-zero result

- Held-out zero-label RECOVERY states/samples: {len(set(recovery_ids.tolist()))}/{int(recovery_zero_test.sum())}.
- False intervention mean / median / P95 / max: **{recovery_norms.mean():.6f} / {np.median(recovery_norms):.6f} / {np.quantile(recovery_norms, .95):.6f} / {recovery_norms.max():.6f} m/s**.
- V2 published mean: **0.111549 m/s**; V3 change: **{recovery_norms.mean()-0.111549:+.6f} m/s ({recovery_reduction:.2%} reduction)**.
- Exact matched nine V2 zero-label RECOVERY states: {matched_recovery['V2']['state_grouped_mean_l2']:.6f} -> {matched_recovery['V3']['state_grouped_mean_l2']:.6f} m/s ({matched_reduction:.2%} reduction).
- Test nonzero-label error: **{nonzero_metrics['mean_l2_error']:.6f} m/s** versus V2 **0.007794 m/s**; material degradation: **{nonzero_degraded}**.

## Coverage scaling

With all old training data fixed, validation RECOVERY-zero false intervention changed from **{scaling_values[0]:.6f}** at +0 new states to **{scaling_values[-1]:.6f} m/s** at +{len(ordered_new)} new states. Decreasing transitions: {scaling_decrease_count}/{len(scaling_values)-1}; systematic improvement under the preregistered diagnostic rule: **{systematic_scaling}**.

## Projection replay and scope

Test projected executed-action error is {projection['test']['executed_action_error']['mean']:.6f} m/s; mean projection rewrite is {projection['test']['projection_rewrite_norm']['mean']:.6f} m/s. Solver failures: {len(projection['test']['projection_failures'])}; invalid/nonfinite predictions: {not finite}.

This was an offline state-coverage experiment only. No formal closed-loop benchmark was run.
"""
    (HERE / "training_report.md").write_text(report)

    required = (
        "training_report.md", "config.json", "normalization.json", "training_history.csv",
        "model_comparison.csv", "state_grouped_metrics.csv", "category_metrics.csv",
        "recovery_zero_metrics.json", "matched_v2_v3_comparison.json", "scaling_analysis.csv",
        "projection_replay_metrics.json", "zero_label_metrics.json", "nonzero_label_metrics.json",
        "test_metrics.json", "best_checkpoint.npz", "sanity_checks.json", "runtime_statistics.json",
    )
    manifest = {
        "study": "GPHI_RECOVERY_ZERO_COVERAGE", "generated_at_utc": finished_utc,
        "classification": conclusion, "readiness": readiness, "training_performed": True,
        "large_closed_loop_benchmark_performed": False, "dataset_directory": str(DATA),
        "dataset_manifest_sha256": sha(DATA / "manifest.json"), "dataset_samples_sha256": sha(DATA / "samples.npz"),
        "resource_policy": "one GPU shard and four CPU cores because another lab user had an active scheduler job",
        "selected_checkpoint": "best_checkpoint.npz", "frozen_hashes_unchanged": sanity["frozen_hashes_unchanged"],
        "files_sha256": {name: sha(HERE / name) for name in required},
    }
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({
        "conclusion": conclusion, "readiness": readiness, "seed": best["seed"],
        "train_val_test_mean_l2": [best_metrics[name]["state_grouped_mean_l2"] for name in ("train", "validation", "test")],
        "recovery_zero_mean": float(recovery_norms.mean()), "matched_recovery_reduction": matched_reduction,
        "nonzero_error": nonzero_metrics["mean_l2_error"], "runtime_s": runtime["wall_clock_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
