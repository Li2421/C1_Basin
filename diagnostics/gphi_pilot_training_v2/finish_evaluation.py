"""Resume only the offline evaluation after a completed cached training run.

The training job had already persisted all three final-seed checkpoints and
validation-only model selection before an import-path error in projection
replay.  This script deliberately reuses `best_checkpoint.npz`; it does not
perform another optimizer step.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "diagnostics" / "gphi_training_dataset_v2"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
sys.path.insert(0, str(ROOT))

from diagnostics.gphi_pilot_training_v2.train_and_evaluate import (
    VMAX, dataset_audit, git_revision, nearest_neighbor_audit, normalize,
    predict_mlp, projection_replay, ridge_fit, ridge_predict, save_checkpoint,
    sha, state_metric_rows, subset_metrics, write_csv, write_json,
)


def load_checkpoint(path: Path):
    with np.load(path, allow_pickle=False) as saved:
        config = json.loads(str(saved["architecture_json"]))
        mean = np.asarray(saved["normalization_mean"])
        scale = np.asarray(saved["normalization_scale"])
        binary = np.asarray(saved["normalization_binary_mask"], dtype=bool)
        params = []
        index = 0
        while f"layer_{index}_weight" in saved:
            params.append({
                "w": np.asarray(saved[f"layer_{index}_weight"]),
                "b": np.asarray(saved[f"layer_{index}_bias"]),
            })
            index += 1
    return params, config, mean, scale, binary


def parse_csv(path: Path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def run() -> None:
    started = time.monotonic()
    arrays, audit, metadata, schema, dataset_manifest = dataset_audit()
    if not audit["passed"]:
        raise RuntimeError(audit["failures"])
    params, config, mean, scale, binary = load_checkpoint(HERE / "best_checkpoint.npz")
    x_norm = normalize(arrays["features"], mean, scale).astype(np.float32)
    target = arrays["targets"].astype(np.float32)
    indices = {split: np.flatnonzero(arrays["split"] == split)
               for split in ("train", "validation", "test")}
    metadata_by_sample = {row["sample_id"]: row for row in metadata}

    predictions, metrics = {}, {}
    for split, chosen in indices.items():
        predictions[split] = predict_mlp(params, x_norm[chosen])
        metrics[split] = subset_metrics(
            predictions[split], target[chosen], arrays["state_id"][chosen])

    # Required baselines.  The linear regularization was already selected on
    # validation during the completed training job (1e-4); test is diagnostic.
    train = indices["train"]
    train_mean = target[train].mean(axis=0)
    linear = ridge_fit(x_norm[train], target[train], 1e-4)
    baseline_metrics = {}
    baseline_nonzero_metrics = {}
    test_nonzero = np.linalg.norm(target[indices["test"]], axis=1) > 1e-14
    for name in ("ZERO", "TRAIN_MEAN", "LINEAR"):
        chosen = indices["test"]
        if name == "ZERO":
            prediction = np.zeros_like(target[chosen])
        elif name == "TRAIN_MEAN":
            prediction = np.broadcast_to(train_mean, target[chosen].shape).copy()
        else:
            prediction = ridge_predict(linear, x_norm[chosen])
        baseline_metrics[name] = subset_metrics(
            prediction, target[chosen], arrays["state_id"][chosen])
        baseline_nonzero_metrics[name] = subset_metrics(
            prediction[test_nonzero], target[chosen][test_nonzero],
            arrays["state_id"][chosen][test_nonzero])

    protocol = json.loads((DATA / "protocol.json").read_text())
    projection = projection_replay(predictions, arrays, indices, protocol)
    write_json(HERE / "projection_replay_metrics.json", projection)
    nearest = nearest_neighbor_audit(x_norm, arrays, indices, predictions)

    category_rows, ambiguity_rows = [], []
    for split, chosen in indices.items():
        pred, truth = predictions[split], target[chosen]
        state_ids = arrays["state_id"][chosen]
        for category in ("NORMAL", "PRE_DEADLOCK", "RECOVERY"):
            mask = arrays["category"][chosen] == category
            category_rows.append({"split": split, "group_type": "category", "group": category,
                                  **subset_metrics(pred[mask], truth[mask], state_ids[mask])})
        zero = np.linalg.norm(truth, axis=1) <= 1e-14
        for label, mask in (("zero_label", zero), ("nonzero_label", ~zero)):
            row = {"split": split, "group_type": "label", "group": label,
                   **subset_metrics(pred[mask], truth[mask], state_ids[mask])}
            if label == "zero_label" and np.any(mask):
                norms = np.linalg.norm(pred[mask], axis=1)
                row["false_intervention_mean_norm"] = float(norms.mean())
                row["false_intervention_max_norm"] = float(norms.max())
            category_rows.append(row)
        classes = np.asarray([
            metadata_by_sample[str(arrays["sample_id"][index])]["label_classification"]
            for index in chosen
        ])
        for label in ("LABEL_STABLE", "LABEL_MILDLY_AMBIGUOUS"):
            mask = classes == label
            ambiguity_rows.append({"split": split, "label_classification": label,
                                   **subset_metrics(pred[mask], truth[mask], state_ids[mask])})
    write_csv(HERE / "category_metrics.csv", category_rows)
    write_csv(HERE / "label_ambiguity_metrics.csv", ambiguity_rows)

    best_name = f"SELECTED_MLP_seed{config['seed']}"
    test_comparison = {**baseline_metrics, best_name: metrics["test"]}
    zero_test = next(row for row in category_rows
                     if row["split"] == "test" and row["group"] == "zero_label")
    nonzero_test = next(row for row in category_rows
                        if row["split"] == "test" and row["group"] == "nonzero_label")
    stable_test = next(row for row in ambiguity_rows
                       if row["split"] == "test" and row["label_classification"] == "LABEL_STABLE")
    mild_test = next(row for row in ambiguity_rows
                     if row["split"] == "test" and row["label_classification"] == "LABEL_MILDLY_AMBIGUOUS")
    train_val_ratio = metrics["validation"]["state_grouped_mse"] / max(
        metrics["train"]["state_grouped_mse"], 1e-12)
    train_test_ratio = metrics["test"]["state_grouped_mse"] / max(
        metrics["train"]["state_grouped_mse"], 1e-12)
    nonlinear_beats = (
        nonzero_test["state_grouped_mean_l2"]
        < min(baseline_nonzero_metrics["ZERO"]["state_grouped_mean_l2"],
              baseline_nonzero_metrics["TRAIN_MEAN"]["state_grouped_mean_l2"])
    )
    projection_not_rewriting = (
        projection["test"]["projection_changed_fraction_over_1e-6"] < 0.5
        or projection["test"]["projection_rewrite_norm"]["mean"]
        < metrics["test"]["state_grouped_mean_l2"]
    )
    severe_memorization = bool(train_test_ratio > 5 and train_val_ratio > 3)
    test_nonzero_mask = np.linalg.norm(target[indices["test"]], axis=1) > 1e-14
    test_nonzero_target_mean_norm = float(np.mean(np.linalg.norm(
        target[indices["test"]][test_nonzero_mask], axis=1)))
    false_to_nonzero_scale = (zero_test["false_intervention_mean_norm"] /
                              max(test_nonzero_target_mean_norm, 1e-12))
    # This is an evidence statement, not a newly optimized threshold: the
    # alleged zero correction is of the same order as a real nonzero target.
    false_intervention_reasonable = bool(false_to_nonzero_scale < 0.25)
    finite = all(np.isfinite(value).all() for value in predictions.values())
    projection_failures = sum(len(projection[split]["projection_failures"])
                              for split in ("validation", "test"))
    acceptance = {
        "nonlinear_materially_beats_zero_and_mean_on_nonzero_test_states": bool(nonlinear_beats),
        "test_nonzero_state_grouped_mean_l2": nonzero_test["state_grouped_mean_l2"],
        "test_nonzero_mean_target_relative_error": nonzero_test["mean_relative_l2_nonzero"],
        "test_zero_false_intervention_mean_norm": zero_test["false_intervention_mean_norm"],
        "test_zero_false_intervention_over_vmax": zero_test["false_intervention_mean_norm"] / VMAX,
        "test_nonzero_target_mean_norm": test_nonzero_target_mean_norm,
        "zero_false_intervention_over_nonzero_target_mean_norm": false_to_nonzero_scale,
        "near_zero_intervention_on_zero_label_states_supported": false_intervention_reasonable,
        "train_to_validation_state_mse_ratio": train_val_ratio,
        "train_to_test_state_mse_ratio": train_test_ratio,
        "severe_memorization_signal": severe_memorization,
        "projection_does_not_completely_rewrite_most_outputs": bool(projection_not_rewriting),
        "finite_predictions": finite, "projection_failures": projection_failures,
        "decision_note": "Evidence-based pilot gate; numeric comparisons are reported rather than presented as a theorem.",
    }
    readiness = ("READY_FOR_CLOSED_LOOP_PILOT"
                 if nonlinear_beats and false_intervention_reasonable
                 and not severe_memorization and projection_not_rewriting
                 and finite and projection_failures == 0
                 else "TRAINING_NEEDS_REVISION")
    write_json(HERE / "test_metrics.json", {
        "selected_model": best_name, "selection_used_test": False,
        "selected_from_validation_seed": config["seed"],
        "metrics_by_split": metrics,
        "required_test_baseline_comparison": test_comparison,
        "required_nonzero_test_baseline_comparison": baseline_nonzero_metrics,
        "overfitting_and_nearest_neighbor_audit": nearest,
        "acceptance_evidence": acceptance, "classification": readiness,
    })

    parameter_count = int(sum(layer["w"].size + layer["b"].size for layer in params))
    comparison = parse_csv(HERE / "model_comparison.csv")
    selected_row = next(row for row in comparison
                        if row["model"] == f"FINAL_seed{config['seed']}")
    best_epoch = int(float(selected_row["best_epoch"]))
    config_output = {
        "task": "deterministic G_phi supervised pilot", "input_dimension": 214,
        "output_dimension": 4, "target": "g*_exec = u*_exec - u_safe",
        "primary_loss": "mean squared error to g*_exec",
        "selected_architecture": config["hidden"], "activation": "SiLU",
        "output_activation": "identity", "selected_training_seed": config["seed"],
        "final_architecture_seeds": [17, 23, 41], "optimizer": "AdamW",
        "learning_rate": config["learning_rate"], "weight_decay": config["weight_decay"],
        "batch_size": config["batch_size"], "early_stopping": {
            "metric": "validation state-grouped MSE", "best_epoch": best_epoch,
            "patience_evaluations": config["patience_evaluations"],
            "eval_interval_epochs": config["eval_interval"],
        },
        "model_selection": "validation only; test opened after architecture/hyperparameter/seed frozen",
        "parameter_count": parameter_count,
        "dataset_manifest_sha256": sha(DATA / "manifest.json"),
    }
    write_json(HERE / "config.json", config_output)
    (HERE / "model_summary.txt").write_text(
        "Deterministic G_phi\n"
        f"Architecture: 214 -> {' -> '.join(map(str, config['hidden']))} -> 4\n"
        "Hidden activation: SiLU; output: linear\n"
        f"Parameters: {parameter_count}\n"
        "Primary loss: direct MSE to four-dimensional executed correction g*_exec\n"
        f"Selected seed: {config['seed']} (validation state-grouped MSE only)\n")

    category_lookup = {(row["split"], row["group"]): row for row in category_rows}
    report = f"""# Deterministic G_phi pilot training v1

## Decision

**{readiness}**

The selected deterministic model predicts the four-dimensional instantaneous
executed correction `g*_exec = u*_exec - u_safe`. It does not predict eta or a
trajectory. No frozen controller, physics, event, oracle, or dataset artifact
was modified. No large closed-loop benchmark was run.

## Selected model

- Architecture: `214 -> {' -> '.join(map(str, config['hidden']))} -> 4`, SiLU.
- Parameters: {parameter_count}.
- AdamW: learning rate {config['learning_rate']}, weight decay {config['weight_decay']}.
- Validation-selected seed: {config['seed']} of 17/23/41; best epoch {best_epoch}.
- Checkpoint: `best_checkpoint.npz`.

## Held-out state results

| Split | state-grouped RMSE | state-grouped mean L2 | sample mean L2 |
|---|---:|---:|---:|
| Train | {metrics['train']['state_grouped_rmse']:.6g} | {metrics['train']['state_grouped_mean_l2']:.6g} | {metrics['train']['mean_l2']:.6g} |
| Validation | {metrics['validation']['state_grouped_rmse']:.6g} | {metrics['validation']['state_grouped_mean_l2']:.6g} | {metrics['validation']['mean_l2']:.6g} |
| Test | {metrics['test']['state_grouped_rmse']:.6g} | {metrics['test']['state_grouped_mean_l2']:.6g} | {metrics['test']['mean_l2']:.6g} |

## Test baselines

| Model | state-grouped mean L2 | state-grouped RMSE |
|---|---:|---:|
| ZERO | {baseline_metrics['ZERO']['state_grouped_mean_l2']:.6g} | {baseline_metrics['ZERO']['state_grouped_rmse']:.6g} |
| TRAIN-MEAN | {baseline_metrics['TRAIN_MEAN']['state_grouped_mean_l2']:.6g} | {baseline_metrics['TRAIN_MEAN']['state_grouped_rmse']:.6g} |
| Linear | {baseline_metrics['LINEAR']['state_grouped_mean_l2']:.6g} | {baseline_metrics['LINEAR']['state_grouped_rmse']:.6g} |
| Selected MLP | {metrics['test']['state_grouped_mean_l2']:.6g} | {metrics['test']['state_grouped_rmse']:.6g} |

## Category and label behavior on test

| Group | state-grouped mean L2 |
|---|---:|
| NORMAL | {category_lookup[('test','NORMAL')]['state_grouped_mean_l2']:.6g} |
| PRE_DEADLOCK | {category_lookup[('test','PRE_DEADLOCK')]['state_grouped_mean_l2']:.6g} |
| RECOVERY | {category_lookup[('test','RECOVERY')]['state_grouped_mean_l2']:.6g} |
| zero label | {zero_test['state_grouped_mean_l2']:.6g} |
| nonzero label | {nonzero_test['state_grouped_mean_l2']:.6g} |
| LABEL_STABLE | {stable_test['state_grouped_mean_l2']:.6g} |
| LABEL_MILDLY_AMBIGUOUS | {mild_test['state_grouped_mean_l2']:.6g} |

Zero-label false intervention mean norm is {zero_test['false_intervention_mean_norm']:.6g} m/s
({zero_test['false_intervention_mean_norm']/VMAX:.3%} of vmax and
{false_to_nonzero_scale:.3%} of the test nonzero-target mean norm). This does
not support approximately-zero intervention on held-out zero-label states.

## Hard-projection replay

Test post-projection executed-action error is
{projection['test']['executed_action_error']['mean']:.6g} m/s (mean). The second
projection rewrites the predicted action by {projection['test']['projection_rewrite_norm']['mean']:.6g}
m/s on average; {projection['test']['projection_changed_fraction_over_1e-6']:.3%}
of samples change by more than 1e-6. Projection failures: {projection_failures}.

## Generalization audit

The 4,288 Flow-seed samples represent only 67 independent augmented states.
The split is source-state/source-trajectory grouped. Validation/train and
test/train state-MSE ratios are {train_val_ratio:.3g} and {train_test_ratio:.3g}.
Nearest-held-out-state distances and output-spread checks are in
`test_metrics.json`. This pilot warrants only a small closed-loop smoke check,
not a performance claim.
"""
    (HERE / "training_report.md").write_text(report)

    # Reconstruct measured training runtime from each run's final history row.
    histories = parse_csv(HERE / "training_history.csv")
    per_run = {}
    for row in histories:
        per_run[row["run"]] = max(per_run.get(row["run"], 0.0), float(row["elapsed_s"]))
    training_runtime_sum = sum(per_run.values())
    slurm_elapsed = None
    try:
        raw = subprocess.check_output(
            ["sacct", "-j", "199", "--noheader", "--parsable2", "-o", "ElapsedRaw"],
            text=True, stderr=subprocess.DEVNULL)
        values = [int(line.split("|")[0]) for line in raw.splitlines()
                  if line.split("|")[0].isdigit()]
        slurm_elapsed = max(values) if values else None
    except Exception:
        pass
    if slurm_elapsed is None:
        # Accounting is disabled on this workstation.  Use the sum of each
        # sequential run's internally measured elapsed time; this omits only
        # lightweight report I/O and is therefore labeled accordingly below.
        slurm_elapsed = training_runtime_sum
    runtime = {
        "training_job_id": "199", "training_job_one_gpu_shard": True,
        "training_job_xla_preallocate": False, "training_job_memory_fraction": 0.10,
        "observed_gpu_process_memory_mib": 724,
        "observed_gpu_memory_source": "nvidia-smi sampled during Slurm job 199",
        "training_job_measured_elapsed_seconds": slurm_elapsed,
        "training_job_elapsed_source": "Slurm accounting, else sum of sequential internally timed runs",
        "sum_of_per_run_training_seconds": training_runtime_sum,
        "offline_resume_evaluation_seconds": time.monotonic() - started,
        "cpu_workers": 4, "duplicate_training_after_import_error": False,
        "cached_checkpoints_reused": True,
        "python": sys.version, "platform": platform.platform(),
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    frozen_paths = {
        "environment": SYSROOT / "single_integrator" / "environment.py",
        "projection": SYSROOT / "single_integrator" / "cbf.py",
        "corrector": ROOT / "diagnostics" / "cl_fhcb" / "closed_loop.py",
        "retry": ROOT / "diagnostics" / "success_basin_multimodality" / "exact_projector.py",
        "checkpoint": Path(protocol["checkpoint"]),
    }
    frozen_actual = {key: sha(path) for key, path in frozen_paths.items()}
    frozen_unchanged = frozen_actual == protocol["frozen_hashes"]
    sanity = dict(audit)
    sanity.update({
        "dataset_samples_sha256_after_training": sha(DATA / "samples.npz"),
        "dataset_unchanged": sha(DATA / "samples.npz") == audit["samples_sha256_actual"],
        "frozen_hashes_expected": protocol["frozen_hashes"],
        "frozen_hashes_after": frozen_actual, "frozen_hashes_unchanged": frozen_unchanged,
        "prediction_all_finite": finite, "checkpoint_exists": True,
        "test_not_used_for_selection": True,
    })
    write_json(HERE / "sanity_checks.json", sanity)

    required = (
        "training_report.md", "config.json", "model_summary.txt", "normalization.json",
        "training_history.csv", "model_comparison.csv", "test_metrics.json",
        "state_grouped_metrics.csv", "category_metrics.csv", "label_ambiguity_metrics.csv",
        "projection_replay_metrics.json", "best_checkpoint.npz", "sanity_checks.json",
        "runtime_statistics.json",
    )
    manifest = {
        "study": "first deterministic G_phi supervised pilot",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "classification": readiness, "training_performed": True,
        "large_closed_loop_benchmark_performed": False,
        "dataset_directory": str(DATA), "dataset_manifest_sha256": sha(DATA / "manifest.json"),
        "dataset_samples_sha256": sha(DATA / "samples.npz"), "git": git_revision(),
        "resource_policy": "one GPU shard; no preallocation; 10% cap; four CPUs",
        "selected_checkpoint": "best_checkpoint.npz", "frozen_hashes_unchanged": frozen_unchanged,
        "files_sha256": {name: sha(HERE / name) for name in required},
    }
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({
        "classification": readiness, "architecture": config["hidden"],
        "parameter_count": parameter_count, "seed": config["seed"],
        "validation_state_grouped_mean_l2": metrics["validation"]["state_grouped_mean_l2"],
        "test_state_grouped_mean_l2": metrics["test"]["state_grouped_mean_l2"],
        "test_projected_mean_l2": projection["test"]["executed_action_error"]["mean"],
        "offline_resume_seconds": runtime["offline_resume_evaluation_seconds"],
    }, indent=2))


if __name__ == "__main__":
    run()
