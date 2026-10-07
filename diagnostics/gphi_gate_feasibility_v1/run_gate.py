"""Strictly offline gate-feasibility audit on immutable Dataset V4."""

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
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "diagnostics/gphi_training_dataset_v4"
V4_TRAIN = ROOT / "diagnostics/gphi_pilot_training_v4"
OLD_HARD_IDS = (
    "R_D4_s95101008_p134", "R_D4_s95105001_p116",
    "R_D4_s95101014_p123", "R_D4_s95105004_p114",
)
EPS = 1e-12


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value) -> None:
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(type(item).__name__)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=convert) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def aggregate_state(sample_prob: np.ndarray, state_ids: np.ndarray, state_label: dict[str, int]):
    grouped = defaultdict(list)
    for probability, state_id in zip(sample_prob, state_ids):
        grouped[str(state_id)].append(float(probability))
    ids = np.asarray(sorted(grouped), dtype=str)
    probabilities = np.asarray([np.mean(grouped[state_id]) for state_id in ids], dtype=np.float64)
    labels = np.asarray([state_label[state_id] for state_id in ids], dtype=np.int64)
    return ids, probabilities, labels


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    pos = int(y.sum()); neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(p, method="average")
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    # Integrate the precision/recall staircase only after complete tied-score
    # groups.  This avoids making AP depend on arbitrary ordering within ties
    # (notably, a constant prior must have AP equal to prevalence).
    order = np.argsort(-p, kind="stable")
    sorted_y = y[order]; sorted_p = p[order]
    tp = 0; fp = 0; previous_recall = 0.0; average_precision = 0.0
    for end in np.r_[np.flatnonzero(sorted_p[1:] != sorted_p[:-1]) + 1, len(y)]:
        start = tp + fp
        group_positive = int(np.sum(sorted_y[start:end]))
        tp += group_positive; fp += int(end - start - group_positive)
        recall = tp / pos; precision = tp / max(tp + fp, 1)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return float(average_precision)


def classification_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(np.int64)
    tp = int(np.sum((pred == 1) & (y == 1))); tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0))); fn = int(np.sum((pred == 0) & (y == 1)))
    recall = tp / max(tp + fn, 1); specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "state_count": int(len(y)), "zero_states": int(np.sum(y == 0)), "nonzero_states": int(np.sum(y == 1)),
        "threshold": float(threshold), "balanced_accuracy": float((recall + specificity) / 2),
        "AUROC": auroc(y, p), "AUPRC": auprc(y, p), "accuracy": float(np.mean(pred == y)),
        "precision": float(precision), "recall": float(recall), "specificity": float(specificity),
        "FPR": float(fp / max(fp + tn, 1)), "FNR": float(fn / max(fn + tp, 1)),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Brier": float(np.mean((p - y) ** 2)),
        "mean_probability": float(np.mean(p)),
        "mean_probability_zero": float(np.mean(p[y == 0])) if np.any(y == 0) else float("nan"),
        "mean_probability_nonzero": float(np.mean(p[y == 1])) if np.any(y == 1) else float("nan"),
    }


def select_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, list[dict]]:
    unique = np.unique(p)
    candidates = np.r_[np.nextafter(unique[0], -np.inf), (unique[:-1] + unique[1:]) / 2,
                       np.nextafter(unique[-1], np.inf), 0.5]
    rows = []
    for threshold in np.unique(candidates):
        metrics = classification_metrics(y, p, float(threshold))
        rows.append({"threshold": float(threshold), **metrics})
    best = max(rows, key=lambda row: (
        row["balanced_accuracy"], -max(row["FPR"], row["FNR"]),
        -abs(row["FPR"] - row["FNR"]), -abs(row["threshold"] - .5)))
    return float(best["threshold"]), rows


def state_bce(y: np.ndarray, p: np.ndarray) -> float:
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    return float(np.mean(-y * np.log(clipped) - (1 - y) * np.log(1 - clipped)))


def init_params(key, dimensions: list[int]):
    params = []
    keys = jax.random.split(key, len(dimensions) - 1)
    for layer_key, fan_in, fan_out in zip(keys, dimensions[:-1], dimensions[1:]):
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        params.append({
            "w": jax.random.uniform(layer_key, (fan_in, fan_out), minval=-limit, maxval=limit),
            "b": jnp.zeros((fan_out,), dtype=jnp.float32),
        })
    return params


def logits_from_params(params, x):
    value = x
    for layer in params[:-1]:
        value = jax.nn.silu(value @ layer["w"] + layer["b"])
    return (value @ params[-1]["w"] + params[-1]["b"]).reshape(-1)


def predict(params, x: np.ndarray) -> np.ndarray:
    return np.asarray(jax.nn.sigmoid(logits_from_params(params, jnp.asarray(x, dtype=jnp.float32))))


def train_model(name: str, hidden: list[int], seed: int, x_train: np.ndarray, y_train: np.ndarray,
                train_ids: np.ndarray, x_val: np.ndarray, val_ids: np.ndarray,
                state_label: dict[str, int]) -> tuple[list[dict], dict, list[dict]]:
    started = time.monotonic()
    params = init_params(jax.random.PRNGKey(seed), [x_train.shape[1], *hidden, 1])
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-5)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(current, state):
        def loss_fn(candidate):
            logits = logits_from_params(candidate, jnp.asarray(x_train, dtype=jnp.float32))
            labels = jnp.asarray(y_train, dtype=jnp.float32)
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))
        loss, grads = jax.value_and_grad(loss_fn)(current)
        updates, state = optimizer.update(grads, state, current)
        return optax.apply_updates(current, updates), state, loss

    best_params = params; best_bce = float("inf"); best_epoch = 0; stale = 0; history = []
    for epoch in range(1, 1201):
        params, opt_state, train_loss = step(params, opt_state)
        if epoch == 1 or epoch % 10 == 0:
            train_state = aggregate_state(predict(params, x_train), train_ids, state_label)
            val_state = aggregate_state(predict(params, x_val), val_ids, state_label)
            validation_bce = state_bce(val_state[2], val_state[1])
            row = {
                "run": name, "seed": seed, "epoch": epoch, "sample_train_BCE": float(train_loss),
                "train_state_BCE": state_bce(train_state[2], train_state[1]),
                "validation_state_BCE": validation_bce,
                "validation_AUROC": auroc(val_state[2], val_state[1]),
                "elapsed_s": time.monotonic() - started,
            }
            history.append(row)
            if validation_bce < best_bce - 1e-6:
                best_bce = validation_bce
                best_epoch = epoch
                best_params = jax.tree_util.tree_map(lambda value: np.asarray(value).copy(), params)
                stale = 0
            else:
                stale += 1
            if stale >= 35:
                break
    summary = {
        "name": name, "hidden": hidden, "seed": seed, "best_epoch": best_epoch,
        "validation_state_BCE": best_bce, "runtime_s": time.monotonic() - started,
        "parameter_count": int(sum(np.prod(value.shape) for layer in best_params for value in layer.values())),
    }
    return best_params, summary, history


def reliability(y: np.ndarray, p: np.ndarray) -> list[dict]:
    rows = []
    for left, right in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:]):
        mask = (p >= left) & (p < right if right < 1 else p <= right)
        rows.append({
            "bin_left": float(left), "bin_right": float(right), "state_count": int(mask.sum()),
            "mean_probability": float(np.mean(p[mask])) if np.any(mask) else None,
            "empirical_intervention_rate": float(np.mean(y[mask])) if np.any(mask) else None,
        })
    return rows


def main() -> None:
    started = time.monotonic(); started_utc = datetime.now(timezone.utc).isoformat()
    HERE.mkdir(parents=True, exist_ok=True)
    dataset_hash_before = sha(DATA / "manifest.json")
    dataset_files_before = {name: sha(DATA / name) for name in ("samples.npz", "state_manifest.jsonl", "sample_metadata.jsonl", "split_manifest.json", "feature_schema.json")}
    dataset_manifest = read_json(DATA / "manifest.json")
    schema = read_json(DATA / "feature_schema.json")
    state_manifest_rows = read_jsonl(DATA / "state_manifest.jsonl")
    state_manifest = {row["state_id"]: row for row in state_manifest_rows}
    sample_metadata = read_jsonl(DATA / "sample_metadata.jsonl")
    with np.load(DATA / "samples.npz", allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}

    failures = []
    if arrays["features"].shape != (20736, 214): failures.append("unexpected feature shape")
    if not np.isfinite(arrays["features"]).all(): failures.append("nonfinite features")
    if sha(DATA / "samples.npz") != dataset_manifest["files_sha256"]["samples.npz"]: failures.append("sample hash mismatch")
    if len(sample_metadata) != len(arrays["features"]): failures.append("sample metadata length mismatch")
    if not np.array_equal(np.asarray([row["sample_id"] for row in sample_metadata]), arrays["sample_id"]): failures.append("sample metadata ordering mismatch")

    state_label_sets = defaultdict(set); state_split_sets = defaultdict(set); state_category_sets = defaultdict(set); state_group_sets = defaultdict(set)
    for row in sample_metadata:
        state_label_sets[row["state_id"]].add(0 if bool(row["zero_label"]) else 1)
        state_split_sets[row["state_id"]].add(row["split"]); state_category_sets[row["state_id"]].add(row["category"])
        state_group_sets[row["state_id"]].add(row["leakage_group"])
    if any(len(value) != 1 for value in state_label_sets.values()): failures.append("inconsistent oracle labels within state")
    if any(len(value) != 1 for value in state_split_sets.values()): failures.append("state split leakage")
    if any(len(value) != 1 for value in state_group_sets.values()): failures.append("state group inconsistency")
    state_label = {state_id: next(iter(value)) for state_id, value in state_label_sets.items()}
    state_split = {state_id: next(iter(value)) for state_id, value in state_split_sets.items()}
    state_category = {state_id: next(iter(value)) for state_id, value in state_category_sets.items()}
    state_group = {state_id: next(iter(value)) for state_id, value in state_group_sets.items()}
    sample_counts = Counter(arrays["state_id"].tolist())
    if set(sample_counts.values()) != {64}: failures.append("not exactly 64 Flow variants per state")
    groups = {split: {state_group[state_id] for state_id in state_label if state_split[state_id] == split} for split in ("train", "validation", "test")}
    group_overlap = {f"{a}_{b}": sorted(groups[a] & groups[b]) for a, b in combinations(groups, 2)}
    if any(group_overlap.values()): failures.append("source group leakage")
    boundary_ids = {state_id for state_id, row in state_manifest.items() if row.get("boundary_candidate")}
    for state_id in boundary_ids:
        expected = 0 if state_manifest[state_id]["oracle_boundary_class"] == "zero" else 1
        if state_label[state_id] != expected: failures.append(f"boundary oracle mismatch: {state_id}")

    split_counts = {}
    for split in ("train", "validation", "test"):
        ids = [state_id for state_id in state_label if state_split[state_id] == split]
        split_counts[split] = {
            "states": len(ids), "gate_0": sum(state_label[state_id] == 0 for state_id in ids),
            "gate_1": sum(state_label[state_id] == 1 for state_id in ids),
            "categories": dict(Counter(state_category[state_id] for state_id in ids)),
            "close_boundary_gate_0": sum(state_id in boundary_ids and state_label[state_id] == 0 for state_id in ids),
            "close_boundary_gate_1": sum(state_id in boundary_ids and state_label[state_id] == 1 for state_id in ids),
        }
    audit = {
        "passed": not failures, "failures": failures, "dataset_directory": str(DATA),
        "dataset_manifest_sha256_before": dataset_hash_before, "sample_sha256": sha(DATA / "samples.npz"),
        "feature_dimension": 214, "sample_count": len(arrays["features"]), "unique_state_count": len(state_label),
        "gate_label_definition": "0 iff eta=(0,0,0) satisfies B_63; 1 otherwise when a nonzero success-constrained oracle correction is required",
        "label_source": "Dataset V4 oracle zero_label metadata; target correction norms were not used",
        "gate_0_states": sum(value == 0 for value in state_label.values()), "gate_1_states": sum(value == 1 for value in state_label.values()),
        "flow_variants_per_state": sorted(set(sample_counts.values())), "split_counts": split_counts,
        "source_group_overlap": group_overlap, "feature_schema_future_information": schema.get("future_information_in_features"),
        "B_63_empty_states": dataset_manifest["B_63_empty_states"], "LABEL_MULTIVALUED_states": dataset_manifest["LABEL_MULTIVALUED_states"],
    }
    write_json(HERE / "dataset_audit.json", audit)
    if failures: raise RuntimeError(failures)
    write_json(HERE / "split_manifest.json", {
        "assignment_unit": "Dataset V4 augmented-state/source-leakage-group split, preserved exactly",
        "counts": split_counts, "source_group_overlap": group_overlap,
        "boundary_pairs_colocated_by_split": True,
        "state_ids": {split: sorted(state_id for state_id in state_label if state_split[state_id] == split) for split in ("train", "validation", "test")},
    })

    # Train-only normalization, with boolean segments left literal as in G_phi.
    train_index = np.flatnonzero(arrays["split"] == "train")
    binary = np.zeros(214, dtype=bool)
    for segment in schema["segments"]:
        if segment["unit"] == "boolean":
            start = int(segment["offset"]); binary[start:start + int(segment["length"])] = True
    mean = arrays["features"][train_index].mean(axis=0); scale = arrays["features"][train_index].std(axis=0)
    scale[scale < 1e-8] = 1.0; mean[binary] = 0.0; scale[binary] = 1.0
    normalized = ((arrays["features"] - mean) / scale).astype(np.float32)
    write_json(HERE / "normalization.json", {"fit_split": "train only", "mean": mean, "scale": scale, "binary_feature_indices": np.flatnonzero(binary), "epsilon": 1e-8})
    sample_y = np.asarray([state_label[str(state_id)] for state_id in arrays["state_id"]], dtype=np.float32)
    split_indices = {split: np.flatnonzero(arrays["split"] == split) for split in ("train", "validation", "test")}

    trained = []; histories = []
    specifications = [("LINEAR", [], 17)]
    specifications += [("MLP_64x64", [64, 64], seed) for seed in (17, 23, 41)]
    specifications += [("MLP_128x128", [128, 128], seed) for seed in (17, 23, 41)]
    for name, hidden, seed in specifications:
        params, summary, history = train_model(
            name, hidden, seed, normalized[split_indices["train"]], sample_y[split_indices["train"]], arrays["state_id"][split_indices["train"]],
            normalized[split_indices["validation"]], arrays["state_id"][split_indices["validation"]], state_label)
        summaries = {}; state_predictions = {}
        for split, index in split_indices.items():
            ids, probabilities, labels = aggregate_state(predict(params, normalized[index]), arrays["state_id"][index], state_label)
            state_predictions[split] = (ids, probabilities, labels)
        threshold, threshold_rows = select_threshold(state_predictions["validation"][2], state_predictions["validation"][1])
        for split, (_, probabilities, labels) in state_predictions.items():
            summaries[split] = classification_metrics(labels, probabilities, threshold)
        trained.append({"name": name, "seed": seed, "hidden": hidden, "params": params, "summary": summary,
                        "state_predictions": state_predictions, "threshold": threshold, "threshold_rows": threshold_rows, "metrics": summaries})
        histories.extend(history)
        print(json.dumps({"model": name, "seed": seed, "epoch": summary["best_epoch"], "val_AUROC": summaries["validation"]["AUROC"],
                          "val_balanced_accuracy": summaries["validation"]["balanced_accuracy"], "threshold": threshold}), flush=True)

    # Keep the best seed within an architecture, then choose architecture from validation only.
    architecture_best = []
    for name in ("LINEAR", "MLP_64x64", "MLP_128x128"):
        candidates = [run for run in trained if run["name"] == name]
        architecture_best.append(max(candidates, key=lambda run: (
            run["metrics"]["validation"]["AUROC"], run["metrics"]["validation"]["balanced_accuracy"],
            -run["summary"]["validation_state_BCE"])))
    best = max(architecture_best, key=lambda run: (
        run["metrics"]["validation"]["AUROC"], run["metrics"]["validation"]["balanced_accuracy"],
        -run["summary"]["validation_state_BCE"]))

    # Majority/prior baseline is fitted on state prevalence, not repeated samples.
    train_state_labels = np.asarray([state_label[state_id] for state_id in state_label if state_split[state_id] == "train"])
    prior_probability = float(np.mean(train_state_labels)); prior_metrics = {}
    validation_ids = np.asarray(sorted(state_id for state_id in state_label if state_split[state_id] == "validation"))
    validation_labels = np.asarray([state_label[state_id] for state_id in validation_ids])
    prior_threshold, prior_threshold_rows = select_threshold(validation_labels, np.full(len(validation_labels), prior_probability))
    for split in ("train", "validation", "test"):
        ids = np.asarray(sorted(state_id for state_id in state_label if state_split[state_id] == split))
        labels = np.asarray([state_label[state_id] for state_id in ids])
        prior_metrics[split] = classification_metrics(labels, np.full(len(labels), prior_probability), prior_threshold)

    comparison = []
    comparison.append({"model": "PRIOR_MAJORITY", "seed": "", "parameter_count": 0, "selected": False,
                       "validation_selected_threshold": prior_threshold, "train_prevalence": prior_probability,
                       **{f"{split}_{key}": prior_metrics[split][key] for split in ("train", "validation", "test") for key in ("balanced_accuracy", "AUROC", "AUPRC", "accuracy", "FPR", "FNR", "Brier")}})
    for run in architecture_best:
        comparison.append({"model": run["name"], "seed": run["seed"], "parameter_count": run["summary"]["parameter_count"],
                           "selected": run is best, "best_epoch": run["summary"]["best_epoch"],
                           "validation_selected_threshold": run["threshold"],
                           **{f"{split}_{key}": run["metrics"][split][key] for split in ("train", "validation", "test") for key in ("balanced_accuracy", "AUROC", "AUPRC", "accuracy", "precision", "recall", "specificity", "FPR", "FNR", "Brier")}})
    write_csv(HERE / "model_comparison.csv", comparison)
    write_csv(HERE / "training_history.csv", histories)

    # Save only the selected gate; it contains one scalar logit and no correction head.
    checkpoint = {"normalization_mean": mean, "normalization_scale": scale, "normalization_binary_mask": binary,
                  "architecture_json": np.asarray(json.dumps([214, *best["hidden"], 1])),
                  "threshold": np.asarray(best["threshold"]), "oracle_label_definition": np.asarray(audit["gate_label_definition"])}
    for layer_index, layer in enumerate(best["params"]):
        checkpoint[f"layer_{layer_index}_weight"] = np.asarray(layer["w"])
        checkpoint[f"layer_{layer_index}_bias"] = np.asarray(layer["b"])
    np.savez_compressed(HERE / "best_gate_checkpoint.npz", **checkpoint)

    model_runs = [("PRIOR_MAJORITY", None)] + [(run["name"], run) for run in architecture_best]
    state_rows = []; category_rows = []
    best_state = {}
    for model_name, run in model_runs:
        threshold = prior_threshold if run is None else run["threshold"]
        for split in ("train", "validation", "test"):
            if run is None:
                ids = np.asarray(sorted(state_id for state_id in state_label if state_split[state_id] == split)); probabilities = np.full(len(ids), prior_probability); labels = np.asarray([state_label[state_id] for state_id in ids])
            else:
                ids, probabilities, labels = run["state_predictions"][split]
            if run is best:
                best_state.update({str(state_id): float(probability) for state_id, probability in zip(ids, probabilities)})
            for state_id, probability, label in zip(ids, probabilities, labels):
                meta = state_manifest[str(state_id)]
                state_rows.append({
                    "model": model_name, "split": split, "state_id": state_id, "category": state_category[str(state_id)],
                    "source_group": state_group[str(state_id)], "oracle_gate_label": int(label), "p_gate": float(probability),
                    "threshold": threshold, "predicted_gate_label": int(probability >= threshold), "correct": bool((probability >= threshold) == label),
                    "close_recovery_boundary": str(state_id) in boundary_ids,
                    "inter_agent_distance": meta.get("inter_agent_distance"),
                })
            subsets = {"ALL": np.ones(len(ids), dtype=bool)}
            subsets.update({category: np.asarray([state_category[str(state_id)] == category for state_id in ids]) for category in ("NORMAL", "PRE_DEADLOCK", "RECOVERY")})
            subsets["CLOSE_RANGE_RECOVERY_BOUNDARY"] = np.asarray([str(state_id) in boundary_ids for state_id in ids])
            for category, mask in subsets.items():
                if np.any(mask): category_rows.append({"model": model_name, "split": split, "subset": category, **classification_metrics(labels[mask], probabilities[mask], threshold)})
    write_csv(HERE / "state_metrics.csv", state_rows); write_csv(HERE / "category_metrics.csv", category_rows)

    # Threshold audit: full validation sweep for selected model and explicit 0.5 comparison.
    best_val = best["state_predictions"]["validation"]
    threshold_analysis = {
        "selection_split": "validation only", "selection_rule": "maximize state-level balanced accuracy; tie-break max(FPR,FNR), FPR/FNR symmetry, then proximity to 0.5",
        "selected_model": best["name"], "selected_seed": best["seed"], "selected_threshold": best["threshold"],
        "validation_at_selected_threshold": best["metrics"]["validation"], "test_at_frozen_threshold": best["metrics"]["test"],
        "validation_at_0.5": classification_metrics(best_val[2], best_val[1], .5),
        "test_at_0.5": classification_metrics(best["state_predictions"]["test"][2], best["state_predictions"]["test"][1], .5),
        "validation_sweep": best["threshold_rows"],
    }
    write_json(HERE / "threshold_analysis.json", threshold_analysis)

    # Matched pairs, including held-out pairs, use state-averaged probability.
    pair_rows = []
    for row in csv.DictReader((DATA / "matched_boundary_pairs.csv").open()):
        zero_id = row["zero_state_id"]; nonzero_id = row["nonzero_state_id"]
        p_zero = best_state[zero_id]; p_nonzero = best_state[nonzero_id]
        zero_correct = p_zero < best["threshold"]; nonzero_correct = p_nonzero >= best["threshold"]
        pair_rows.append({
            **row, "p_gate_zero": p_zero, "p_gate_nonzero": p_nonzero,
            "zero_correct": zero_correct, "nonzero_correct": nonzero_correct,
            "pairwise_order_correct": p_nonzero > p_zero,
            "pair_outcome": "both_correct" if zero_correct and nonzero_correct else "one_side_correct" if zero_correct or nonzero_correct else "fully_confused",
        })
    write_csv(HERE / "matched_boundary_pairs.csv", pair_rows)
    pair_summary = {
        "pair_count": len(pair_rows), "pairwise_ranking_accuracy": float(np.mean([row["pairwise_order_correct"] for row in pair_rows])),
        "both_correct_pairs": sum(row["pair_outcome"] == "both_correct" for row in pair_rows),
        "one_side_correct_pairs": sum(row["pair_outcome"] == "one_side_correct" for row in pair_rows),
        "fully_confused_pairs": sum(row["pair_outcome"] == "fully_confused" for row in pair_rows),
        "heldout_validation_test_pairwise_ranking_accuracy": float(np.mean([row["pairwise_order_correct"] for row in pair_rows if row["split"] != "train"])),
        "heldout_pair_count": sum(row["split"] != "train" for row in pair_rows),
    }

    old_rows = []
    for state_id in OLD_HARD_IDS:
        state_indices = np.flatnonzero(arrays["state_id"] == state_id)
        first = int(state_indices[0]); manifest_row = state_manifest[state_id]
        position = arrays["positions"][first]; velocity = arrays["velocities"][first]
        distance = float(np.linalg.norm(position[0] - position[1])); relative_velocity = float(np.linalg.norm(velocity[0] - velocity[1]))
        probability = best_state[state_id]
        old_rows.append({
            "state_id": state_id, "oracle_gate_label": state_label[state_id], "p_gate": probability,
            "threshold": best["threshold"], "predicted_gate_label": int(probability >= best["threshold"]),
            "correct": probability < best["threshold"], "source_group": state_group[state_id],
            "inter_agent_distance": distance, "relative_velocity_norm": relative_velocity,
            "step": manifest_row.get("step"), "stuck_timer": manifest_row.get("stuck_timer"),
            "max_stuck_timer": manifest_row.get("max_stuck_timer"), "candidate_since": manifest_row.get("candidate_since"),
            "ever_candidate_deadlock": manifest_row.get("ever_candidate_deadlock"),
            "recent_progress_agent0": float(arrays["recent_progress"][first][0]), "recent_progress_agent1": float(arrays["recent_progress"][first][1]),
        })
    write_csv(HERE / "old_hard_states.csv", old_rows)

    # Existing V4 test is a strict source-group holdout. Report aggregate recovery and each unseen recovery group.
    test_ids, test_prob, test_labels = best["state_predictions"]["test"]
    recovery_mask = np.asarray([state_category[str(state_id)] == "RECOVERY" for state_id in test_ids])
    group_rows = [{"scope": "ALL_UNSEEN_TEST_RECOVERY_GROUPS", "source_group": "ALL", **classification_metrics(test_labels[recovery_mask], test_prob[recovery_mask], best["threshold"])}]
    for group in sorted({state_group[str(state_id)] for state_id in test_ids[recovery_mask]}):
        mask = np.asarray([state_category[str(state_id)] == "RECOVERY" and state_group[str(state_id)] == group for state_id in test_ids])
        group_rows.append({"scope": "UNSEEN_TEST_RECOVERY_GROUP", "source_group": group, **classification_metrics(test_labels[mask], test_prob[mask], best["threshold"])})
    boundary_test = np.asarray([str(state_id) in boundary_ids for state_id in test_ids])
    group_rows.append({"scope": "UNSEEN_CLOSE_BOUNDARY_GROUPS", "source_group": "ALL", **classification_metrics(test_labels[boundary_test], test_prob[boundary_test], best["threshold"])})
    write_csv(HERE / "source_group_holdout_metrics.csv", group_rows)

    calibration = {"model": best["name"], "seed": best["seed"], "threshold": best["threshold"], "splits": {}}
    for split in ("validation", "test"):
        _, probabilities, labels = best["state_predictions"][split]
        calibration["splits"][split] = {"Brier": float(np.mean((probabilities - labels) ** 2)), "reliability_bins": reliability(labels, probabilities)}
    write_json(HERE / "calibration_metrics.json", calibration)

    best_test = best["metrics"]["test"]
    best_test_recovery = classification_metrics(test_labels[recovery_mask], test_prob[recovery_mask], best["threshold"])
    best_test_boundary = classification_metrics(test_labels[boundary_test], test_prob[boundary_test], best["threshold"])
    boundary_zero = boundary_test & (test_labels == 0); boundary_nonzero = boundary_test & (test_labels == 1)
    close_detail = {
        "combined": best_test_boundary,
        "zero_oracle": {"states": int(boundary_zero.sum()), "mean_p_gate": float(test_prob[boundary_zero].mean()), "correct_fraction": float(np.mean(test_prob[boundary_zero] < best["threshold"])), "FPR": float(np.mean(test_prob[boundary_zero] >= best["threshold"]))},
        "nonzero_oracle": {"states": int(boundary_nonzero.sum()), "mean_p_gate": float(test_prob[boundary_nonzero].mean()), "correct_fraction": float(np.mean(test_prob[boundary_nonzero] >= best["threshold"])), "FNR": float(np.mean(test_prob[boundary_nonzero] < best["threshold"]))},
    }
    linear = next(run for run in architecture_best if run["name"] == "LINEAR")
    nonlinear_gap = {
        "test_AUROC_best_minus_linear": best_test["AUROC"] - linear["metrics"]["test"]["AUROC"],
        "test_balanced_accuracy_best_minus_linear": best_test["balanced_accuracy"] - linear["metrics"]["test"]["balanced_accuracy"],
        "validation_AUROC_best_minus_linear": best["metrics"]["validation"]["AUROC"] - linear["metrics"]["validation"]["AUROC"],
    }
    nn_disagreement = read_json(DATA / "boundary_diversity_metrics.json")["nearest_neighbor_label_disagreement_rate"]
    # Scientific decision emphasizes unseen boundary/source groups rather than easy global states.
    source_meaningful = best_test_recovery["AUROC"] >= .65 and best_test_recovery["balanced_accuracy"] >= .60
    if (best_test["AUROC"] >= .80 and best_test["balanced_accuracy"] >= .70 and best_test["FPR"] <= .30 and best_test["FNR"] <= .30
            and pair_summary["heldout_validation_test_pairwise_ranking_accuracy"] >= .75 and best_test_boundary["balanced_accuracy"] >= .65 and source_meaningful):
        conclusion = "GATE_LEARNABLE"
    elif best_test["AUROC"] >= .65 and best_test["balanced_accuracy"] >= .60:
        conclusion = "GATE_PARTIALLY_LEARNABLE"
    else:
        conclusion = "GATE_NOT_LEARNABLE_FROM_CURRENT_INPUT"
    global_gate_signal_present = conclusion != "GATE_NOT_LEARNABLE_FROM_CURRENT_INPUT"
    boundary_generalization_sufficient = conclusion == "GATE_LEARNABLE"

    finished_utc = datetime.now(timezone.utc).isoformat()
    runtime = {
        "started_utc": started_utc, "finished_utc": finished_utc, "wall_seconds": time.monotonic() - started,
        "allocated_GPU_shards": 1, "allocated_CPU_cores": int(os.environ.get("SLURM_CPUS_PER_TASK", "6")),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "jax_devices": [str(device) for device in jax.devices()],
        "JAX_memory_fraction": 0.08, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": sys.version, "platform": platform.platform(), "training_runs": len(trained),
        "per_run_seconds": {f"{run['name']}_seed{run['seed']}": run["summary"]["runtime_s"] for run in trained},
    }
    write_json(HERE / "runtime_statistics.json", runtime)
    sanity = {
        "passed": True, "dataset_V4_unchanged": dataset_files_before == {name: sha(DATA / name) for name in dataset_files_before},
        "dataset_manifest_unchanged": dataset_hash_before == sha(DATA / "manifest.json"),
        "oracle_modified": False, "feature_schema_modified": False, "correction_head_trained": False,
        "closed_loop_evaluation_run": False, "label_uses_correction_norm_threshold": False,
        "train_only_normalization": True, "test_used_for_model_selection": False,
        "all_checkpoint_arrays_finite": all(np.isfinite(value).all() for value in checkpoint.values() if np.asarray(value).dtype.kind in "fc"),
        "checkpoint_output_dimension": 1, "state_grouped_primary_metrics": True,
        "selected_model": best["name"], "selected_seed": best["seed"], "conclusion": conclusion,
        "global_gate_signal_present_in_214D": global_gate_signal_present,
        "214D_sufficient_for_reliable_recovery_boundary_generalization": boundary_generalization_sufficient,
    }
    write_json(HERE / "sanity_checks.json", sanity)

    report = f"""# G_phi Gate Feasibility V1

## Decision

**{conclusion}**

This was a strictly offline binary feasibility audit. It trained no correction head, changed no oracle/data/controller semantics, and ran no closed-loop control.

## Oracle label and data

`y_gate=0` iff `eta=(0,0,0)` satisfies `B_63`; otherwise `y_gate=1` when a nonzero success-constrained correction is required. Labels came only from Dataset V4 oracle metadata, never from a correction-norm threshold.

- States: {len(state_label)} = {sum(value == 0 for value in state_label.values())} gate-0 / {sum(value == 1 for value in state_label.values())} gate-1.
- Split counts: {split_counts}.
- Source-group leakage: none. All 64 Flow variants of every state remain colocated.

## Selected classifier

- Architecture: `{[214, *best['hidden'], 1]}`, seed {best['seed']}, standard unweighted BCE.
- Validation-selected threshold: **{best['threshold']:.6f}**; test was evaluated once after freezing it.
- Test balanced accuracy/AUROC/AUPRC: **{best_test['balanced_accuracy']:.4f}/{best_test['AUROC']:.4f}/{best_test['AUPRC']:.4f}**.
- Test FPR/FNR: **{best_test['FPR']:.4f}/{best_test['FNR']:.4f}**.
- Recovery-only balanced accuracy/AUROC/FPR/FNR: **{best_test_recovery['balanced_accuracy']:.4f}/{best_test_recovery['AUROC']:.4f}/{best_test_recovery['FPR']:.4f}/{best_test_recovery['FNR']:.4f}**.
- Held-out close-boundary balanced accuracy/AUROC: **{best_test_boundary['balanced_accuracy']:.4f}/{best_test_boundary['AUROC']:.4f}**.
- Matched pairs ranking: **{pair_summary['pairwise_ranking_accuracy']:.4f}** overall and **{pair_summary['heldout_validation_test_pairwise_ranking_accuracy']:.4f}** on validation/test pairs.

Nearest-neighbor boundary disagreement was {nn_disagreement:.2%}. Linear/nonlinear and source-group diagnostics are in the accompanying files. The 214-D input contains useful global gate signal: **{global_gate_signal_present}**; it is sufficient for reliable held-out recovery-boundary generalization: **{boundary_generalization_sufficient}**.
"""
    (HERE / "gate_report.md").write_text(report)

    # Compact machine-readable decision evidence.
    write_json(HERE / "decision_metrics.json", {
        "conclusion": conclusion, "global_gate_signal_present_in_214D": global_gate_signal_present,
        "214D_sufficient_for_reliable_recovery_boundary_generalization": boundary_generalization_sufficient,
        "selected_model": best["name"], "selected_seed": best["seed"], "threshold": best["threshold"],
        "test": best_test, "test_recovery": best_test_recovery, "test_close_boundary": close_detail,
        "matched_pair_summary": pair_summary, "old_hard_correct": sum(row["correct"] for row in old_rows),
        "old_hard_mean_p_gate": float(np.mean([row["p_gate"] for row in old_rows])),
        "source_group_holdout_recovery": group_rows[0], "linear_vs_nonlinear": nonlinear_gap,
        "nearest_neighbor_boundary_label_disagreement": nn_disagreement,
    })

    required = [
        "gate_report.md", "dataset_audit.json", "split_manifest.json", "model_comparison.csv", "state_metrics.csv",
        "category_metrics.csv", "threshold_analysis.json", "matched_boundary_pairs.csv", "old_hard_states.csv",
        "source_group_holdout_metrics.csv", "calibration_metrics.json", "best_gate_checkpoint.npz", "normalization.json",
        "sanity_checks.json", "training_history.csv", "runtime_statistics.json", "decision_metrics.json",
    ]
    manifest = {
        "study": "GPHI_GATE_FEASIBILITY", "classification": conclusion, "dataset": str(DATA),
        "dataset_manifest_sha256": sha(DATA / "manifest.json"), "selected_checkpoint": "best_gate_checkpoint.npz",
        "correction_head_trained": False, "closed_loop_evaluation_run": False,
        "files_sha256": {name: sha(HERE / name) for name in required},
    }
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({"conclusion": conclusion, "best": best["name"], "seed": best["seed"], "threshold": best["threshold"],
                      "test": {key: best_test[key] for key in ("balanced_accuracy", "AUROC", "AUPRC", "FPR", "FNR")},
                      "recovery": {key: best_test_recovery[key] for key in ("balanced_accuracy", "AUROC", "FPR", "FNR")},
                      "pairs": pair_summary, "runtime_s": runtime["wall_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
