"""Source-group-held-out cross-validation on existing hard stable gate states only."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONF = ROOT / "diagnostics/gphi_gate_confidence_aware_v1"
DATA = ROOT / "diagnostics/gphi_training_dataset_v4"
ORACLE = ROOT / "diagnostics/oracle_boundary_confidence_audit"
FEATURE = ROOT / "diagnostics/stable_oracle_feature_audit"
PREVIOUS = ROOT / "diagnostics/gphi_gate_feasibility_v1"
SEEDS = (17, 23, 41)
MODELS = {"LINEAR": [], "MLP_64x64": [64, 64]}


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
        if isinstance(item, float) and not math.isfinite(item):
            return None
        raise TypeError(type(item).__name__)

    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=convert) + "\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    keys = fieldnames or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sigmoid(x):
    x = np.asarray(x, np.float64)
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata(p, method="average")
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def auprc(y: np.ndarray, p: np.ndarray) -> float:
    pos = int(y.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-p, kind="stable")
    yy = y[order]
    pp = p[order]
    tp = fp = 0
    previous_recall = 0.0
    ap = 0.0
    for end in np.r_[np.flatnonzero(pp[1:] != pp[:-1]) + 1, len(y)]:
        start = tp + fp
        group_positive = int(np.sum(yy[start:end]))
        tp += group_positive
        fp += int(end - start - group_positive)
        recall = tp / pos
        precision = tp / max(tp + fp, 1)
        ap += (recall - previous_recall) * precision
        previous_recall = recall
    return float(ap)


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    predicted = (p >= threshold).astype(int)
    tp = int(np.sum((predicted == 1) & (y == 1)))
    tn = int(np.sum((predicted == 0) & (y == 0)))
    fp = int(np.sum((predicted == 1) & (y == 0)))
    fn = int(np.sum((predicted == 0) & (y == 1)))
    pos = tp + fn
    neg = tn + fp
    recall = tp / pos if pos else float("nan")
    specificity = tn / neg if neg else float("nan")
    balanced = (recall + specificity) / 2 if pos and neg else float("nan")
    return {
        "state_count": len(y),
        "stable_zero": neg,
        "stable_nonzero": pos,
        "threshold": float(threshold),
        "balanced_accuracy": balanced,
        "AUROC": auroc(y, p),
        "AUPRC": auprc(y, p),
        "accuracy": float(np.mean(predicted == y)),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": recall,
        "specificity": specificity,
        "FPR": fp / neg if neg else float("nan"),
        "FNR": fn / pos if pos else float("nan"),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Brier": float(np.mean((p - y) ** 2)),
        "mean_probability": float(np.mean(p)),
    }


def select_threshold(y: np.ndarray, p: np.ndarray) -> float:
    unique = np.unique(p)
    candidates = np.r_[
        np.nextafter(unique[0], -np.inf),
        (unique[:-1] + unique[1:]) / 2,
        np.nextafter(unique[-1], np.inf),
        0.5,
    ]
    rows = [metrics(y, p, float(threshold)) for threshold in np.unique(candidates)]
    best = max(
        rows,
        key=lambda row: (
            row["balanced_accuracy"],
            -max(row["FPR"], row["FNR"]),
            -abs(row["FPR"] - row["FNR"]),
            -abs(row["threshold"] - 0.5),
        ),
    )
    return float(best["threshold"])


def aggregate_state(probability: np.ndarray, state_ids: np.ndarray, label: dict[str, int]):
    grouped = defaultdict(list)
    for value, state_id in zip(probability, state_ids):
        grouped[str(state_id)].append(float(value))
    ids = np.asarray(sorted(grouped), dtype=str)
    p = np.asarray([np.mean(grouped[state_id]) for state_id in ids])
    y = np.asarray([label[state_id] for state_id in ids], dtype=int)
    return ids, p, y


def state_bce(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p)))


def init_params(key, dimensions: list[int]):
    params = []
    keys = jax.random.split(key, len(dimensions) - 1)
    for layer_key, fan_in, fan_out in zip(keys, dimensions[:-1], dimensions[1:]):
        limit = math.sqrt(6 / (fan_in + fan_out))
        params.append(
            {
                "w": jax.random.uniform(
                    layer_key, (fan_in, fan_out), minval=-limit, maxval=limit
                ),
                "b": jnp.zeros((fan_out,), jnp.float32),
            }
        )
    return params


def logits(params, x):
    value = x
    for layer in params[:-1]:
        value = jax.nn.silu(value @ layer["w"] + layer["b"])
    return (value @ params[-1]["w"] + params[-1]["b"]).reshape(-1)


def predict(params, x: np.ndarray) -> np.ndarray:
    return np.asarray(jax.nn.sigmoid(logits(params, jnp.asarray(x, jnp.float32))))


def normalization(features: np.ndarray, schema: dict):
    binary = np.zeros(214, bool)
    for segment in schema["segments"]:
        if segment["unit"] == "boolean":
            start = int(segment["offset"])
            binary[start : start + int(segment["length"])] = True
    mean = features.mean(0)
    scale = features.std(0)
    scale[scale < 1e-8] = 1.0
    mean[binary] = 0.0
    scale[binary] = 1.0
    return mean, scale, binary


def train_model(
    hidden: list[int],
    seed: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_ids: np.ndarray,
    x_validation: np.ndarray,
    validation_ids: np.ndarray,
    label: dict[str, int],
):
    params = init_params(jax.random.PRNGKey(seed), [214, *hidden, 1])
    optimizer = optax.adamw(learning_rate=1e-3, weight_decay=1e-5)
    optimizer_state = optimizer.init(params)
    x_train_jax = jnp.asarray(x_train, jnp.float32)
    y_train_jax = jnp.asarray(y_train, jnp.float32)

    @jax.jit
    def step(current, state):
        def loss_fn(candidate):
            return jnp.mean(
                optax.sigmoid_binary_cross_entropy(
                    logits(candidate, x_train_jax), y_train_jax
                )
            )

        loss, gradients = jax.value_and_grad(loss_fn)(current)
        updates, state = optimizer.update(gradients, state, current)
        return optax.apply_updates(current, updates), state, loss

    best = params
    best_validation_bce = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, 1201):
        params, optimizer_state, _ = step(params, optimizer_state)
        if epoch == 1 or epoch % 10 == 0:
            _, validation_probability, validation_y = aggregate_state(
                predict(params, x_validation), validation_ids, label
            )
            validation_bce = state_bce(validation_y, validation_probability)
            if validation_bce < best_validation_bce - 1e-6:
                best_validation_bce = validation_bce
                best_epoch = epoch
                best = jax.tree_util.tree_map(
                    lambda value: np.asarray(value).copy(), params
                )
                stale = 0
            else:
                stale += 1
            if stale >= 35:
                break
    return best, best_epoch, best_validation_bce


def choose_validation_groups(
    rows: list[dict], outer_group: str, fraction: float = 0.18
) -> list[str]:
    """Deterministic label-stratified greedy group split, independent of test predictions."""
    candidates = defaultdict(list)
    for row in rows:
        if row["source_group"] != outer_group:
            candidates[row["source_group"]].append(row)
    total = np.asarray(
        [
            sum(len(value) for value in candidates.values()),
            sum(int(row["original_gate_label"]) == 0 for value in candidates.values() for row in value),
            sum(int(row["original_gate_label"]) == 1 for value in candidates.values() for row in value),
        ],
        float,
    )
    target = fraction * total
    vectors = {
        group: np.asarray(
            [
                len(value),
                sum(int(row["original_gate_label"]) == 0 for row in value),
                sum(int(row["original_gate_label"]) == 1 for row in value),
            ],
            float,
        )
        for group, value in candidates.items()
    }

    def score(vector):
        return float(np.sum(((vector - target) / np.maximum(target, 1.0)) ** 2))

    selected: list[str] = []
    current = np.zeros(3)
    remaining = set(candidates)
    while remaining:
        ranked = sorted(
            remaining,
            key=lambda group: (
                score(current + vectors[group]),
                hashlib.sha256(f"{outer_group}:{group}".encode()).hexdigest(),
            ),
        )
        candidate = ranked[0]
        if selected and score(current + vectors[candidate]) >= score(current):
            break
        selected.append(candidate)
        current += vectors[candidate]
        remaining.remove(candidate)
    for class_index in (1, 2):
        if current[class_index] == 0:
            eligible = [
                group
                for group in remaining
                if vectors[group][class_index] > 0
            ]
            candidate = min(
                eligible,
                key=lambda group: (
                    vectors[group][0],
                    hashlib.sha256(f"{outer_group}:{group}".encode()).hexdigest(),
                ),
            )
            selected.append(candidate)
            current += vectors[candidate]
            remaining.remove(candidate)
    if not selected or current[1] == 0 or current[2] == 0:
        raise RuntimeError(("invalid inner validation", outer_group, selected, current))
    return sorted(selected)


def finite_or_none(value):
    value = float(value)
    return value if math.isfinite(value) else None


def bootstrap_metrics(rows: list[dict], bootstrap_seed: int, repetitions: int = 10000):
    groups = sorted({row["heldout_source_group"] for row in rows})
    by_group = {
        group: [row for row in rows if row["heldout_source_group"] == group]
        for group in groups
    }
    random = np.random.default_rng(bootstrap_seed)
    names = ("balanced_accuracy", "AUROC", "AUPRC", "FPR", "FNR", "accuracy")
    samples = {name: [] for name in names}
    for _ in range(repetitions):
        chosen = random.choice(groups, size=len(groups), replace=True)
        block = [row for group in chosen for row in by_group[str(group)]]
        y = np.asarray([int(row["oracle_label"]) for row in block])
        p = np.asarray([float(row["p_gate"]) for row in block])
        # Thresholds legitimately differ across outer folds; preserve each OOF decision.
        predicted = np.asarray([int(row["predicted_label"]) for row in block])
        tp = int(np.sum((predicted == 1) & (y == 1)))
        tn = int(np.sum((predicted == 0) & (y == 0)))
        fp = int(np.sum((predicted == 1) & (y == 0)))
        fn = int(np.sum((predicted == 0) & (y == 1)))
        pos = tp + fn
        neg = tn + fp
        values = {
            "balanced_accuracy": (tp / pos + tn / neg) / 2 if pos and neg else float("nan"),
            "AUROC": auroc(y, p),
            "AUPRC": auprc(y, p),
            "FPR": fp / neg if neg else float("nan"),
            "FNR": fn / pos if pos else float("nan"),
            "accuracy": float(np.mean(predicted == y)),
        }
        for name, value in values.items():
            if math.isfinite(value):
                samples[name].append(value)
    result = {
        "method": "source-group bootstrap with replacement",
        "repetitions": repetitions,
        "source_group_count": len(groups),
    }
    for name, values in samples.items():
        result[name] = {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "valid_replicates": len(values),
        }
    return result


def pooled_from_rows(rows: list[dict]) -> dict:
    y = np.asarray([int(row["oracle_label"]) for row in rows])
    p = np.asarray([float(row["p_gate"]) for row in rows])
    predicted = np.asarray([int(row["predicted_label"]) for row in rows])
    # OOF operating thresholds differ by fold, so classification statistics use saved OOF decisions.
    tp = int(np.sum((predicted == 1) & (y == 1)))
    tn = int(np.sum((predicted == 0) & (y == 0)))
    fp = int(np.sum((predicted == 1) & (y == 0)))
    fn = int(np.sum((predicted == 0) & (y == 1)))
    pos = tp + fn
    neg = tn + fp
    return {
        "state_count": len(y),
        "stable_zero": neg,
        "stable_nonzero": pos,
        "balanced_accuracy": (tp / pos + tn / neg) / 2,
        "AUROC": auroc(y, p),
        "AUPRC": auprc(y, p),
        "FPR": fp / neg,
        "FNR": fn / pos,
        "accuracy": float(np.mean(predicted == y)),
        "correct": int(np.sum(predicted == y)),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Brier": float(np.mean((p - y) ** 2)),
    }


def main() -> None:
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    jax.config.update("jax_enable_x64", True)
    HERE.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "confidence_dataset": CONF / "oracle_confidence_dataset.csv",
        "confidence_manifest": CONF / "manifest.json",
        "dataset_samples": DATA / "samples.npz",
        "dataset_manifest": DATA / "manifest.json",
        "feature_schema": DATA / "feature_schema.json",
        "oracle_stability": ORACLE / "b63_resampling_stability.csv",
        "feature_nearest_neighbors": FEATURE / "nearest_neighbor_analysis.csv",
        "previous_gate_metrics": PREVIOUS / "state_metrics.csv",
        "previous_gate_decision": PREVIOUS / "decision_metrics.json",
    }
    before = {name: sha(path) for name, path in source_paths.items()}

    confidence_rows = list(csv.DictReader(source_paths["confidence_dataset"].open()))
    stable_rows = [
        row
        for row in confidence_rows
        if row["oracle_confidence_class"]
        in ("ORACLE_STABLE_ZERO", "ORACLE_STABLE_NONZERO")
    ]
    ambiguous_ids = {
        row["state_id"]
        for row in confidence_rows
        if row["oracle_confidence_class"] == "ORACLE_AMBIGUOUS"
    }
    state = {row["state_id"]: row for row in stable_rows}
    label = {state_id: int(row["original_gate_label"]) for state_id, row in state.items()}
    stability_rows = list(csv.DictReader(source_paths["oracle_stability"].open()))
    difficult_ids = sorted(
        row["state_id"]
        for row in stability_rows
        if row["original_label_stable_at_95pct"] == "True"
        and (
            "MATCHED_BOUNDARY" in row["audit_tags"]
            or "OLD_HARD_ZERO" in row["audit_tags"]
        )
    )
    if len(difficult_ids) != 13 or not set(difficult_ids) <= set(state):
        raise RuntimeError(("unexpected difficult stable set", len(difficult_ids), set(difficult_ids) - set(state)))
    difficult_groups = sorted({state[state_id]["source_group"] for state_id in difficult_ids})
    if len(difficult_groups) != 6:
        raise RuntimeError(("unexpected difficult group count", difficult_groups))

    previous_decision = read_json(source_paths["previous_gate_decision"])
    previous_model = previous_decision["selected_model"]
    previous_threshold = float(previous_decision["threshold"])
    previous_probability = {
        row["state_id"]: float(row["p_gate"])
        for row in csv.DictReader(source_paths["previous_gate_metrics"].open())
        if row["model"] == previous_model
    }
    stability_by_id = {row["state_id"]: row for row in stability_rows}
    difficult_rows = []
    for state_id in difficult_ids:
        row = state[state_id]
        old_p = previous_probability[state_id]
        difficult_rows.append(
            {
                "state_id": state_id,
                "oracle_label": label[state_id],
                "category": row["category"],
                "source_trajectory": row["source_trajectory"],
                "source_group": row["source_group"],
                "audit_tags": stability_by_id[state_id]["audit_tags"],
                "Q0": float(row["Q0"]),
                "oracle_label_reproduction_probability": float(
                    row["original_label_reproduction_probability"]
                ),
                "previous_gate_model": previous_model,
                "previous_p_gate": old_p,
                "previous_threshold": previous_threshold,
                "previous_prediction": int(old_p >= previous_threshold),
                "previous_correct": int(old_p >= previous_threshold) == label[state_id],
            }
        )
    write_csv(HERE / "difficult_stable_states.csv", difficult_rows)

    inventory = []
    for group in difficult_groups:
        ids = [state_id for state_id in difficult_ids if state[state_id]["source_group"] == group]
        inventory.append(
            {
                "source_group": group,
                "difficult_state_count": len(ids),
                "stable_zero": sum(label[state_id] == 0 for state_id in ids),
                "stable_nonzero": sum(label[state_id] == 1 for state_id in ids),
                "state_ids": "|".join(ids),
                "source_trajectories": "|".join(sorted({state[state_id]["source_trajectory"] for state_id in ids})),
            }
        )
    write_csv(HERE / "source_group_inventory.csv", inventory)

    with np.load(source_paths["dataset_samples"], allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    if arrays["features"].shape[1] != 214:
        raise RuntimeError(("feature dimension changed", arrays["features"].shape))
    schema = read_json(source_paths["feature_schema"])
    sample_ids = arrays["state_id"].astype(str)
    stable_sample = np.isin(sample_ids, list(state))
    if np.any(np.isin(sample_ids[stable_sample], list(ambiguous_ids))):
        raise RuntimeError("ambiguous sample entered stable mask")
    sample_label = np.asarray([label.get(state_id, -1) for state_id in sample_ids], int)

    fold_manifest = {
        "design": "Leave-One-Difficult-Source-Group-Out",
        "outer_fold_count": len(difficult_groups),
        "difficult_definition": "pre-existing prior-audited MATCHED_BOUNDARY or OLD_HARD_ZERO with stable original label",
        "inner_validation": "deterministic label-stratified greedy source-group split targeting 18% of non-held-out stable states",
        "models": {"LINEAR": [214, 1], "MLP_64x64": [214, 64, 64, 1]},
        "training_seeds": list(SEEDS),
        "loss": "ordinary BCE on oracle-stable states only",
        "optimizer": "AdamW(lr=1e-3, weight_decay=1e-5)",
        "folds": [],
    }
    all_predictions: list[dict] = []
    fold_metrics: list[dict] = []
    run_summaries: list[dict] = []

    for fold_index, outer_group in enumerate(difficult_groups):
        fold_started = time.monotonic()
        validation_groups = choose_validation_groups(stable_rows, outer_group)
        test_ids = sorted(state_id for state_id, row in state.items() if row["source_group"] == outer_group)
        validation_ids = sorted(state_id for state_id, row in state.items() if row["source_group"] in validation_groups)
        train_ids = sorted(
            state_id
            for state_id, row in state.items()
            if row["source_group"] != outer_group
            and row["source_group"] not in validation_groups
        )
        train_groups = {state[state_id]["source_group"] for state_id in train_ids}
        if outer_group in train_groups or outer_group in validation_groups or train_groups & set(validation_groups):
            raise RuntimeError(("fold leakage", outer_group))
        train_index = np.flatnonzero(stable_sample & np.isin(sample_ids, train_ids))
        validation_index = np.flatnonzero(stable_sample & np.isin(sample_ids, validation_ids))
        test_index = np.flatnonzero(stable_sample & np.isin(sample_ids, test_ids))
        norm_mean, norm_scale, binary = normalization(arrays["features"][train_index], schema)
        normalized = ((arrays["features"] - norm_mean) / norm_scale).astype(np.float32)
        fold_id = f"LOGO_{fold_index:02d}_{outer_group}"
        fold_manifest["folds"].append(
            {
                "fold_id": fold_id,
                "heldout_source_group": outer_group,
                "difficult_test_state_ids": sorted(set(test_ids) & set(difficult_ids)),
                "all_stable_test_state_ids": test_ids,
                "validation_source_groups": validation_groups,
                "train_source_groups": sorted(train_groups),
                "counts": {
                    "train_states": len(train_ids),
                    "validation_states": len(validation_ids),
                    "all_stable_test_states": len(test_ids),
                    "difficult_test_states": len(set(test_ids) & set(difficult_ids)),
                    "train_zero": sum(label[state_id] == 0 for state_id in train_ids),
                    "train_nonzero": sum(label[state_id] == 1 for state_id in train_ids),
                    "validation_zero": sum(label[state_id] == 0 for state_id in validation_ids),
                    "validation_nonzero": sum(label[state_id] == 1 for state_id in validation_ids),
                },
                "normalization_fit_state_ids": train_ids,
            }
        )

        for model_name, hidden in MODELS.items():
            seed_validation = []
            seed_test = []
            for seed in SEEDS:
                run_started = time.monotonic()
                params, best_epoch, validation_bce = train_model(
                    hidden,
                    seed,
                    normalized[train_index],
                    sample_label[train_index],
                    sample_ids[train_index],
                    normalized[validation_index],
                    sample_ids[validation_index],
                    label,
                )
                val_ids, val_p, val_y = aggregate_state(
                    predict(params, normalized[validation_index]),
                    sample_ids[validation_index],
                    label,
                )
                heldout_ids, heldout_p, heldout_y = aggregate_state(
                    predict(params, normalized[test_index]),
                    sample_ids[test_index],
                    label,
                )
                threshold = select_threshold(val_y, val_p)
                seed_validation.append((val_ids, val_p, val_y))
                seed_test.append((heldout_ids, heldout_p, heldout_y))
                for state_id, probability, oracle_label in zip(heldout_ids, heldout_p, heldout_y):
                    prediction = int(probability >= threshold)
                    all_predictions.append(
                        {
                            "model": model_name,
                            "fold_id": fold_id,
                            "heldout_source_group": outer_group,
                            "state_id": str(state_id),
                            "is_difficult": str(state_id) in difficult_ids,
                            "oracle_label": int(oracle_label),
                            "p_gate": float(probability),
                            "validation_selected_threshold": threshold,
                            "predicted_label": prediction,
                            "correct": prediction == int(oracle_label),
                            "training_seed": seed,
                        }
                    )
                run_summaries.append(
                    {
                        "fold_id": fold_id,
                        "model": model_name,
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "validation_state_BCE": validation_bce,
                        "runtime_s": time.monotonic() - run_started,
                    }
                )

            val_ids = seed_validation[0][0]
            heldout_ids = seed_test[0][0]
            if any(not np.array_equal(val_ids, item[0]) for item in seed_validation) or any(
                not np.array_equal(heldout_ids, item[0]) for item in seed_test
            ):
                raise RuntimeError(("state aggregation mismatch", fold_id, model_name))
            validation_mean = np.mean([item[1] for item in seed_validation], axis=0)
            heldout_mean = np.mean([item[1] for item in seed_test], axis=0)
            validation_y = seed_validation[0][2]
            heldout_y = seed_test[0][2]
            ensemble_threshold = select_threshold(validation_y, validation_mean)
            for state_id, probability, oracle_label in zip(heldout_ids, heldout_mean, heldout_y):
                prediction = int(probability >= ensemble_threshold)
                all_predictions.append(
                    {
                        "model": model_name,
                        "fold_id": fold_id,
                        "heldout_source_group": outer_group,
                        "state_id": str(state_id),
                        "is_difficult": str(state_id) in difficult_ids,
                        "oracle_label": int(oracle_label),
                        "p_gate": float(probability),
                        "validation_selected_threshold": ensemble_threshold,
                        "predicted_label": prediction,
                        "correct": prediction == int(oracle_label),
                        "training_seed": "SEED_MEAN",
                    }
                )
            for scope, keep in (
                ("ALL_STABLE", np.ones(len(heldout_ids), bool)),
                ("DIFFICULT_STABLE", np.isin(heldout_ids, difficult_ids)),
            ):
                if not np.any(keep):
                    continue
                fold_metric = metrics(
                    heldout_y[keep], heldout_mean[keep], ensemble_threshold
                )
                zero = heldout_mean[keep][heldout_y[keep] == 0]
                nonzero = heldout_mean[keep][heldout_y[keep] == 1]
                fold_metrics.append(
                    {
                        "fold_id": fold_id,
                        "heldout_source_group": outer_group,
                        "model": model_name,
                        "scope": scope,
                        **{key: finite_or_none(value) if isinstance(value, (float, np.floating)) else value for key, value in fold_metric.items()},
                        "mean_p_gate_zero": float(np.mean(zero)) if len(zero) else None,
                        "mean_p_gate_nonzero": float(np.mean(nonzero)) if len(nonzero) else None,
                    }
                )
        print(
            json.dumps(
                {
                    "fold": fold_id,
                    "heldout_group": outer_group,
                    "difficult_states": len(set(test_ids) & set(difficult_ids)),
                    "all_stable_states": len(test_ids),
                    "elapsed_s": time.monotonic() - fold_started,
                }
            ),
            flush=True,
        )

    write_json(HERE / "fold_manifest.json", fold_manifest)
    write_csv(HERE / "out_of_fold_predictions.csv", all_predictions)
    write_csv(HERE / "fold_metrics.csv", fold_metrics)

    ensemble_rows = [row for row in all_predictions if row["training_seed"] == "SEED_MEAN"]
    pooled = {}
    global_vs_difficult = []
    linear_vs_mlp = []
    bootstrap = {}
    for model_name in MODELS:
        pooled[model_name] = {}
        bootstrap[model_name] = {}
        for scope, predicate in (
            ("ALL_STABLE", lambda row: True),
            ("DIFFICULT_STABLE", lambda row: row["is_difficult"] is True),
        ):
            rows = [row for row in ensemble_rows if row["model"] == model_name and predicate(row)]
            result = pooled_from_rows(rows)
            pooled[model_name][scope] = result
            global_vs_difficult.append({"model": model_name, "scope": scope, **result})
            bootstrap_seed = int.from_bytes(hashlib.sha256(f"{model_name}:{scope}".encode()).digest()[:4], "little")
            bootstrap[model_name][scope] = bootstrap_metrics(rows, bootstrap_seed)
    for scope in ("ALL_STABLE", "DIFFICULT_STABLE"):
        linear = pooled["LINEAR"][scope]
        mlp = pooled["MLP_64x64"][scope]
        linear_vs_mlp.append(
            {
                "scope": scope,
                **{f"linear_{key}": linear[key] for key in ("state_count", "balanced_accuracy", "AUROC", "AUPRC", "FPR", "FNR", "accuracy", "correct")},
                **{f"mlp_{key}": mlp[key] for key in ("state_count", "balanced_accuracy", "AUROC", "AUPRC", "FPR", "FNR", "accuracy", "correct")},
                "mlp_minus_linear_balanced_accuracy": mlp["balanced_accuracy"] - linear["balanced_accuracy"],
                "mlp_minus_linear_AUROC": mlp["AUROC"] - linear["AUROC"],
            }
        )
    write_json(HERE / "pooled_metrics.json", pooled)
    write_json(HERE / "bootstrap_confidence_intervals.json", bootstrap)
    write_csv(HERE / "linear_vs_mlp.csv", linear_vs_mlp)
    write_csv(HERE / "global_vs_difficult_metrics.csv", global_vs_difficult)

    sensitivity_rows = []
    for model_name in MODELS:
        for state_id in difficult_ids:
            rows = [
                row
                for row in all_predictions
                if row["model"] == model_name
                and row["state_id"] == state_id
                and row["training_seed"] != "SEED_MEAN"
            ]
            ensemble = next(
                row
                for row in all_predictions
                if row["model"] == model_name
                and row["state_id"] == state_id
                and row["training_seed"] == "SEED_MEAN"
            )
            probabilities = np.asarray([float(row["p_gate"]) for row in rows])
            correct_count = sum(row["correct"] is True for row in rows)
            predicted = [int(row["predicted_label"]) for row in rows]
            description = (
                "ROBUST_MISTAKE"
                if correct_count == 0
                else "ROBUST_CORRECT"
                if correct_count == len(rows)
                else "OPTIMIZATION_SENSITIVE"
            )
            sensitivity_rows.append(
                {
                    "model": model_name,
                    "state_id": state_id,
                    "source_group": state[state_id]["source_group"],
                    "oracle_label": label[state_id],
                    "mean_p_gate": float(np.mean(probabilities)),
                    "std_p_gate": float(np.std(probabilities)),
                    "min_p_gate": float(np.min(probabilities)),
                    "max_p_gate": float(np.max(probabilities)),
                    "seed_predictions": "|".join(map(str, predicted)),
                    "correct_seed_count": correct_count,
                    "seed_count": len(rows),
                    "sensitivity_class": description,
                    "ensemble_p_gate": ensemble["p_gate"],
                    "ensemble_threshold": ensemble["validation_selected_threshold"],
                    "ensemble_prediction": ensemble["predicted_label"],
                    "ensemble_correct": ensemble["correct"],
                }
            )
    write_csv(HERE / "seed_sensitivity.csv", sensitivity_rows)

    nearest = {
        row["state_id"]: row
        for row in csv.DictReader(source_paths["feature_nearest_neighbors"].open())
    }
    repeated_errors = []
    for state_id in difficult_ids:
        mlp = next(
            row
            for row in sensitivity_rows
            if row["model"] == "MLP_64x64" and row["state_id"] == state_id
        )
        if mlp["sensitivity_class"] != "ROBUST_MISTAKE":
            continue
        linear = next(
            row
            for row in sensitivity_rows
            if row["model"] == "LINEAR" and row["state_id"] == state_id
        )
        neighbor = nearest[state_id]
        source = state[state_id]
        repeated_errors.append(
            {
                "state_id": state_id,
                "oracle_label": label[state_id],
                "Q0": float(source["Q0"]),
                "oracle_label_reproduction_probability": float(source["original_label_reproduction_probability"]),
                "source_group": source["source_group"],
                "mlp_mean_p_gate": mlp["mean_p_gate"],
                "mlp_std_p_gate": mlp["std_p_gate"],
                "mlp_min_p_gate": mlp["min_p_gate"],
                "mlp_max_p_gate": mlp["max_p_gate"],
                "linear_sensitivity_class": linear["sensitivity_class"],
                "both_models_robustly_wrong": linear["sensitivity_class"] == "ROBUST_MISTAKE",
                "nearest_stable_same_label_state": neighbor["same_label_nearest_state_id"],
                "nearest_stable_same_label_distance": float(neighbor["same_label_nearest_distance"]),
                "nearest_stable_opposite_label_state": neighbor["opposite_label_nearest_state_id"],
                "nearest_stable_opposite_label_distance": float(neighbor["opposite_label_nearest_distance"]),
                "opposite_over_same_distance_ratio": float(neighbor["opposite_over_same_distance_ratio"]),
            }
        )
    repeated_fields = [
        "state_id", "oracle_label", "Q0", "oracle_label_reproduction_probability", "source_group",
        "mlp_mean_p_gate", "mlp_std_p_gate", "mlp_min_p_gate", "mlp_max_p_gate",
        "linear_sensitivity_class", "both_models_robustly_wrong",
        "nearest_stable_same_label_state", "nearest_stable_same_label_distance",
        "nearest_stable_opposite_label_state", "nearest_stable_opposite_label_distance",
        "opposite_over_same_distance_ratio",
    ]
    write_csv(HERE / "repeated_errors.csv", repeated_errors, repeated_fields)

    mlp_difficult = pooled["MLP_64x64"]["DIFFICULT_STABLE"]
    mlp_ci = bootstrap["MLP_64x64"]["DIFFICULT_STABLE"]
    robust_mistakes = sum(
        row["model"] == "MLP_64x64" and row["sensitivity_class"] == "ROBUST_MISTAKE"
        for row in sensitivity_rows
    )
    if (
        mlp_difficult["balanced_accuracy"] >= 0.75
        and mlp_difficult["AUROC"] >= 0.75
        and mlp_difficult["accuracy"] >= 0.75
        and mlp_ci["balanced_accuracy"]["lower_95"] > 0.5
        and mlp_ci["AUROC"]["lower_95"] > 0.5
        and robust_mistakes <= 2
    ):
        conclusion = "HARD_BOUNDARY_GENERALIZATION_SUPPORTED"
    elif (
        mlp_difficult["balanced_accuracy"] >= 0.60
        or mlp_difficult["AUROC"] >= 0.65
        or mlp_difficult["accuracy"] >= 0.65
    ):
        conclusion = "HARD_BOUNDARY_DATA_SCARCE_BUT_SIGNAL_PRESENT"
    else:
        conclusion = "HARD_BOUNDARY_GENERALIZATION_FAILS"

    all_difficult_seed_mean = [
        row
        for row in ensemble_rows
        if row["model"] == "MLP_64x64" and row["is_difficult"] is True
    ]
    group_error_counts = {
        group: sum(
            row["correct"] is False
            for row in all_difficult_seed_mean
            if row["heldout_source_group"] == group
        )
        for group in difficult_groups
    }
    smallest_next = (
        "Repeat the unchanged gate on one independently defined stable-boundary pool before any correction-head work."
        if conclusion == "HARD_BOUNDARY_DATA_SCARCE_BUT_SIGNAL_PRESENT"
        else "Proceed only to an offline gate-plus-frozen-correction integration audit; do not run closed loop."
        if conclusion == "HARD_BOUNDARY_GENERALIZATION_SUPPORTED"
        else "Audit the recurrently misclassified stable states against their existing full augmented snapshots; do not increase gate capacity."
    )

    after = {name: sha(path) for name, path in source_paths.items()}
    total_runtime = time.monotonic() - started
    runtime = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": total_runtime,
        "training_runs": len(run_summaries),
        "outer_folds": len(difficult_groups),
        "models": list(MODELS),
        "seeds_per_model_per_fold": len(SEEDS),
        "GPU_shards": int(os.environ.get("SLURM_GPUS_ON_NODE", "1").split("(")[0] or 1),
        "allocated_CPU_cores": int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
        "new_oracle_rollouts": 0,
        "python": sys.version,
        "platform": platform.platform(),
        "jax_backend": jax.default_backend(),
        "run_runtime_summary_s": {
            "min": min(row["runtime_s"] for row in run_summaries),
            "median": float(np.median([row["runtime_s"] for row in run_summaries])),
            "max": max(row["runtime_s"] for row in run_summaries),
        },
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    difficult_oof = [
        row
        for row in ensemble_rows
        if row["model"] == "MLP_64x64" and row["is_difficult"] is True
    ]
    difficult_once = Counter(row["state_id"] for row in difficult_oof)
    outer_leaks = []
    normalization_leaks = []
    for fold in fold_manifest["folds"]:
        heldout = fold["heldout_source_group"]
        train_groups = set(fold["train_source_groups"])
        validation_groups = set(fold["validation_source_groups"])
        if heldout in train_groups or heldout in validation_groups or train_groups & validation_groups:
            outer_leaks.append(fold["fold_id"])
        if any(state[state_id]["source_group"] == heldout for state_id in fold["normalization_fit_state_ids"]):
            normalization_leaks.append(fold["fold_id"])
    sanity = {
        "passed": before == after
        and not outer_leaks
        and not normalization_leaks
        and set(difficult_once) == set(difficult_ids)
        and all(count == 1 for count in difficult_once.values())
        and all(math.isfinite(float(row["p_gate"])) for row in all_predictions)
        and not any(state_id in ambiguous_ids for state_id in label),
        "prior_artifacts_unchanged": before == after,
        "new_states_collected": 0,
        "new_oracle_rollouts": 0,
        "oracle_definition_modified": False,
        "feature_schema_modified": False,
        "gate_architectures_modified": False,
        "correction_head_trained": False,
        "closed_loop_run": False,
        "ambiguous_states_used": False,
        "outer_source_group_leakage_folds": outer_leaks,
        "normalization_leakage_folds": normalization_leaks,
        "difficult_states_have_exactly_one_seed_mean_OOF_prediction": all(count == 1 for count in difficult_once.values()),
        "all_predictions_finite": all(math.isfinite(float(row["p_gate"])) for row in all_predictions),
        "difficult_stable_count": len(difficult_ids),
        "difficult_source_group_count": len(difficult_groups),
    }
    write_json(HERE / "sanity_checks.json", sanity)
    if not sanity["passed"]:
        raise RuntimeError(sanity)

    report = f"""# Hard stable boundary source-group cross-validation

## Decision

**{conclusion}**

No states or oracle rollouts were added. The pre-defined difficult stable pool contains {len(difficult_ids)} states from {len(difficult_groups)} source groups ({sum(label[state_id] == 0 for state_id in difficult_ids)} zero / {sum(label[state_id] == 1 for state_id in difficult_ids)} nonzero). Six strict leave-one-source-group-out folds were used. Each fold excluded its outer group from training, normalization, inner validation, early stopping, and threshold selection.

## Pooled difficult out-of-fold result

- MLP 64x64 seed-mean: balanced accuracy {mlp_difficult['balanced_accuracy']:.4f}, AUROC {mlp_difficult['AUROC']:.4f}, AUPRC {mlp_difficult['AUPRC']:.4f}.
- FPR {mlp_difficult['FPR']:.4f}, FNR {mlp_difficult['FNR']:.4f}, accuracy {mlp_difficult['accuracy']:.4f} ({mlp_difficult['correct']}/{mlp_difficult['state_count']}).
- Source-group bootstrap 95% CI: balanced accuracy [{mlp_ci['balanced_accuracy']['lower_95']:.4f}, {mlp_ci['balanced_accuracy']['upper_95']:.4f}], AUROC [{mlp_ci['AUROC']['lower_95']:.4f}, {mlp_ci['AUROC']['upper_95']:.4f}], accuracy [{mlp_ci['accuracy']['lower_95']:.4f}, {mlp_ci['accuracy']['upper_95']:.4f}].
- Robust MLP mistakes across all three seeds: {robust_mistakes}.
- Difficult ensemble errors by source group: {group_error_counts}.

## Linear versus nonlinear and global gap

- Linear difficult balanced accuracy / AUROC: {pooled['LINEAR']['DIFFICULT_STABLE']['balanced_accuracy']:.4f} / {pooled['LINEAR']['DIFFICULT_STABLE']['AUROC']:.4f}.
- MLP difficult balanced accuracy / AUROC: {mlp_difficult['balanced_accuracy']:.4f} / {mlp_difficult['AUROC']:.4f}.
- MLP all-stable held-out-group balanced accuracy / AUROC: {pooled['MLP_64x64']['ALL_STABLE']['balanced_accuracy']:.4f} / {pooled['MLP_64x64']['ALL_STABLE']['AUROC']:.4f} over {pooled['MLP_64x64']['ALL_STABLE']['state_count']} states.

## Interpretation

The pooled result, group bootstrap uncertainty, per-fold behavior, seed sensitivity, and recurrent errors determine the classification above. The smallest justified next step is: {smallest_next}

No correction head was trained and no closed-loop control was run.
"""
    (HERE / "crossval_report.md").write_text(report)

    required = [
        "crossval_report.md",
        "difficult_stable_states.csv",
        "source_group_inventory.csv",
        "fold_manifest.json",
        "out_of_fold_predictions.csv",
        "pooled_metrics.json",
        "fold_metrics.csv",
        "seed_sensitivity.csv",
        "linear_vs_mlp.csv",
        "global_vs_difficult_metrics.csv",
        "repeated_errors.csv",
        "bootstrap_confidence_intervals.json",
        "sanity_checks.json",
        "runtime_statistics.json",
    ]
    manifest = {
        "study": "HARD_STABLE_BOUNDARY_CROSSVAL",
        "classification": conclusion,
        "difficult_stable_states": len(difficult_ids),
        "source_groups": len(difficult_groups),
        "outer_folds": len(difficult_groups),
        "new_oracle_rollouts": 0,
        "source_artifact_sha256": before,
        "files_sha256": {name: sha(HERE / name) for name in required},
    }
    write_json(HERE / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "classification": conclusion,
                "difficult_states": len(difficult_ids),
                "source_groups": len(difficult_groups),
                "MLP_difficult": mlp_difficult,
                "MLP_global": pooled["MLP_64x64"]["ALL_STABLE"],
                "LINEAR_difficult": pooled["LINEAR"]["DIFFICULT_STABLE"],
                "runtime_s": total_runtime,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
