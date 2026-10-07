"""Source-held-out identifiability probes for recovery-entry advantage.

This is an audit-only runner: it writes metrics and out-of-fold predictions,
never a deployable checkpoint.  It refuses to run until branch evidence has
been fail-closed finalized by ``finalize_paired_branches.py``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from branch_common import atomic_csv, atomic_json, atomic_text, canonical_hash, file_hash, verify_semantic_hash


HERE = Path(__file__).resolve().parent
QUERY = HERE / "queried_state_manifest.json"
FINALIZATION = HERE / "branch_finalization_manifest.json"
STATEWISE = HERE / "statewise_recovery_advantage.csv"
PAIRED = HERE / "paired_branch_outcomes.csv"
FEATURE_SCHEMA = Path(
    "/home/zhihan/research/Basin_C1/diagnostics/"
    "gphi_training_dataset_startup_complete_v1/feature_schema.json"
)
FEATURE_DIM = 214
REPEATS = 3
FOLDS = 5
SEEDS = (17, 23, 41)
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
MAX_EPOCHS = 600
PATIENCE_CHECKS = 12
CHECK_EVERY = 10
MAX_ROWS = 256


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def mean_or_none(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(clean)) if clean else None


def std_or_none(values: Iterable[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.std(clean)) if clean else None


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def correlation(left: np.ndarray, right: np.ndarray, *, rank: bool = False) -> float | None:
    left, right = np.asarray(left, float), np.asarray(right, float)
    if rank:
        left, right = _rank(left), _rank(right)
    if len(left) < 2 or np.std(left) <= 1e-14 or np.std(right) <= 1e-14:
        return None
    return finite(np.corrcoef(left, right)[0, 1])


def continuous_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    error = np.asarray(prediction) - np.asarray(target)
    return {
        "count": int(len(target)),
        "pearson": correlation(target, prediction),
        "spearman": correlation(target, prediction, rank=True),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
    }


def auc_roc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, int)
    positives, negatives = int(np.sum(labels == 1)), int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return None
    ranks = _rank(np.asarray(scores, float)) + 1.0
    return float((np.sum(ranks[labels == 1]) - positives * (positives + 1) / 2) / (positives * negatives))


def auc_pr(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, int)
    positives = int(np.sum(labels == 1))
    if positives == 0 or positives == len(labels):
        return None
    order = np.argsort(-np.asarray(scores), kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    precision = tp / np.arange(1, len(labels) + 1)
    return float(np.sum(precision[sorted_labels == 1]) / positives)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels = np.asarray(labels, int)
    predicted = np.asarray(scores) > threshold
    positive = labels == 1
    negative = labels == 0
    tp, fn = int(np.sum(predicted & positive)), int(np.sum(~predicted & positive))
    tn, fp = int(np.sum(~predicted & negative)), int(np.sum(predicted & negative))
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    return {
        "count": int(len(labels)), "beneficial_count": int(np.sum(positive)),
        "harmful_count": int(np.sum(negative)), "auroc": auc_roc(labels, scores),
        "auprc": auc_pr(labels, scores), "threshold": float(threshold),
        "balanced_accuracy": None if sensitivity is None or specificity is None else 0.5 * (sensitivity + specificity),
        "sensitivity": finite(sensitivity), "specificity": finite(specificity),
        "tp": tp, "fn": fn, "tn": tn, "fp": fp,
    }


def choose_binary_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    labels, scores = np.asarray(labels, int), np.asarray(scores, float)
    if len(np.unique(labels)) < 2:
        return 0.0
    unique = np.unique(scores)
    # Finite sentinels preserve the exact all-positive/all-negative decisions
    # while keeping strict-JSON audit artifacts serializable.
    below = np.nextafter(float(unique[0]), -math.inf)
    above = np.nextafter(float(unique[-1]), math.inf)
    candidates = np.concatenate(([below], (unique[:-1] + unique[1:]) / 2, [above]))
    ranked = []
    for threshold in candidates:
        metric = binary_metrics(labels, scores, float(threshold))["balanced_accuracy"]
        ranked.append((-1.0 if metric is None else metric, -float(np.mean(scores > threshold)), -abs(float(threshold)), float(threshold)))
    return max(ranked)[-1]


def policy_metrics(score: np.ndarray, threshold: float, q_n: np.ndarray, q_r: np.ndarray,
                   rescue: np.ndarray, break_rate: np.ndarray) -> dict[str, Any]:
    enter = np.asarray(score) > threshold
    value = np.where(enter, q_r, q_n)
    return {
        "expected_success": float(np.mean(value)),
        "entry_fraction": float(np.mean(enter)),
        "rescue_captured": float(np.mean(enter * rescue)),
        "breaks_introduced": float(np.mean(enter * break_rate)),
        "selected_recovery_states": int(np.sum(enter)),
        "state_count": int(len(enter)),
    }


def choose_policy_threshold(score: np.ndarray, q_n: np.ndarray, q_r: np.ndarray) -> float:
    score = np.asarray(score, float)
    unique = np.unique(score)
    below = np.nextafter(float(unique[0]), -math.inf)
    above = np.nextafter(float(unique[-1]), math.inf)
    candidates = np.concatenate(([below], (unique[:-1] + unique[1:]) / 2, [above]))
    choices = []
    for threshold in candidates:
        enter = score > threshold
        value = float(np.mean(np.where(enter, q_r, q_n)))
        # Conservative tie break: fewer entries, then larger threshold.
        choices.append((value, -float(np.mean(enter)), float(threshold)))
    return max(choices)[-1]


def load_inputs() -> dict[str, Any]:
    required = (QUERY, FINALIZATION, STATEWISE, PAIRED, FEATURE_SCHEMA)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(("required finalized inputs absent; probes must not run", missing))
    query = json.loads(QUERY.read_text())
    verify_semantic_hash(query, QUERY)
    finalized = json.loads(FINALIZATION.read_text())
    verify_semantic_hash(finalized, FINALIZATION)
    if finalized.get("status") != "COMPLETE" or not finalized.get("all_results_complete") or not finalized.get("all_pairs_matched"):
        raise RuntimeError("branch finalization is not complete")
    for name, path in (("statewise_recovery_advantage", STATEWISE), ("paired_branch_outcomes", PAIRED)):
        expected = finalized.get("outputs", {}).get(name, {}).get("sha256")
        if expected != file_hash(path):
            raise RuntimeError((name, "hash mismatch", expected, file_hash(path)))
    if query.get("status") != "COMPLETE_FROZEN" or query.get("source_grouping_key") != "root_source_id":
        raise RuntimeError("queried-state manifest not frozen/source-grouped")
    states = list(query["states"])
    rows = read_csv(STATEWISE)
    paired = read_csv(PAIRED)
    by_id = {row["state_id"]: row for row in states}
    if len(by_id) != len(states) or set(by_id) != {row["state_id"] for row in rows}:
        raise RuntimeError("queried-state/statewise ID mismatch")
    for row in rows:
        source = by_id[row["state_id"]]
        if row["root_source_id"] != source["root_source_id"] or row["state_sha256"] != source["state_sha256"]:
            raise RuntimeError((row["state_id"], "statewise source/hash lineage mismatch"))
    if len(rows) != int(finalized["queried_state_count"]):
        raise RuntimeError("statewise count does not match finalization")
    pair_counts: dict[str, int] = {}
    for row in paired:
        state_id = row["state_id"]
        if state_id not in by_id or row["root_source_id"] != by_id[state_id]["root_source_id"]:
            raise RuntimeError((state_id, "paired source lineage mismatch"))
        pair_counts[state_id] = pair_counts.get(state_id, 0) + 1
    arrays: dict[str, Any] = {
        "state_id": np.asarray([row["state_id"] for row in rows]),
        "root": np.asarray([row["root_source_id"] for row in rows]),
        "x": np.asarray([by_id[row["state_id"]]["h_t"] for row in rows], dtype=np.float64),
        "delta": np.asarray([float(row["delta_q"]) for row in rows]),
        "q_n": np.asarray([float(row["q_n"]) for row in rows]),
        "q_r": np.asarray([float(row["q_r"]) for row in rows]),
        "rescue": np.asarray([float(row["rescue"]) for row in rows]),
        "break": np.asarray([float(row["break"]) for row in rows]),
        "matched": np.asarray([int(row["matched_futures"]) for row in rows]),
    }
    if arrays["x"].shape != (len(rows), FEATURE_DIM) or not all(np.isfinite(arrays[key]).all() for key in ("x", "delta", "q_n", "q_r", "rescue", "break")):
        raise RuntimeError("invalid/nonfinite probe arrays")
    if np.max(np.abs(arrays["delta"] - (arrays["rescue"] - arrays["break"]))) > 1e-12:
        raise RuntimeError("DeltaQ != rescue-break")
    if any(pair_counts.get(row["state_id"]) != int(row["matched_futures"]) for row in rows):
        raise RuntimeError("paired future coverage mismatch")
    for row in states:
        if file_hash(Path(row["state_file"])) != row["state_sha256"]:
            raise RuntimeError((row["state_id"], "materialized state hash mismatch"))
    if len(set(arrays["root"].tolist())) != int(finalized["root_source_count"]):
        raise RuntimeError("root-source count mismatch")
    schema = json.loads(FEATURE_SCHEMA.read_text())
    if sum(int(segment["length"]) for segment in schema["segments"]) != FEATURE_DIM:
        raise RuntimeError("authoritative feature schema no longer totals 214")
    return {"query": query, "finalized": finalized, "states": states, "state_by_id": by_id,
            "rows": rows, "paired": paired, "arrays": arrays, "schema": schema}


def make_group_folds(groups: np.ndarray, repeats: int = REPEATS, folds: int = FOLDS) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unique = np.asarray(sorted(set(groups.tolist())))
    assignments: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []
    for repeat in range(repeats):
        rng = np.random.default_rng(20260926 + repeat)
        shuffled = unique.copy()
        rng.shuffle(shuffled)
        buckets = [sorted(shuffled[index::folds].tolist()) for index in range(folds)]
        if set().union(*map(set, buckets)) != set(unique.tolist()):
            raise RuntimeError("group fold coverage error")
        for fold, test_groups in enumerate(buckets):
            test_set = set(test_groups)
            test = np.asarray([index for index, value in enumerate(groups) if value in test_set], int)
            outer_train = np.asarray([index for index, value in enumerate(groups) if value not in test_set], int)
            outer_groups = sorted(set(groups[outer_train].tolist()))
            vrng = np.random.default_rng(20261926 + repeat * 101 + fold)
            vrng.shuffle(outer_groups)
            n_val = max(1, round(0.2 * len(outer_groups)))
            validation_groups = set(outer_groups[:n_val])
            val = np.asarray([index for index in outer_train if groups[index] in validation_groups], int)
            train = np.asarray([index for index in outer_train if groups[index] not in validation_groups], int)
            sets = [set(groups[index].tolist()) for index in (train, val, test)]
            if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
                raise RuntimeError((repeat, fold, "source leakage"))
            splits.append({"repeat": repeat, "fold": fold, "train": train, "validation": val, "test": test})
            for source in unique:
                split = "test" if source in sets[2] else "validation" if source in sets[1] else "train"
                assignments.append({"repeat": repeat, "fold": fold, "root_source_id": source, "split": split})
    return splits, assignments


def normalize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    x_mean = np.mean(x, axis=0)
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    centered = y - y_mean
    # Dual form is stable and much cheaper when n << d.
    dual = np.linalg.solve(x_centered @ x_centered.T + alpha * np.eye(len(x)), centered)
    weights = x_centered.T @ dual
    return weights, float(y_mean - x_mean @ weights)


def select_ridge(x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray) -> tuple[float, dict[str, Any]]:
    mean, scale = normalize_fit(x[train])
    x_train, x_val = (x[train] - mean) / scale, (x[val] - mean) / scale
    scores = []
    for alpha in RIDGE_ALPHAS:
        weights, intercept = ridge_fit(x_train, y[train], alpha)
        prediction = x_val @ weights + intercept
        scores.append({"alpha": alpha, "validation_mae": float(np.mean(np.abs(prediction - y[val])))})
    selected = min(scores, key=lambda row: (row["validation_mae"], row["alpha"]))
    return float(selected["alpha"]), {"criterion": "inner_validation_MAE", "candidates": scores}


def fit_ridge_outer(x: np.ndarray, y: np.ndarray, outer_train: np.ndarray, alpha: float,
                    target: np.ndarray) -> np.ndarray:
    mean, scale = normalize_fit(x[outer_train])
    weights, intercept = ridge_fit((x[outer_train] - mean) / scale, y[outer_train], alpha)
    return (x[target] - mean) / scale @ weights + intercept


def _load_jax() -> tuple[Any, Any, Any]:
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import jax
    import jax.numpy as jnp
    import optax
    jax.config.update("jax_enable_x64", False)
    return jax, jnp, optax


def fit_mlp(x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray, seed: int) -> tuple[Any, dict[str, Any]]:
    jax, jnp, optax = _load_jax()
    mean, scale = normalize_fit(x[train])
    x_train = ((x[train] - mean) / scale).astype(np.float32)
    x_val = ((x[val] - mean) / scale).astype(np.float32)
    if len(train) > MAX_ROWS or len(val) > MAX_ROWS:
        raise RuntimeError("MLP padding capacity exceeded")

    def padded(values: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
        out = np.zeros((MAX_ROWS, width), np.float32)
        mask = np.zeros(MAX_ROWS, np.float32)
        out[:len(values)] = values
        mask[:len(values)] = 1.0
        return out, mask

    tx, tm = padded(x_train, FEATURE_DIM)
    vx, vm = padded(x_val, FEATURE_DIM)
    ty = np.zeros(MAX_ROWS, np.float32); ty[:len(train)] = y[train]
    vy = np.zeros(MAX_ROWS, np.float32); vy[:len(val)] = y[val]
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 3)
    dimensions = ((FEATURE_DIM, 64), (64, 64), (64, 1))
    params = tuple({
        "w": jax.random.normal(keys[index], shape, dtype=jnp.float32) * math.sqrt(2.0 / shape[0]),
        "b": jnp.zeros((shape[1],), dtype=jnp.float32),
    } for index, shape in enumerate(dimensions))
    optimizer = optax.adamw(1e-3, weight_decay=1e-5)
    state = optimizer.init(params)

    def forward(parameters: Any, values: Any) -> Any:
        hidden = jax.nn.silu(values @ parameters[0]["w"] + parameters[0]["b"])
        hidden = jax.nn.silu(hidden @ parameters[1]["w"] + parameters[1]["b"])
        return (hidden @ parameters[2]["w"] + parameters[2]["b"]).reshape(-1)

    @jax.jit
    def step(parameters: Any, optimizer_state: Any, values: Any, targets: Any, mask: Any) -> tuple[Any, Any, Any]:
        def objective(candidate: Any) -> Any:
            residual = forward(candidate, values) - targets
            return jnp.sum(mask * residual * residual) / jnp.maximum(1.0, jnp.sum(mask))
        loss, gradient = jax.value_and_grad(objective)(parameters)
        updates, optimizer_state = optimizer.update(gradient, optimizer_state, parameters)
        return optax.apply_updates(parameters, updates), optimizer_state, loss

    @jax.jit
    def validation(parameters: Any, values: Any, targets: Any, mask: Any) -> Any:
        residual = forward(parameters, values) - targets
        return jnp.sum(mask * residual * residual) / jnp.maximum(1.0, jnp.sum(mask))

    best_params, best_loss, best_epoch, stale = params, math.inf, 0, 0
    txj, tyj, tmj = map(jnp.asarray, (tx, ty, tm))
    vxj, vyj, vmj = map(jnp.asarray, (vx, vy, vm))
    for epoch in range(1, MAX_EPOCHS + 1):
        params, state, _ = step(params, state, txj, tyj, tmj)
        if epoch % CHECK_EVERY == 0:
            value = float(validation(params, vxj, vyj, vmj))
            if value < best_loss - 1e-8:
                best_params, best_loss, best_epoch, stale = params, value, epoch, 0
            else:
                stale += 1
            if stale >= PATIENCE_CHECKS:
                break

    def predict(values: np.ndarray) -> np.ndarray:
        normalized = ((np.asarray(values) - mean) / scale).astype(np.float32)
        return np.asarray(forward(best_params, jnp.asarray(normalized)), dtype=np.float64)

    return predict, {"seed": seed, "selected_epoch": best_epoch,
                     "validation_mse": best_loss, "epochs_run": epoch,
                     "selection": "minimum inner-validation MSE"}


def fold_probe(arrays: Mapping[str, Any], splits: Sequence[Mapping[str, Any]], model: str,
               target: np.ndarray | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    x = arrays["x"]
    y = arrays["delta"] if target is None else np.asarray(target, float)
    predictions: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for split in splits:
        train, val, test = split["train"], split["validation"], split["test"]
        if model == "ridge":
            alpha, selection = select_ridge(x, y, train, val)
            pred_val = fit_ridge_outer(x, y, train, alpha, val)
            # Preserve calibration: the test predictor is the same train-core
            # model whose validation scores set the threshold (no post-hoc
            # refit that could shift score scale).
            pred_test = fit_ridge_outer(x, y, train, alpha, test)
            seed_records: list[dict[str, Any]] = []
        elif model == "mlp":
            val_members, test_members, seed_records = [], [], []
            for seed in SEEDS:
                predictor, record = fit_mlp(x, y, train, val, seed)
                val_members.append(predictor(x[val])); test_members.append(predictor(x[test]))
                seed_records.append(record)
            pred_val = np.mean(np.stack(val_members), axis=0)
            pred_test = np.mean(np.stack(test_members), axis=0)
            selection = {"criterion": "seed ensemble; each seed minimum inner-validation MSE",
                         "seed_records": seed_records}
        else:
            raise ValueError(model)
        policy_threshold = choose_policy_threshold(pred_val, arrays["q_n"][val], arrays["q_r"][val])
        confident_val = np.abs(arrays["delta"][val]) >= 0.05
        if np.sum(confident_val) >= 2 and len(np.unique((arrays["delta"][val][confident_val] >= 0.05).astype(int))) == 2:
            binary_threshold = choose_binary_threshold(
                (arrays["delta"][val][confident_val] >= 0.05).astype(int), pred_val[confident_val]
            )
            binary_status = "VALID"
        else:
            binary_threshold, binary_status = 0.0, "INSUFFICIENT_INNER_VALIDATION_CLASS_SUPPORT"
        provenance.append({
            "model": model, "repeat": split["repeat"], "fold": split["fold"],
            "train_states": len(train), "validation_states": len(val), "test_states": len(test),
            "train_sources": len(set(arrays["root"][train].tolist())),
            "validation_sources": len(set(arrays["root"][val].tolist())),
            "test_sources": len(set(arrays["root"][test].tolist())),
            "selection": selection, "policy_threshold": float(policy_threshold),
            "policy_threshold_provenance": "inner validation maximized empirical expected success; conservative entry tie-break",
            "binary_threshold": float(binary_threshold), "binary_threshold_status": binary_status,
            "binary_threshold_provenance": "inner-validation balanced accuracy only",
        })
        for position, index in enumerate(test):
            predictions.append({
                "model": model, "repeat": split["repeat"], "fold": split["fold"],
                "state_id": arrays["state_id"][index], "root_source_id": arrays["root"][index],
                "delta_q": float(arrays["delta"][index]), "q_safety": float(arrays["q_n"][index]),
                "q_recovery": float(arrays["q_r"][index]), "rescue": float(arrays["rescue"][index]),
                "break": float(arrays["break"][index]), "prediction": float(pred_test[position]),
                "policy_threshold": float(policy_threshold), "binary_threshold": float(binary_threshold),
                "binary_threshold_status": binary_status,
            })
    return predictions, provenance


def summarize_probe(predictions: Sequence[Mapping[str, Any]], provenance: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_repeat = []
    for repeat in range(REPEATS):
        rows = [row for row in predictions if int(row["repeat"]) == repeat]
        y = np.asarray([row["delta_q"] for row in rows]); pred = np.asarray([row["prediction"] for row in rows])
        per_repeat.append({"repeat": repeat, **continuous_metrics(y, pred)})
    return {
        "evaluation": "repeated 5-fold root-source-held-out OOF",
        "repeats": REPEATS, "folds": FOLDS,
        "per_repeat": per_repeat,
        "mean_across_repeats": {
            key: mean_or_none(row[key] for row in per_repeat)
            for key in ("pearson", "spearman", "mae", "rmse")
        },
        "fold_threshold_and_checkpoint_provenance": list(provenance),
        "oof_predictions": list(predictions),
        "warning": "Fold-local scores are not claimed to be one globally calibrated score.",
    }


def summarize_binary(model_predictions: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "class_definition": {"BENEFICIAL": "DeltaQ >= +0.05", "HARMFUL": "DeltaQ <= -0.05", "AMBIGUOUS": "otherwise"},
        "ambiguous_excluded_only_from_binary_metrics": True,
        "models": {},
    }
    for model, predictions in model_predictions.items():
        repeats = []
        for repeat in range(REPEATS):
            folds = []
            for fold in range(FOLDS):
                rows = [row for row in predictions if int(row["repeat"]) == repeat
                        and int(row["fold"]) == fold and abs(float(row["delta_q"])) >= 0.05]
                labels = np.asarray([float(row["delta_q"]) >= 0.05 for row in rows], int)
                scores = np.asarray([row["prediction"] for row in rows], float)
                threshold = float(rows[0]["binary_threshold"]) if rows else 0.0
                fold_metric = binary_metrics(labels, scores, threshold) if rows else {
                    "count": 0, "beneficial_count": 0, "harmful_count": 0, "auroc": None,
                    "auprc": None, "balanced_accuracy": None, "sensitivity": None, "specificity": None,
                }
                folds.append({"fold": fold, **fold_metric,
                              "threshold_provenance": "fold-local inner validation only"})
            beneficial_count = sum(row["beneficial_count"] for row in folds)
            harmful_count = sum(row["harmful_count"] for row in folds)
            repeats.append({
                "repeat": repeat, "count": beneficial_count + harmful_count,
                "beneficial_count": beneficial_count, "harmful_count": harmful_count,
                "mean_fold_auroc": mean_or_none(row["auroc"] for row in folds),
                "mean_fold_auprc": mean_or_none(row["auprc"] for row in folds),
                "mean_fold_balanced_accuracy": mean_or_none(row["balanced_accuracy"] for row in folds),
                "mean_fold_sensitivity": mean_or_none(row["sensitivity"] for row in folds),
                "mean_fold_specificity": mean_or_none(row["specificity"] for row in folds),
                "folds": folds, "fold_scores_not_pooled": True,
            })
        harmful = max(row["harmful_count"] for row in repeats)
        beneficial = max(row["beneficial_count"] for row in repeats)
        result["models"][model] = {
            "status": "VALID" if min(harmful, beneficial) >= 5 else "CLASS_IMBALANCE_PREVENTS_RELIABLE_BINARY_PROBE",
            "per_repeat": repeats,
            "mean_across_repeats": {
                key.removeprefix("mean_fold_"): mean_or_none(row[key] for row in repeats)
                for key in ("mean_fold_auroc", "mean_fold_auprc", "mean_fold_balanced_accuracy",
                            "mean_fold_sensitivity", "mean_fold_specificity")
            },
            "aggregation_note": "AUROC/AUPRC/BAcc are computed inside each held-out fold, then averaged; fold-model scores are never pooled.",
        }
    return result


def policy_rows(model_predictions: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    for model, predictions in model_predictions.items():
        for repeat in range(REPEATS):
            repeat_rows = [row for row in predictions if int(row["repeat"]) == repeat]
            for fold in range(FOLDS):
                rows = [row for row in repeat_rows if int(row["fold"]) == fold]
                score = np.asarray([row["prediction"] for row in rows])
                threshold = float(rows[0]["policy_threshold"])
                q_n = np.asarray([row["q_safety"] for row in rows]); q_r = np.asarray([row["q_recovery"] for row in rows])
                rescue = np.asarray([row["rescue"] for row in rows]); break_rate = np.asarray([row["break"] for row in rows])
                metrics = policy_metrics(score, threshold, q_n, q_r, rescue, break_rate)
                rows_out.append({"model": model, "repeat": repeat, "fold": fold, "row_kind": "heldout_fold",
                                 "threshold": threshold, "threshold_provenance": "inner_validation_only", **metrics})
            q_n = np.asarray([row["q_safety"] for row in repeat_rows]); q_r = np.asarray([row["q_recovery"] for row in repeat_rows])
            scores = np.asarray([row["prediction"] for row in repeat_rows]); thresholds = np.asarray([row["policy_threshold"] for row in repeat_rows])
            enter = scores > thresholds
            rows_out.extend([
                {"model": model, "repeat": repeat, "fold": "ALL", "row_kind": "aggregate_probe_policy",
                 "threshold": "fold_local", "threshold_provenance": "inner_validation_only",
                 "expected_success": float(np.mean(np.where(enter, q_r, q_n))), "entry_fraction": float(np.mean(enter)),
                 "rescue_captured": float(np.mean(enter * np.asarray([row["rescue"] for row in repeat_rows]))),
                 "breaks_introduced": float(np.mean(enter * np.asarray([row["break"] for row in repeat_rows]))),
                 "selected_recovery_states": int(np.sum(enter)), "state_count": len(repeat_rows)},
                {"model": "ALWAYS_SAFETY", "repeat": repeat, "fold": "ALL", "row_kind": "trivial",
                 "threshold": "NA", "threshold_provenance": "predeclared",
                 "expected_success": float(np.mean(q_n)), "entry_fraction": 0.0, "rescue_captured": 0.0,
                 "breaks_introduced": 0.0, "selected_recovery_states": 0, "state_count": len(q_n)},
                {"model": "ALWAYS_RECOVERY", "repeat": repeat, "fold": "ALL", "row_kind": "trivial",
                 "threshold": "NA", "threshold_provenance": "predeclared",
                 "expected_success": float(np.mean(q_r)), "entry_fraction": 1.0,
                 "rescue_captured": float(np.mean([row["rescue"] for row in repeat_rows])),
                 "breaks_introduced": float(np.mean([row["break"] for row in repeat_rows])),
                 "selected_recovery_states": len(q_r), "state_count": len(q_r)},
                {"model": "HINDSIGHT_STATEWISE_ORACLE", "repeat": repeat, "fold": "ALL", "row_kind": "diagnostic_upper_bound",
                 "threshold": "NA", "threshold_provenance": "not_deployable",
                 "expected_success": float(np.mean(np.maximum(q_n, q_r))),
                 "entry_fraction": float(np.mean(q_r > q_n)),
                 "rescue_captured": float(np.mean((q_r > q_n) * np.asarray([row["rescue"] for row in repeat_rows]))),
                 "breaks_introduced": float(np.mean((q_r > q_n) * np.asarray([row["break"] for row in repeat_rows]))),
                 "selected_recovery_states": int(np.sum(q_r > q_n)), "state_count": len(q_n)},
            ])
    return rows_out


def utility_audit(arrays: Mapping[str, Any], splits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {"definition": "utility = rescue - lambda * break", "lambdas_predeclared": [1, 2, 4],
              "probe": "regularized linear score; alpha and decision threshold inner-validation only", "results": []}
    for coefficient in (1, 2, 4):
        target = arrays["rescue"] - coefficient * arrays["break"]
        predictions, provenance = fold_probe(arrays, splits, "ridge", target=target)
        repeat_metrics = []
        for repeat in range(REPEATS):
            rows = [row for row in predictions if int(row["repeat"]) == repeat]
            truth = target[[np.where(arrays["state_id"] == row["state_id"])[0][0] for row in rows]]
            scores = np.asarray([row["prediction"] for row in rows])
            repeat_metrics.append({"repeat": repeat, **continuous_metrics(truth, scores),
                                   "ranking_spearman": correlation(truth, scores, rank=True)})
        result["results"].append({"lambda": coefficient, "per_repeat": repeat_metrics,
                                  "mean_spearman": mean_or_none(row["spearman"] for row in repeat_metrics),
                                  "fold_provenance": provenance})
    return result


def random_split_diagnostic(arrays: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(20260926)
    order = np.arange(len(arrays["delta"])); rng.shuffle(order)
    n_train, n_val = round(0.60 * len(order)), round(0.20 * len(order))
    train, val, test = order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]
    records = []
    output: dict[str, Any] = {
        "split": "single ordinary random-state 60/20/20 diagnostic",
        "source_overlap": {
            "train_validation": len(set(arrays["root"][train]) & set(arrays["root"][val])),
            "train_test": len(set(arrays["root"][train]) & set(arrays["root"][test])),
            "validation_test": len(set(arrays["root"][val]) & set(arrays["root"][test])),
        }, "models": {}
    }
    alpha, selection = select_ridge(arrays["x"], arrays["delta"], train, val)
    pred_val = fit_ridge_outer(arrays["x"], arrays["delta"], train, alpha, val)
    pred_test = fit_ridge_outer(arrays["x"], arrays["delta"], train, alpha, test)
    threshold = choose_policy_threshold(pred_val, arrays["q_n"][val], arrays["q_r"][val])
    output["models"]["ridge"] = {**continuous_metrics(arrays["delta"][test], pred_test),
                                         "policy": policy_metrics(pred_test, threshold, arrays["q_n"][test], arrays["q_r"][test], arrays["rescue"][test], arrays["break"][test]),
                                         "threshold": threshold, "selection": selection}
    members_val, members_test, seed_records = [], [], []
    for seed in SEEDS:
        predictor, record = fit_mlp(arrays["x"], arrays["delta"], train, val, seed)
        members_val.append(predictor(arrays["x"][val])); members_test.append(predictor(arrays["x"][test])); seed_records.append(record)
    pred_val, pred_test = np.mean(members_val, axis=0), np.mean(members_test, axis=0)
    threshold = choose_policy_threshold(pred_val, arrays["q_n"][val], arrays["q_r"][val])
    output["models"]["mlp"] = {**continuous_metrics(arrays["delta"][test], pred_test),
                                       "policy": policy_metrics(pred_test, threshold, arrays["q_n"][test], arrays["q_r"][test], arrays["rescue"][test], arrays["break"][test]),
                                       "threshold": threshold, "seed_records": seed_records}
    for index in test:
        records.append({"state_id": arrays["state_id"][index], "root_source_id": arrays["root"][index], "split": "test"})
    return output, records


def nearest_neighbor_audit(arrays: Mapping[str, Any], splits: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for split in splits:
        train = np.concatenate((split["train"], split["validation"])); test = split["test"]
        mean, scale = normalize_fit(arrays["x"][train])
        tx, qx = (arrays["x"][train] - mean) / scale, (arrays["x"][test] - mean) / scale
        train_label = np.where(arrays["delta"][train] >= 0.05, 1, np.where(arrays["delta"][train] <= -0.05, -1, 0))
        test_label = np.where(arrays["delta"][test] >= 0.05, 1, np.where(arrays["delta"][test] <= -0.05, -1, 0))
        for local, index in enumerate(test):
            label = int(test_label[local])
            if label == 0:
                continue
            distances = np.sqrt(np.mean((tx - qx[local]) ** 2, axis=1))
            same = np.where(train_label == label)[0]; opposite = np.where(train_label == -label)[0]
            nearest_same = float(np.min(distances[same])) if len(same) else None
            nearest_opposite = float(np.min(distances[opposite])) if len(opposite) else None
            nearest_opposite_id = arrays["state_id"][train[opposite[np.argmin(distances[opposite])]]] if len(opposite) else ""
            k = min(5, len(train)); knn = np.argsort(distances)[:k]
            output.append({
                "repeat": split["repeat"], "fold": split["fold"], "state_id": arrays["state_id"][index],
                "root_source_id": arrays["root"][index], "label": "BENEFICIAL" if label == 1 else "HARMFUL",
                "delta_q": float(arrays["delta"][index]), "nearest_same_distance": nearest_same,
                "nearest_opposite_distance": nearest_opposite, "nearest_opposite_state_id": nearest_opposite_id,
                "opposite_label_frequency_k5": float(np.mean(train_label[knn] == -label)),
                "ambiguous_frequency_k5": float(np.mean(train_label[knn] == 0)),
                "distance": "RMS Euclidean; train-fold normalization only",
            })
    return output


def ridge_ablation(arrays: Mapping[str, Any], splits: Sequence[Mapping[str, Any]], schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    groups = {
        "physical_geometry_kinematics": list(range(0, 40)),
        "flow_safety_actions_and_bases": list(range(40, 56)),
        "absolute_and_remaining_time": list(range(56, 60)),
        "goal_progress_and_monitor": list(range(60, 72)),
        "goal_error_history_tail_41": list(range(72, 154)),
        "projection_and_barrier_state": list(range(154, 214)),
    }
    if sorted(sum(groups.values(), [])) != list(range(FEATURE_DIM)):
        raise RuntimeError("feature blocks do not partition 214 dimensions")
    rows = []
    first_repeat = [split for split in splits if split["repeat"] == 0]
    for omitted, dimensions in [("NONE_BASELINE", [])] + list(groups.items()):
        keep = np.asarray([index for index in range(FEATURE_DIM) if index not in set(dimensions)], int)
        local = dict(arrays); local["x"] = arrays["x"][:, keep]
        # Reuse ridge functions, which are dimension agnostic.
        predictions = []
        for split in first_repeat:
            train, val, test = split["train"], split["validation"], split["test"]
            alpha, _ = select_ridge(local["x"], arrays["delta"], train, val)
            pred_val = fit_ridge_outer(local["x"], arrays["delta"], train, alpha, val)
            pred_test = fit_ridge_outer(local["x"], arrays["delta"], train, alpha, test)
            threshold = choose_policy_threshold(pred_val, arrays["q_n"][val], arrays["q_r"][val])
            for position, index in enumerate(test):
                predictions.append((index, pred_test[position], threshold))
        indices = np.asarray([value[0] for value in predictions]); pred = np.asarray([value[1] for value in predictions]); thresholds = np.asarray([value[2] for value in predictions])
        enter = pred > thresholds
        rows.append({
            "omitted_block": omitted, "omitted_dimension_count": len(dimensions), "retained_dimension_count": len(keep),
            "segment_names": [segment["name"] for segment in schema["segments"]
                              if set(range(int(segment["offset"]), int(segment["offset"]) + int(segment["length"]))) & set(dimensions)],
            **continuous_metrics(arrays["delta"][indices], pred),
            "policy_value": float(np.mean(np.where(enter, arrays["q_r"][indices], arrays["q_n"][indices]))),
            "entry_fraction": float(np.mean(enter)), "evaluation": "repeat-0 5-fold source-held-out ridge",
        })
    return rows


def omitted_field_audit(inputs: Mapping[str, Any], arrays: Mapping[str, Any], splits: Sequence[Mapping[str, Any]], primary_weak: bool) -> str:
    mappings = {
        "positions": "encoded (positions and observation)", "velocities": "encoded",
        "step": "encoded by four timing coordinates", "history_start_step": "encoded",
        "candidate_since": "encoded", "stuck_timer": "encoded", "max_stuck_timer": "encoded",
        "ever_candidate_deadlock": "encoded", "error_history": "last 41 samples encoded; earlier real history omitted",
        "first_success_step": "omitted but invariant -1 at nonterminal queries",
        "first_deadlock_step": "omitted but invariant -1 at nonterminal queries",
        "first_wall_collision_step": "omitted but invariant -1 at nonterminal queries",
        "first_agent_collision_step": "omitted but invariant -1 at nonterminal queries",
        "done": "omitted but invariant false", "complete_real_history": "omitted but invariant true",
    }
    secondary = None
    if primary_weak:
        extras = []
        for state_id in arrays["state_id"]:
            state = inputs["state_by_id"][state_id]
            with np.load(state["state_file"], allow_pickle=False) as payload:
                history = np.asarray(payload["error_history"], float)
            prior = history[:-41] if len(history) > 41 else history[:1]
            extras.append(np.concatenate((history[0], np.mean(prior, axis=0), np.std(prior, axis=0),
                                          np.min(prior, axis=0), np.max(prior, axis=0), history[0] - history[-1])))
        extended = dict(arrays); extended["x"] = np.concatenate((arrays["x"], np.asarray(extras)), axis=1)
        # One repeat, group-held-out ridge secondary probe. Ridge is dimension agnostic.
        predictions = []
        for split in [value for value in splits if value["repeat"] == 0]:
            train, val, test = split["train"], split["validation"], split["test"]
            alpha, _ = select_ridge(extended["x"], arrays["delta"], train, val)
            predictions.extend(zip(test.tolist(), fit_ridge_outer(extended["x"], arrays["delta"], train, alpha, test).tolist()))
        index = np.asarray([row[0] for row in predictions]); pred = np.asarray([row[1] for row in predictions])
        secondary = continuous_metrics(arrays["delta"][index], pred)
    lines = [
        "# Omitted deployment-state field audit", "",
        "The primary probe uses only the frozen 214-D deployment feature. No future or terminal-relative field is included.", "",
        "| Saved field | Representation status |", "|---|---|",
    ] + [f"| `{key}` | {value} |" for key, value in mappings.items()]
    lines += ["", "The only nonconstant deployment-available information omitted from `h_t` is the portion of real goal-error history older than the 41-sample tail. Physical and monitor state used by control are otherwise encoded, often redundantly.", ""]
    if secondary is None:
        lines += ["The predeclared primary-weak condition was not met, so no secondary augmented-state probe was run."]
    else:
        lines += ["Because the primary grouped probes were weak, one secondary source-held-out ridge probe added 12 causal summaries of pre-tail real history (initial, mean, standard deviation, min, max, and total progress for two agents). It did not use future information.", "", f"Secondary metrics: `{json.dumps(secondary, sort_keys=True)}`"]
    return "\n".join(lines) + "\n"


def main() -> None:
    outputs = [HERE / name for name in (
        "group_split_manifest.json", "linear_probe_results.json", "mlp_probe_results.json",
        "binary_probe_results.json", "utility_probe_results.json", "heldout_policy_value.csv",
        "nearest_neighbor_audit.csv", "feature_block_ablation.csv", "omitted_state_field_audit.md",
        "shortcut_split_comparison.csv", "probe_integrity_checks.json",
    )]
    existing = [path for path in outputs if path.exists()]
    # A failed strict-JSON write can leave only the deterministic split
    # manifest. Regenerating that one manifest is safe; any fitted/statistical
    # artifact still causes a fail-closed stop.
    unexpected_existing = [str(path) for path in existing if path != outputs[0]]
    if unexpected_existing:
        raise RuntimeError(("refusing to overwrite probe outputs", unexpected_existing))
    inputs = load_inputs(); arrays = inputs["arrays"]
    splits, assignments = make_group_folds(arrays["root"])
    split_manifest = {
        "schema": "recovery_entry_probe_group_splits_v1", "status": "FROZEN",
        "method": "3 repeats x 5 folds, root-source-held-out; inner validation groups are disjoint",
        "state_count": len(arrays["delta"]), "root_source_count": len(set(arrays["root"].tolist())),
        "assignments": assignments,
    }
    split_manifest["content_sha256"] = canonical_hash(split_manifest)
    atomic_json(outputs[0], split_manifest)

    linear_predictions, linear_provenance = fold_probe(arrays, splits, "ridge")
    mlp_predictions, mlp_provenance = fold_probe(arrays, splits, "mlp")
    linear = summarize_probe(linear_predictions, linear_provenance)
    mlp = summarize_probe(mlp_predictions, mlp_provenance)
    atomic_json(outputs[1], linear); atomic_json(outputs[2], mlp)
    binary = summarize_binary({"ridge": linear_predictions, "mlp": mlp_predictions})
    atomic_json(outputs[3], binary)
    atomic_json(outputs[4], utility_audit(arrays, splits))
    policies = policy_rows({"ridge": linear_predictions, "mlp": mlp_predictions})
    atomic_csv(outputs[5], policies, list(policies[0]))
    nearest = nearest_neighbor_audit(arrays, splits)
    atomic_csv(outputs[6], nearest, list(nearest[0]) if nearest else ["repeat", "fold", "state_id"])
    ablations = ridge_ablation(arrays, splits, inputs["schema"])
    atomic_csv(outputs[7], ablations, list(ablations[0]))

    best_group_spearman = max(abs(float(value or 0.0)) for value in (
        linear["mean_across_repeats"]["spearman"], mlp["mean_across_repeats"]["spearman"]
    ))
    aggregate_policy = [row for row in policies if row["row_kind"] == "aggregate_probe_policy"]
    best_probe_value = max(float(row["expected_success"]) for row in aggregate_policy)
    best_trivial = max(float(np.mean(arrays["q_n"])), float(np.mean(arrays["q_r"])))
    primary_weak = best_group_spearman < 0.30 and best_probe_value - best_trivial < 0.02
    atomic_text(outputs[8], omitted_field_audit(inputs, arrays, splits, primary_weak))

    random_result, random_assignments = random_split_diagnostic(arrays)
    shortcut_rows = []
    for model, grouped in (("ridge", linear), ("mlp", mlp)):
        for metric in ("pearson", "spearman", "mae", "rmse"):
            grouped_value = grouped["mean_across_repeats"][metric]
            random_value = random_result["models"][model][metric]
            shortcut_rows.append({"model": model, "metric": metric, "group_source_heldout": grouped_value,
                                  "random_state_split": random_value,
                                  "random_minus_group": None if grouped_value is None or random_value is None else float(random_value - grouped_value),
                                  "random_split_is_secondary_only": True,
                                  "random_source_overlap": random_result["source_overlap"]})
    atomic_csv(outputs[9], shortcut_rows, list(shortcut_rows[0]))
    integrity = {
        "status": "PASS", "audit_only_no_production_gate_saved": True,
        "query_manifest_sha256": file_hash(QUERY), "branch_finalization_sha256": file_hash(FINALIZATION),
        "statewise_sha256": file_hash(STATEWISE), "paired_sha256": file_hash(PAIRED),
        "feature_schema_path": str(FEATURE_SCHEMA), "feature_schema_sha256": file_hash(FEATURE_SCHEMA),
        "state_count": len(arrays["delta"]), "root_source_count": len(set(arrays["root"])),
        "group_split_source_leakage": 0, "all_materialized_state_hashes_verified": True,
        "future_or_terminal_relative_features_used": False, "output_hashes": {},
        "random_split_assignments": random_assignments,
    }
    for path in outputs[:10]: integrity["output_hashes"][path.name] = file_hash(path)
    atomic_json(outputs[10], integrity)
    print(json.dumps({"status": "PASS", "states": len(arrays["delta"]),
                      "roots": len(set(arrays["root"])), "outputs": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
