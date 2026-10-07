"""Frozen fold utilities for the source-weighting gate diagnostic.

This module deliberately imports the original hard-stable CV implementation
for model initialization, full-batch optimization, normalization, state
aggregation, and validation-only threshold selection.  It is the single
common implementation point for all conditions in this diagnostic.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
CV_DIR = ROOT / "diagnostics" / "hard_stable_boundary_crossval"
CONF_DIR = ROOT / "diagnostics" / "gphi_gate_confidence_aware_v1"
DATA_DIR = ROOT / "diagnostics" / "gphi_training_dataset_v4"
AUDIT_DIR = ROOT / "diagnostics" / "gate_source_balancing_v1"
sys.path.insert(0, str(CV_DIR))
import run_crossval as cv  # noqa: E402


SEEDS = (17, 23, 41)
HIDDEN = [64, 64]
TOLERANCE = 2e-6


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = fieldnames or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, float) and not math.isfinite(item):
            return None
        raise TypeError(type(item).__name__)

    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=convert) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logit(probability: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(probability, float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def serialize_checkpoint(path: Path, params, mean, scale, binary, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "normalization_mean": np.asarray(mean),
        "normalization_scale": np.asarray(scale),
        "normalization_binary_mask": np.asarray(binary),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for index, layer in enumerate(params):
        values[f"layer_{index}_weight"] = np.asarray(layer["w"])
        values[f"layer_{index}_bias"] = np.asarray(layer["b"])
    np.savez_compressed(path, **values)


def load_context() -> dict:
    """Load frozen stable state labels, samples, schema, and seven manifests."""
    frozen = json.loads((AUDIT_DIR / "frozen_fold_manifest.json").read_text())
    confidence = read_csv(CONF_DIR / "oracle_confidence_dataset.csv")
    stable_rows = [
        row for row in confidence
        if row["oracle_confidence_class"] in ("ORACLE_STABLE_ZERO", "ORACLE_STABLE_NONZERO")
    ]
    state = {row["state_id"]: row for row in stable_rows}
    label = {state_id: int(row["original_gate_label"]) for state_id, row in state.items()}
    with np.load(DATA_DIR / "samples.npz", allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    schema = json.loads((DATA_DIR / "feature_schema.json").read_text())
    sample_ids = arrays["state_id"].astype(str)
    sample_label = np.asarray([label.get(state_id, -1) for state_id in sample_ids], int)
    if arrays["features"].shape[1] != 214:
        raise RuntimeError(("unexpected feature width", arrays["features"].shape))
    if set(sample_ids[sample_label >= 0]) - set(state):
        raise RuntimeError("nonstable state entered stable labels")
    return {
        "frozen": frozen,
        "state": state,
        "label": label,
        "arrays": arrays,
        "schema": schema,
        "sample_ids": sample_ids,
        "sample_label": sample_label,
        "stable_ids": set(state),
    }


def prepare_folds(context: dict) -> list[dict]:
    """Reconstruct exactly the frozen train/validation/test partitions."""
    arrays = context["arrays"]
    state = context["state"]
    sample_ids = context["sample_ids"]
    stable_ids = context["stable_ids"]
    result = []
    for raw in context["frozen"]["folds"]:
        train_ids = list(raw["normalization_fit_state_ids"])
        test_ids = list(raw["all_stable_test_state_ids"])
        validation_groups = list(raw["validation_source_groups"])
        validation_ids = sorted(
            state_id for state_id, row in state.items()
            if row["source_group"] in validation_groups
        )
        train_groups = set(raw["train_source_groups"])
        test_group = raw["heldout_source_group"]
        if (test_group in train_groups or test_group in validation_groups
                or train_groups & set(validation_groups)):
            raise RuntimeError(("source group leakage", raw["fold_id"]))
        if set(train_ids) & set(validation_ids) or set(train_ids) & set(test_ids) or set(validation_ids) & set(test_ids):
            raise RuntimeError(("state split overlap", raw["fold_id"]))
        if set(train_ids) | set(validation_ids) | set(test_ids) != stable_ids:
            raise RuntimeError(("stable state coverage mismatch", raw["fold_id"], len(train_ids), len(validation_ids), len(test_ids), len(stable_ids)))
        if any(state[state_id]["source_group"] == test_group for state_id in train_ids + validation_ids):
            raise RuntimeError(("test source entered train/validation", raw["fold_id"]))
        stable_mask = np.isin(sample_ids, list(stable_ids))
        train_index = np.flatnonzero(stable_mask & np.isin(sample_ids, train_ids))
        validation_index = np.flatnonzero(stable_mask & np.isin(sample_ids, validation_ids))
        test_index = np.flatnonzero(stable_mask & np.isin(sample_ids, test_ids))
        mean, scale, binary = cv.normalization(arrays["features"][train_index], context["schema"])
        normalized = ((arrays["features"] - mean) / scale).astype(np.float32)
        result.append({
            "fold_id": raw["fold_id"], "outer_group": test_group,
            "train_ids": train_ids, "validation_ids": validation_ids, "test_ids": test_ids,
            "train_groups": sorted(train_groups), "validation_groups": validation_groups,
            "train_index": train_index, "validation_index": validation_index, "test_index": test_index,
            "mean": mean, "scale": scale, "binary": binary, "normalized": normalized,
            "raw_manifest": raw,
        })
    return result


def train_full_batch_baseline(fold: dict, context: dict, checkpoint_dir: Path | None = None) -> dict:
    """Exact prior full-batch ordinary-BCE training condition, seed by seed."""
    arrays = context["arrays"]
    label = context["label"]
    sample_ids = context["sample_ids"]
    sample_label = context["sample_label"]
    seed_models, validation, test, training_rows = {}, [], [], []
    for seed in SEEDS:
        params, best_epoch, validation_bce = cv.train_model(
            HIDDEN, seed,
            fold["normalized"][fold["train_index"]], sample_label[fold["train_index"]], sample_ids[fold["train_index"]],
            fold["normalized"][fold["validation_index"]], sample_ids[fold["validation_index"]], label,
        )
        val_ids, val_p, val_y = cv.aggregate_state(
            cv.predict(params, fold["normalized"][fold["validation_index"]]),
            sample_ids[fold["validation_index"]], label,
        )
        test_ids, test_p, test_y = cv.aggregate_state(
            cv.predict(params, fold["normalized"][fold["test_index"]]),
            sample_ids[fold["test_index"]], label,
        )
        threshold = cv.select_threshold(val_y, val_p)
        if not (np.isfinite(val_p).all() and np.isfinite(test_p).all()):
            raise RuntimeError(("nonfinite output", fold["fold_id"], seed))
        if checkpoint_dir is not None:
            serialize_checkpoint(
                checkpoint_dir / fold["fold_id"] / f"seed_{seed}.npz", params, fold["mean"], fold["scale"], fold["binary"],
                {"condition": "BASELINE_SAMPLE_UNIFORM", "architecture": [214, 64, 64, 1], "activation": "SiLU", "loss": "ordinary BCE", "optimizer": "AdamW(lr=1e-3, weight_decay=1e-5)", "seed": seed, "fold_id": fold["fold_id"], "outer_group": fold["outer_group"], "best_epoch": best_epoch, "validation_selected_threshold": threshold, "normalization": "train-only mean/std; boolean dimensions literal"},
            )
        seed_models[seed] = {"params": params, "threshold": threshold, "best_epoch": best_epoch, "validation_bce": validation_bce}
        validation.append((val_ids, val_p, val_y))
        test.append((test_ids, test_p, test_y))
        training_rows.append({"condition": "BASELINE_SAMPLE_UNIFORM", "fold_id": fold["fold_id"], "heldout_source_group": fold["outer_group"], "training_seed": seed, "best_epoch": best_epoch, "validation_state_BCE": validation_bce, "validation_selected_threshold": threshold, "finite": True})
    val_ids = validation[0][0]
    test_ids = test[0][0]
    if any(not np.array_equal(val_ids, item[0]) for item in validation) or any(not np.array_equal(test_ids, item[0]) for item in test):
        raise RuntimeError(("state aggregation mismatch", fold["fold_id"]))
    val_mean = np.mean([item[1] for item in validation], axis=0)
    test_mean = np.mean([item[1] for item in test], axis=0)
    ensemble_threshold = cv.select_threshold(validation[0][2], val_mean)
    training_rows.append({"condition": "BASELINE_SAMPLE_UNIFORM", "fold_id": fold["fold_id"], "heldout_source_group": fold["outer_group"], "training_seed": "SEED_MEAN", "best_epoch": "", "validation_state_BCE": cv.state_bce(validation[0][2], val_mean), "validation_selected_threshold": ensemble_threshold, "finite": bool(np.isfinite(val_mean).all() and np.isfinite(test_mean).all())})
    return {"fold": fold, "seed_models": seed_models, "validation": validation, "test": test, "val_ids": val_ids, "test_ids": test_ids, "validation_mean": val_mean, "test_mean": test_mean, "ensemble_threshold": ensemble_threshold, "training_rows": training_rows}


def hierarchical_epoch_indices(fold: dict, context: dict, sampler: str, rng: np.random.Generator) -> np.ndarray:
    """Draw exactly N_train saved variants from a frozen hierarchy, with replacement.

    A full training epoch remains one optimizer update, exactly as the original
    full-batch recipe.  Only the probability law of the N_train rows changes.
    This avoids silently confounding source weighting with a different number
    of optimizer updates or minibatch size.
    """
    if sampler not in ("STATE_BALANCED_ONLY", "SOURCE_GROUP_BALANCED"):
        raise ValueError(sampler)
    arrays = context["arrays"]
    sample_ids = context["sample_ids"]
    train_index = fold["train_index"]
    by_state: dict[str, np.ndarray] = {}
    for state_id in fold["train_ids"]:
        indices = train_index[sample_ids[train_index] == state_id]
        if len(indices) != 64:
            raise RuntimeError(("training state does not have exactly 64 variants", fold["fold_id"], state_id, len(indices)))
        by_state[state_id] = indices
    count = len(train_index)
    state_ids = np.asarray(sorted(by_state), dtype=object)
    if sampler == "STATE_BALANCED_ONLY":
        sampled_state = rng.integers(0, len(state_ids), size=count)
        sampled_variant = rng.integers(0, 64, size=count)
        return np.asarray([by_state[str(state_ids[s])][v] for s, v in zip(sampled_state, sampled_variant)], dtype=int)
    grouped: dict[str, list[str]] = defaultdict(list)
    for state_id in state_ids:
        grouped[context["state"][str(state_id)]["source_group"]].append(str(state_id))
    groups = np.asarray(sorted(grouped), dtype=object)
    sampled_groups = rng.integers(0, len(groups), size=count)
    sampled_variant = rng.integers(0, 64, size=count)
    indices = np.empty(count, dtype=int)
    for row, (group_index, variant_index) in enumerate(zip(sampled_groups, sampled_variant)):
        group_states = grouped[str(groups[group_index])]
        state_id = group_states[int(rng.integers(0, len(group_states)))]
        indices[row] = by_state[state_id][variant_index]
    return indices


def train_hierarchical_resampled_fullbatch(
    fold: dict,
    context: dict,
    sampler: str,
    condition: str,
    checkpoint_dir: Path | None = None,
) -> dict:
    """Train frozen MLP/BCE with one hierarchy-resampled full batch per epoch.

    This intentionally preserves initialization, AdamW settings, validation
    aggregation, early stopping cadence/patience, and threshold rule from the
    original full-batch CV.  The specified hierarchy is the sole changed
    training-distribution variable.
    """
    if sampler not in ("STATE_BALANCED_ONLY", "SOURCE_GROUP_BALANCED"):
        raise ValueError(sampler)
    label = context["label"]
    sample_ids = context["sample_ids"]
    sample_label = context["sample_label"]
    seed_models, validation, test, training_rows = {}, [], [], []
    for seed in SEEDS:
        params = cv.init_params(jax.random.PRNGKey(seed), [214, *HIDDEN, 1])
        optimizer = cv.optax.adamw(learning_rate=1e-3, weight_decay=1e-5)
        optimizer_state = optimizer.init(params)
        sample_rng = np.random.default_rng(seed)

        @jax.jit
        def step(current, optimizer_state, batch_x, batch_y):
            def loss_fn(candidate):
                return jnp.mean(cv.optax.sigmoid_binary_cross_entropy(cv.logits(candidate, batch_x), batch_y))
            loss, gradients = jax.value_and_grad(loss_fn)(current)
            updates, new_state = optimizer.update(gradients, optimizer_state, current)
            return cv.optax.apply_updates(current, updates), new_state, loss

        best = params
        best_validation_bce = float("inf")
        best_epoch = 0
        stale = 0
        for epoch in range(1, 1201):
            sampled = hierarchical_epoch_indices(fold, context, sampler, sample_rng)
            batch_x = jnp.asarray(fold["normalized"][sampled], jnp.float32)
            batch_y = jnp.asarray(sample_label[sampled], jnp.float32)
            params, optimizer_state, _ = step(params, optimizer_state, batch_x, batch_y)
            if epoch == 1 or epoch % 10 == 0:
                _, validation_probability, validation_y = cv.aggregate_state(
                    cv.predict(params, fold["normalized"][fold["validation_index"]]),
                    sample_ids[fold["validation_index"]], label,
                )
                validation_bce = cv.state_bce(validation_y, validation_probability)
                if validation_bce < best_validation_bce - 1e-6:
                    best_validation_bce = validation_bce
                    best_epoch = epoch
                    best = jax.tree_util.tree_map(lambda value: np.asarray(value).copy(), params)
                    stale = 0
                else:
                    stale += 1
                if stale >= 35:
                    break
        val_ids, val_p, val_y = cv.aggregate_state(cv.predict(best, fold["normalized"][fold["validation_index"]]), sample_ids[fold["validation_index"]], label)
        test_ids, test_p, test_y = cv.aggregate_state(cv.predict(best, fold["normalized"][fold["test_index"]]), sample_ids[fold["test_index"]], label)
        threshold = cv.select_threshold(val_y, val_p)
        if not (np.isfinite(val_p).all() and np.isfinite(test_p).all()):
            raise RuntimeError(("nonfinite output", condition, fold["fold_id"], seed))
        if checkpoint_dir is not None:
            serialize_checkpoint(checkpoint_dir / fold["fold_id"] / f"seed_{seed}.npz", best, fold["mean"], fold["scale"], fold["binary"], {"condition": condition, "sampler": sampler, "architecture": [214, 64, 64, 1], "activation": "SiLU", "loss": "ordinary BCE", "optimizer": "AdamW(lr=1e-3, weight_decay=1e-5)", "seed": seed, "fold_id": fold["fold_id"], "outer_group": fold["outer_group"], "best_epoch": best_epoch, "validation_selected_threshold": threshold, "normalization": "train-only mean/std; boolean dimensions literal", "epoch_semantics": "one N_train-draw full-batch update per epoch"})
        seed_models[seed] = {"params": best, "threshold": threshold, "best_epoch": best_epoch, "validation_bce": best_validation_bce}
        validation.append((val_ids, val_p, val_y))
        test.append((test_ids, test_p, test_y))
        training_rows.append({"condition": condition, "sampler": sampler, "fold_id": fold["fold_id"], "heldout_source_group": fold["outer_group"], "training_seed": seed, "best_epoch": best_epoch, "validation_state_BCE": best_validation_bce, "validation_selected_threshold": threshold, "finite": True})
    val_ids = validation[0][0]
    test_ids = test[0][0]
    if any(not np.array_equal(val_ids, item[0]) for item in validation) or any(not np.array_equal(test_ids, item[0]) for item in test):
        raise RuntimeError(("state aggregation mismatch", condition, fold["fold_id"]))
    val_mean = np.mean([item[1] for item in validation], axis=0)
    test_mean = np.mean([item[1] for item in test], axis=0)
    ensemble_threshold = cv.select_threshold(validation[0][2], val_mean)
    training_rows.append({"condition": condition, "sampler": sampler, "fold_id": fold["fold_id"], "heldout_source_group": fold["outer_group"], "training_seed": "SEED_MEAN", "best_epoch": "", "validation_state_BCE": cv.state_bce(validation[0][2], val_mean), "validation_selected_threshold": ensemble_threshold, "finite": bool(np.isfinite(val_mean).all() and np.isfinite(test_mean).all())})
    return {"fold": fold, "seed_models": seed_models, "validation": validation, "test": test, "val_ids": val_ids, "test_ids": test_ids, "validation_mean": val_mean, "test_mean": test_mean, "ensemble_threshold": ensemble_threshold, "training_rows": training_rows}


def aggregate_prediction_rows(trained: dict, context: dict, condition: str) -> list[dict]:
    label = context["label"]
    rows = []
    for seed_index, seed in enumerate(SEEDS):
        ids, probabilities, labels = trained["test"][seed_index]
        threshold = trained["seed_models"][seed]["threshold"]
        for state_id, p, y in zip(ids, probabilities, labels):
            rows.append({"condition": condition, "fold_id": trained["fold"]["fold_id"], "heldout_source_group": trained["fold"]["outer_group"], "state_id": str(state_id), "oracle_label": int(y), "p_gate": float(p), "gate_logit": float(logit(p)), "validation_selected_threshold": threshold, "validation_threshold_logit": float(logit(threshold)), "fold_relative_logit_margin": float(logit(p) - logit(threshold)), "predicted_label": int(p >= threshold), "correct": bool(int(p >= threshold) == int(y)), "training_seed": seed})
    for state_id, p, y in zip(trained["test_ids"], trained["test_mean"], trained["test"][0][2]):
        threshold = trained["ensemble_threshold"]
        rows.append({"condition": condition, "fold_id": trained["fold"]["fold_id"], "heldout_source_group": trained["fold"]["outer_group"], "state_id": str(state_id), "oracle_label": int(y), "p_gate": float(p), "gate_logit": float(logit(p)), "validation_selected_threshold": threshold, "validation_threshold_logit": float(logit(threshold)), "fold_relative_logit_margin": float(logit(p) - logit(threshold)), "predicted_label": int(p >= threshold), "correct": bool(int(p >= threshold) == int(y)), "training_seed": "SEED_MEAN"})
    return rows


def infer_variants(state_id: str, trained: dict, context: dict, condition: str, role: str = "") -> list[dict]:
    arrays = context["arrays"]
    sample_ids = context["sample_ids"]
    label = context["label"]
    indices = np.flatnonzero(sample_ids == state_id)
    indices = indices[np.argsort(arrays["flow_seed"][indices])]
    if len(indices) != 64:
        raise RuntimeError(("expected 64 saved variants", state_id, len(indices)))
    x = trained["fold"]["normalized"][indices]
    seed_probabilities = []
    rows = []
    for seed in SEEDS:
        model = trained["seed_models"][seed]
        logits = np.asarray(cv.logits(model["params"], jnp.asarray(x, jnp.float32)), float)
        probability = np.asarray(jax.nn.sigmoid(jnp.asarray(logits)), float)
        seed_probabilities.append(probability)
        for index, data_index in enumerate(indices):
            threshold = model["threshold"]
            predicted = int(probability[index] >= threshold)
            rows.append({"condition": condition, "role": role, "fold_id": trained["fold"]["fold_id"], "heldout_source_group": trained["fold"]["outer_group"], "state_id": state_id, "oracle_label": label[state_id], "flow_seed": int(arrays["flow_seed"][data_index]), "sample_id": str(arrays["sample_id"][data_index]), "training_seed": seed, "p_gate": float(probability[index]), "gate_logit": float(logits[index]), "validation_selected_threshold": threshold, "validation_threshold_logit": float(logit(threshold)), "fold_relative_logit_margin": float(logits[index] - logit(threshold)), "predicted_label": predicted, "correct": bool(predicted == label[state_id])})
    mean_probability = np.mean(seed_probabilities, axis=0)
    threshold = trained["ensemble_threshold"]
    for index, data_index in enumerate(indices):
        raw_logit = float(logit(mean_probability[index]))
        predicted = int(mean_probability[index] >= threshold)
        rows.append({"condition": condition, "role": role, "fold_id": trained["fold"]["fold_id"], "heldout_source_group": trained["fold"]["outer_group"], "state_id": state_id, "oracle_label": label[state_id], "flow_seed": int(arrays["flow_seed"][data_index]), "sample_id": str(arrays["sample_id"][data_index]), "training_seed": "SEED_MEAN", "p_gate": float(mean_probability[index]), "gate_logit": raw_logit, "validation_selected_threshold": threshold, "validation_threshold_logit": float(logit(threshold)), "fold_relative_logit_margin": float(raw_logit - logit(threshold)), "predicted_label": predicted, "correct": bool(predicted == label[state_id])})
    return rows


def check_original_six(trained_by_fold: dict[str, dict], output_rows: list[dict]) -> list[dict]:
    """Strictly compare state means and fold thresholds to the original six-fold file."""
    saved = read_csv(CV_DIR / "out_of_fold_predictions.csv")
    original = {
        (row["fold_id"], row["training_seed"], row["state_id"]): row
        for row in saved if row["model"] == "MLP_64x64"
    }
    checks = []
    for fold_id, trained in trained_by_fold.items():
        if fold_id == "LOGO_anchor_D1_pair231":
            continue
        for seed in (*SEEDS, "SEED_MEAN"):
            current = [row for row in output_rows if row["fold_id"] == fold_id and str(row["training_seed"]) == str(seed)]
            p_diff = max(abs(float(row["p_gate"]) - float(original[(fold_id, str(seed), row["state_id"])]["p_gate"])) for row in current)
            t_diff = max(abs(float(row["validation_selected_threshold"]) - float(original[(fold_id, str(seed), row["state_id"])]["validation_selected_threshold"])) for row in current)
            checks.append({"fold_id": fold_id, "training_seed": seed, "comparison": "original_six_state_oof", "heldout_state_count": len(current), "max_abs_probability_difference": p_diff, "max_abs_threshold_difference": t_diff, "tolerance": TOLERANCE, "passed": bool(p_diff <= TOLERANCE and t_diff <= TOLERANCE)})
    return checks


def check_missing_d1_variants(trained: dict, context: dict) -> list[dict]:
    """Compare D1 p40 exactly to the prior validated missing-fold run."""
    state_id = "R_D1_s95106004_p40"
    current = infer_variants(state_id, trained, context, "BASELINE_SAMPLE_UNIFORM", "missing_fold_reproduction")
    saved = read_csv(ROOT / "diagnostics" / "oof_gate_flow_variant_error_audit" / "missing_state_variant_predictions.csv")
    lookup = {(row["training_seed"], row["flow_seed"]): row for row in saved}
    checks = []
    for seed in (*SEEDS, "SEED_MEAN"):
        rows = [row for row in current if str(row["training_seed"]) == str(seed)]
        p_diff = max(abs(float(row["p_gate"]) - float(lookup[(str(seed), str(row["flow_seed"]))]["p_gate"])) for row in rows)
        threshold_diff = max(abs(float(row["validation_selected_threshold"]) - float(lookup[(str(seed), str(row["flow_seed"]))]["validation_selected_threshold"])) for row in rows)
        checks.append({"fold_id": trained["fold"]["fold_id"], "training_seed": seed, "comparison": "prior_validated_D1_variant_oof", "heldout_state_count": 1, "variant_count": len(rows), "max_abs_probability_difference": p_diff, "max_abs_threshold_difference": threshold_diff, "tolerance": TOLERANCE, "passed": bool(p_diff <= TOLERANCE and threshold_diff <= TOLERANCE)})
    return checks


def target_variant_states(context: dict) -> dict[str, str]:
    """Fixed pair members plus pre-existing robust false-positive zero states."""
    flow_pairs = read_csv(ROOT / "diagnostics" / "flow_variant_representation_collision_audit" / "audited_pairs.csv")
    target = {}
    for row in flow_pairs:
        target[row["zero_state_id"]] = f"{row['pair_id']}:zero"
        target[row["nonzero_state_id"]] = f"{row['pair_id']}:nonzero"
    robust = read_csv(ROOT / "diagnostics" / "hard_stable_boundary_crossval" / "repeated_errors.csv")
    for row in robust:
        if int(row["oracle_label"]) == 0:
            target.setdefault(row["state_id"], "robust_false_positive_zero")
    if not target:
        raise RuntimeError("no fixed variant targets")
    if not set(target) <= context["stable_ids"]:
        raise RuntimeError(("target is not stable", set(target) - context["stable_ids"]))
    return target


def calibration_rows(state_oof: list[dict], condition: str) -> list[dict]:
    result = []
    for fold_id in sorted({row["fold_id"] for row in state_oof}):
        rows = [row for row in state_oof if row["fold_id"] == fold_id and row["training_seed"] == "SEED_MEAN"]
        for label_value, name in ((0, "stable_zero"), (1, "stable_nonzero")):
            values = [row for row in rows if int(row["oracle_label"]) == label_value]
            margins = np.asarray([float(row["fold_relative_logit_margin"]) for row in values])
            result.append({"condition": condition, "fold_id": fold_id, "heldout_source_group": values[0]["heldout_source_group"] if values else "", "oracle_class": name, "state_count": len(values), "mean_fold_relative_logit_margin": float(margins.mean()) if len(margins) else None, "median_fold_relative_logit_margin": float(np.median(margins)) if len(margins) else None, "std_fold_relative_logit_margin": float(margins.std()) if len(margins) else None, "FPR_or_FNR": float(np.mean([int(row["predicted_label"]) == 1 for row in values])) if label_value == 0 and values else (float(np.mean([int(row["predicted_label"]) == 0 for row in values])) if values else None)})
    return result
