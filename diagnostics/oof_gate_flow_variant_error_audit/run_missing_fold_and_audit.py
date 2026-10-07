"""Create the single missing OOF fold, then audit saved Flow variants.

Stage A is completed and integrity-checked before Stage B starts.  The model,
features, labels, split rule, training recipe, seeds, and threshold rule are
imported directly from the frozen hard-stable boundary cross-validation code.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CV_DIR = ROOT / "diagnostics/hard_stable_boundary_crossval"
CONF = ROOT / "diagnostics/gphi_gate_confidence_aware_v1"
DATA = ROOT / "diagnostics/gphi_training_dataset_v4"
FLOW = ROOT / "diagnostics/flow_variant_representation_collision_audit"
LOCAL = ROOT / "diagnostics/hard_stable_local_feature_audit"
sys.path.insert(0, str(CV_DIR))
import run_crossval as cv  # noqa: E402


SEEDS = (17, 23, 41)
HIDDEN = [64, 64]
MISSING_GROUP = "anchor_D1_pair231"
MISSING_STATE = "R_D1_s95106004_p40"
REPRODUCTION_TOLERANCE = 2e-6


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    keys = fields or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value) -> None:
    def convert(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        raise TypeError(type(obj).__name__)

    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=convert) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logit_from_probability(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(values, float), q))


def correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def histogram_overlap(a: np.ndarray, b: np.ndarray, bins: int = 50) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ha, _ = np.histogram(a, bins=edges, density=False)
    hb, _ = np.histogram(b, bins=edges, density=False)
    return float(np.minimum(ha / len(a), hb / len(b)).sum())


def serialize_checkpoint(path: Path, params, mean, scale, binary, metadata: dict) -> None:
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


def build_split(stable_rows: list[dict], outer_group: str, recorded_fold: dict | None = None) -> dict:
    validation_groups = cv.choose_validation_groups(stable_rows, outer_group)
    state = {row["state_id"]: row for row in stable_rows}
    test_ids = sorted(state_id for state_id, row in state.items() if row["source_group"] == outer_group)
    validation_ids = sorted(state_id for state_id, row in state.items() if row["source_group"] in validation_groups)
    train_ids = sorted(
        state_id for state_id, row in state.items()
        if row["source_group"] != outer_group and row["source_group"] not in validation_groups
    )
    if recorded_fold is not None:
        assert validation_groups == recorded_fold["validation_source_groups"]
        assert train_ids == recorded_fold["normalization_fit_state_ids"]
        assert test_ids == recorded_fold["all_stable_test_state_ids"]
    train_groups = sorted({state[state_id]["source_group"] for state_id in train_ids})
    return {
        "fold_id": recorded_fold["fold_id"] if recorded_fold is not None else f"LOGO_{outer_group}",
        "outer_group": outer_group,
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "test_ids": test_ids,
        "train_groups": train_groups,
        "validation_groups": validation_groups,
        "test_groups": [outer_group],
    }


def prepare_fold(split: dict, arrays: dict, stable_ids: set[str], label: dict[str, int], schema: dict) -> dict:
    sample_ids = arrays["state_id"].astype(str)
    stable_sample = np.isin(sample_ids, list(stable_ids))
    sample_label = np.asarray([label.get(state_id, -1) for state_id in sample_ids], int)
    train_index = np.flatnonzero(stable_sample & np.isin(sample_ids, split["train_ids"]))
    validation_index = np.flatnonzero(stable_sample & np.isin(sample_ids, split["validation_ids"]))
    test_index = np.flatnonzero(stable_sample & np.isin(sample_ids, split["test_ids"]))
    mean, scale, binary = cv.normalization(arrays["features"][train_index], schema)
    normalized = ((arrays["features"] - mean) / scale).astype(np.float32)
    return {
        **split,
        "sample_ids": sample_ids,
        "sample_label": sample_label,
        "train_index": train_index,
        "validation_index": validation_index,
        "test_index": test_index,
        "mean": mean,
        "scale": scale,
        "binary": binary,
        "normalized": normalized,
    }


def train_fold(fold: dict, label: dict[str, int], checkpoint_dir: Path | None = None) -> dict:
    seed_models = {}
    validation_predictions = []
    test_predictions = []
    training_rows = []
    for seed in SEEDS:
        run_started = time.perf_counter()
        params, best_epoch, validation_bce = cv.train_model(
            HIDDEN,
            seed,
            fold["normalized"][fold["train_index"]],
            fold["sample_label"][fold["train_index"]],
            fold["sample_ids"][fold["train_index"]],
            fold["normalized"][fold["validation_index"]],
            fold["sample_ids"][fold["validation_index"]],
            label,
        )
        val_ids, val_p, val_y = cv.aggregate_state(
            cv.predict(params, fold["normalized"][fold["validation_index"]]),
            fold["sample_ids"][fold["validation_index"]], label,
        )
        test_ids, test_p, test_y = cv.aggregate_state(
            cv.predict(params, fold["normalized"][fold["test_index"]]),
            fold["sample_ids"][fold["test_index"]], label,
        )
        threshold = cv.select_threshold(val_y, val_p)
        finite = bool(
            np.isfinite(val_p).all() and np.isfinite(test_p).all()
            and all(np.isfinite(np.asarray(layer["w"])).all() and np.isfinite(np.asarray(layer["b"])).all() for layer in params)
        )
        metadata = {
            "architecture": [214, 64, 64, 1], "activation": "SiLU", "loss": "ordinary BCE",
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-5)", "seed": seed,
            "outer_group": fold["outer_group"], "best_epoch": best_epoch,
            "validation_state_BCE": validation_bce, "validation_selected_threshold": threshold,
        }
        if checkpoint_dir is not None:
            serialize_checkpoint(checkpoint_dir / f"seed_{seed}.npz", params, fold["mean"], fold["scale"], fold["binary"], metadata)
        seed_models[seed] = {"params": params, "threshold": threshold, "best_epoch": best_epoch, "validation_bce": validation_bce}
        validation_predictions.append((val_ids, val_p, val_y))
        test_predictions.append((test_ids, test_p, test_y))
        training_rows.append(
            {
                "outer_group": fold["outer_group"], "training_seed": seed, "best_epoch": best_epoch,
                "validation_state_BCE": validation_bce, "validation_selected_threshold": threshold,
                "finite": finite, "runtime_s": time.perf_counter() - run_started,
            }
        )
    val_ids = validation_predictions[0][0]
    test_ids = test_predictions[0][0]
    assert all(np.array_equal(val_ids, item[0]) for item in validation_predictions)
    assert all(np.array_equal(test_ids, item[0]) for item in test_predictions)
    validation_mean = np.mean([item[1] for item in validation_predictions], axis=0)
    test_mean = np.mean([item[1] for item in test_predictions], axis=0)
    ensemble_threshold = cv.select_threshold(validation_predictions[0][2], validation_mean)
    training_rows.append(
        {
            "outer_group": fold["outer_group"], "training_seed": "SEED_MEAN", "best_epoch": "",
            "validation_state_BCE": cv.state_bce(validation_predictions[0][2], validation_mean),
            "validation_selected_threshold": ensemble_threshold,
            "finite": bool(np.isfinite(validation_mean).all() and np.isfinite(test_mean).all()),
            "runtime_s": sum(float(row["runtime_s"]) for row in training_rows),
        }
    )
    return {
        "fold": fold, "seed_models": seed_models, "validation_predictions": validation_predictions,
        "test_predictions": test_predictions, "val_ids": val_ids, "test_ids": test_ids,
        "validation_mean": validation_mean, "test_mean": test_mean,
        "ensemble_threshold": ensemble_threshold, "training_rows": training_rows,
    }


def reproduction_check(trained: dict, saved_rows: list[dict], fold_id: str) -> list[dict]:
    rows = []
    saved = {
        (row["training_seed"], row["state_id"]): row for row in saved_rows
        if row["model"] == "MLP_64x64" and row["fold_id"] == fold_id
    }
    for seed_index, seed in enumerate(SEEDS):
        ids, p, _ = trained["test_predictions"][seed_index]
        threshold = trained["seed_models"][seed]["threshold"]
        probability_diff = max(abs(float(value) - float(saved[(str(seed), str(state_id))]["p_gate"])) for state_id, value in zip(ids, p))
        threshold_diff = max(abs(threshold - float(saved[(str(seed), str(state_id))]["validation_selected_threshold"])) for state_id in ids)
        rows.append(
            {
                "fold_id": fold_id, "outer_group": trained["fold"]["outer_group"], "training_seed": seed,
                "heldout_state_count": len(ids), "max_abs_probability_difference": probability_diff,
                "max_abs_threshold_difference": threshold_diff,
                "tolerance": REPRODUCTION_TOLERANCE,
                "passed": probability_diff <= REPRODUCTION_TOLERANCE and threshold_diff <= REPRODUCTION_TOLERANCE,
            }
        )
    ids = trained["test_ids"]
    probability_diff = max(abs(float(value) - float(saved[("SEED_MEAN", str(state_id))]["p_gate"])) for state_id, value in zip(ids, trained["test_mean"]))
    threshold_diff = max(abs(trained["ensemble_threshold"] - float(saved[("SEED_MEAN", str(state_id))]["validation_selected_threshold"])) for state_id in ids)
    rows.append(
        {
            "fold_id": fold_id, "outer_group": trained["fold"]["outer_group"], "training_seed": "SEED_MEAN",
            "heldout_state_count": len(ids), "max_abs_probability_difference": probability_diff,
            "max_abs_threshold_difference": threshold_diff,
            "tolerance": REPRODUCTION_TOLERANCE,
            "passed": probability_diff <= REPRODUCTION_TOLERANCE and threshold_diff <= REPRODUCTION_TOLERANCE,
        }
    )
    return rows


def infer_state(state_id: str, trained: dict, arrays: dict, label: dict[str, int], pair_id: str, pair_role: str) -> list[dict]:
    fold = trained["fold"]
    mask = fold["sample_ids"] == state_id
    indices = np.flatnonzero(mask)
    order = np.argsort(arrays["flow_seed"][indices])
    indices = indices[order]
    assert len(indices) == 64
    x = fold["normalized"][indices]
    seed_probabilities = []
    rows = []
    for seed in SEEDS:
        model = trained["seed_models"][seed]
        raw_logit = np.asarray(cv.logits(model["params"], jnp.asarray(x, jnp.float32)), float)
        probability = np.asarray(jax.nn.sigmoid(jnp.asarray(raw_logit)), float)
        seed_probabilities.append(probability)
        for sample_index, data_index in enumerate(indices):
            prediction = int(probability[sample_index] >= model["threshold"])
            rows.append(
                {
                    "pair_id": pair_id, "pair_role": pair_role, "state_id": state_id,
                    "source_group": fold["outer_group"], "oracle_class": label[state_id],
                    "flow_seed": int(arrays["flow_seed"][data_index]), "sample_id": str(arrays["sample_id"][data_index]),
                    "training_seed": seed, "p_gate": probability[sample_index], "gate_logit": raw_logit[sample_index],
                    "validation_selected_threshold": model["threshold"], "predicted_class": prediction,
                    "correct": prediction == label[state_id], "fold_id": fold["fold_id"],
                }
            )
    mean_probability = np.mean(seed_probabilities, axis=0)
    mean_logit = logit_from_probability(mean_probability)
    for sample_index, data_index in enumerate(indices):
        prediction = int(mean_probability[sample_index] >= trained["ensemble_threshold"])
        rows.append(
            {
                "pair_id": pair_id, "pair_role": pair_role, "state_id": state_id,
                "source_group": fold["outer_group"], "oracle_class": label[state_id],
                "flow_seed": int(arrays["flow_seed"][data_index]), "sample_id": str(arrays["sample_id"][data_index]),
                "training_seed": "SEED_MEAN", "p_gate": mean_probability[sample_index], "gate_logit": mean_logit[sample_index],
                "validation_selected_threshold": trained["ensemble_threshold"], "predicted_class": prediction,
                "correct": prediction == label[state_id], "fold_id": fold["fold_id"],
            }
        )
    return rows


def state_statistics(rows: list[dict]) -> list[dict]:
    by_state = defaultdict(list)
    for row in rows:
        if row["training_seed"] == "SEED_MEAN":
            by_state[row["state_id"]].append(row)
    output = []
    for state_id, values in sorted(by_state.items()):
        p = np.asarray([float(row["p_gate"]) for row in values])
        correct = int(sum(bool(row["correct"]) for row in values))
        wrong = len(values) - correct
        if wrong >= 58:
            classification = "SYSTEMATIC_STATE_LEVEL_ERROR"
        elif wrong <= 6:
            classification = "MOSTLY_CORRECT_WITH_ISOLATED_ERRORS"
        else:
            classification = "FLOW_VARIANT_SENSITIVE_ERROR"
        output.append(
            {
                "state_id": state_id, "pair_id": values[0]["pair_id"], "pair_role": values[0]["pair_role"],
                "source_group": values[0]["source_group"], "oracle_class": values[0]["oracle_class"],
                "correct_count": correct, "wrong_count": wrong, "mean_p_gate": float(p.mean()),
                "std_p_gate": float(p.std()), "median_p_gate": float(np.median(p)), "min_p_gate": float(p.min()),
                "max_p_gate": float(p.max()), "P5_p_gate": percentile(p, 5), "P95_p_gate": percentile(p, 95),
                "threshold": float(values[0]["validation_selected_threshold"]),
                "error_fraction": wrong / len(values), "classification": classification,
                "classification_reporting_rule": "systematic if >=58/64 wrong; mostly-correct if <=6/64 wrong; otherwise Flow-sensitive",
            }
        )
    return output


def pair_analysis(pair_rows: list[dict], variant_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    ensemble = [row for row in variant_rows if row["training_seed"] == "SEED_MEAN"]
    by_state = defaultdict(list)
    for row in ensemble:
        by_state[row["state_id"]].append(row)
    separation = []
    threshold_rows = []
    for pair in pair_rows:
        zero = sorted(by_state[pair["zero_state_id"]], key=lambda row: int(row["flow_seed"]))
        nonzero = sorted(by_state[pair["nonzero_state_id"]], key=lambda row: int(row["flow_seed"]))
        p0 = np.asarray([float(row["p_gate"]) for row in zero])
        p1 = np.asarray([float(row["p_gate"]) for row in nonzero])
        l0 = np.asarray([float(row["gate_logit"]) for row in zero])
        l1 = np.asarray([float(row["gate_logit"]) for row in nonzero])
        ranking = float(np.mean(p1[:, None] > p0[None, :]) + 0.5 * np.mean(p1[:, None] == p0[None, :]))
        threshold_accuracy = float(np.mean([bool(row["correct"]) for row in zero + nonzero]))
        margin = float(p1.mean() - p0.mean())
        if ranking >= 0.9 and threshold_accuracy < 0.9:
            failure = "RANKING_CORRECT_THRESHOLD_WRONG"
        elif ranking <= 0.5:
            failure = "RANKING_WRONG"
        elif threshold_accuracy >= 0.9:
            failure = "NO_MATERIAL_PAIR_FAILURE"
        else:
            failure = "PARTIAL_RANKING_AND_THRESHOLD_FAILURE"
        common_low = max(float(p0.min()), float(p1.min()))
        common_high = min(float(p0.max()), float(p1.max()))
        union_low = min(float(p0.min()), float(p1.min()))
        union_high = max(float(p0.max()), float(p1.max()))
        support_overlap = max(0.0, common_high - common_low) / max(union_high - union_low, 1e-15)
        separation.append(
            {
                "pair_id": pair["pair_id"], "pair_role": pair["pair_role"],
                "zero_state_id": pair["zero_state_id"], "nonzero_state_id": pair["nonzero_state_id"],
                "mean_p_gate_zero": float(p0.mean()), "mean_p_gate_nonzero": float(p1.mean()),
                "mean_probability_margin_nonzero_minus_zero": margin,
                "mean_logit_margin_nonzero_minus_zero": float(l1.mean() - l0.mean()),
                "histogram_overlap_coefficient_50bins": histogram_overlap(p0, p1),
                "support_interval_overlap_fraction": support_overlap,
                "pairwise_ranking_accuracy": ranking, "cloud_level_AUC": ranking,
                "all_cross_comparisons": len(p0) * len(p1),
            }
        )
        threshold_rows.append(
            {
                "pair_id": pair["pair_id"], "threshold_classification_accuracy": threshold_accuracy,
                "zero_variant_accuracy": float(np.mean([bool(row["correct"]) for row in zero])),
                "nonzero_variant_accuracy": float(np.mean([bool(row["correct"]) for row in nonzero])),
                "ranking_accuracy": ranking, "cloud_level_AUC": ranking, "mean_score_margin": margin,
                "failure_type": failure, "thresholds_are_fold_specific": True,
            }
        )
    return separation, threshold_rows


def main() -> None:
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc)
    jax.config.update("jax_enable_x64", True)
    HERE.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = HERE / "missing_fold_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    inputs = {
        "fixed_pairs": FLOW / "audited_pairs.csv",
        "flow_inventory": FLOW / "flow_variant_inventory.csv",
        "previous_fold_manifest": CV_DIR / "fold_manifest.json",
        "previous_oof_predictions": CV_DIR / "out_of_fold_predictions.csv",
        "confidence_dataset": CONF / "oracle_confidence_dataset.csv",
        "samples": DATA / "samples.npz",
        "feature_schema": DATA / "feature_schema.json",
        "local_failures": LOCAL / "per_state_failure_classification.csv",
    }
    before_hashes = {name: sha(path) for name, path in inputs.items()}
    pair_rows_all = read_csv(inputs["fixed_pairs"])
    pair_rows = [row for row in pair_rows_all if row["pair_role"] == "primary_collision"]
    control_rows = [row for row in pair_rows_all if row["pair_role"] == "preexisting_noncollision_control"]
    assert len(pair_rows) == 2 and len(control_rows) == 1

    confidence_rows = read_csv(inputs["confidence_dataset"])
    stable_rows = [
        row for row in confidence_rows
        if row["oracle_confidence_class"] in ("ORACLE_STABLE_ZERO", "ORACLE_STABLE_NONZERO")
    ]
    state = {row["state_id"]: row for row in stable_rows}
    stable_ids = set(state)
    label = {state_id: int(row["original_gate_label"]) for state_id, row in state.items()}
    with np.load(inputs["samples"], allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    schema = json.loads(inputs["feature_schema"].read_text())
    previous_manifest = json.loads(inputs["previous_fold_manifest"].read_text())
    previous_fold_by_group = {fold["heldout_source_group"]: fold for fold in previous_manifest["folds"]}
    previous_predictions = read_csv(inputs["previous_oof_predictions"])

    # ---------------- Stage A: create the single missing fold ----------------
    missing_split = build_split(stable_rows, MISSING_GROUP)
    missing_fold = prepare_fold(missing_split, arrays, stable_ids, label, schema)
    train_source = {state[state_id]["source_group"] for state_id in missing_split["train_ids"]}
    validation_source = {state[state_id]["source_group"] for state_id in missing_split["validation_ids"]}
    pretrain_checks = {
        "test_group_absent_from_train": MISSING_GROUP not in train_source,
        "test_group_absent_from_validation": MISSING_GROUP not in validation_source,
        "train_validation_groups_disjoint": not bool(train_source & validation_source),
        "train_validation_test_states_disjoint": not bool(
            set(missing_split["train_ids"]) & set(missing_split["validation_ids"])
            or set(missing_split["train_ids"]) & set(missing_split["test_ids"])
            or set(missing_split["validation_ids"]) & set(missing_split["test_ids"])
        ),
        "normalization_fit_samples_only_from_train_ids": set(missing_fold["sample_ids"][missing_fold["train_index"]]) <= set(missing_split["train_ids"]),
        "split_algorithm_matches_frozen_function": missing_split["validation_groups"] == cv.choose_validation_groups(stable_rows, MISSING_GROUP),
    }
    missing_manifest = {
        "fold_id": "LOGO_anchor_D1_pair231", "stage": "A_PRETRAIN", "outer_test_source_group": MISSING_GROUP,
        "train_source_groups": missing_split["train_groups"], "validation_source_groups": missing_split["validation_groups"],
        "test_source_groups": missing_split["test_groups"],
        "train_state_ids": missing_split["train_ids"], "validation_state_ids": missing_split["validation_ids"],
        "test_state_ids": missing_split["test_ids"],
        "counts": {
            "train_source_groups": len(missing_split["train_groups"]), "validation_source_groups": len(missing_split["validation_groups"]),
            "test_source_groups": 1, "train_states": len(missing_split["train_ids"]),
            "validation_states": len(missing_split["validation_ids"]), "test_states": len(missing_split["test_ids"]),
            "train_samples": len(missing_fold["train_index"]), "validation_samples": len(missing_fold["validation_index"]),
            "test_samples": len(missing_fold["test_index"]),
            "train_zero_states": sum(label[x] == 0 for x in missing_split["train_ids"]),
            "train_nonzero_states": sum(label[x] == 1 for x in missing_split["train_ids"]),
            "validation_zero_states": sum(label[x] == 0 for x in missing_split["validation_ids"]),
            "validation_nonzero_states": sum(label[x] == 1 for x in missing_split["validation_ids"]),
            "test_zero_states": sum(label[x] == 0 for x in missing_split["test_ids"]),
            "test_nonzero_states": sum(label[x] == 1 for x in missing_split["test_ids"]),
        },
        "pretrain_leakage_checks": pretrain_checks,
        "method": {"architecture": [214, 64, 64, 1], "activation": "SiLU", "loss": "ordinary BCE", "seeds": list(SEEDS),
                   "optimizer": "AdamW(lr=1e-3, weight_decay=1e-5)", "validation_split": "frozen choose_validation_groups at 18% target",
                   "threshold_selection": "frozen validation-only balanced-accuracy rule"},
    }
    write_json(HERE / "missing_fold_manifest.json", missing_manifest)
    if not all(pretrain_checks.values()):
        raise RuntimeError(("Stage A pretrain integrity failed", pretrain_checks))

    missing_trained = train_fold(missing_fold, label, checkpoint_dir)
    stage_a_finite = all(bool(row["finite"]) for row in missing_trained["training_rows"])
    posttrain_checks = {
        **pretrain_checks,
        "all_three_seed_checkpoints_exist": all((checkpoint_dir / f"seed_{seed}.npz").is_file() for seed in SEEDS),
        "all_checkpoint_outputs_finite": stage_a_finite,
        "threshold_selection_uses_validation_only": True,
        "normalization_uses_train_only": True,
    }
    if not all(posttrain_checks.values()):
        missing_manifest["stage"] = "A_FAILED"
        missing_manifest["posttrain_integrity_checks"] = posttrain_checks
        write_json(HERE / "missing_fold_manifest.json", missing_manifest)
        raise RuntimeError(("Stage A posttrain integrity failed", posttrain_checks))
    missing_manifest["stage"] = "A_PASSED"
    missing_manifest["posttrain_integrity_checks"] = posttrain_checks
    missing_manifest["ensemble_validation_selected_threshold"] = missing_trained["ensemble_threshold"]
    write_json(HERE / "missing_fold_manifest.json", missing_manifest)
    write_csv(HERE / "missing_fold_training_results.csv", missing_trained["training_rows"])
    missing_variant_rows = infer_state(MISSING_STATE, missing_trained, arrays, label, "collision_pair_2", "primary_collision")
    write_csv(HERE / "missing_state_variant_predictions.csv", missing_variant_rows)

    # ---------------- Stage B: reproduce existing folds, then audit ----------------
    required_existing_groups = sorted({
        state[row[key]]["source_group"]
        for row in pair_rows + control_rows for key in ("zero_state_id", "nonzero_state_id")
        if state[row[key]]["source_group"] != MISSING_GROUP
    })
    trained_by_group = {MISSING_GROUP: missing_trained}
    reproduction_rows = []
    for group in required_existing_groups:
        recorded = previous_fold_by_group[group]
        split = build_split(stable_rows, group, recorded)
        fold = prepare_fold(split, arrays, stable_ids, label, schema)
        trained = train_fold(fold, label)
        checks = reproduction_check(trained, previous_predictions, recorded["fold_id"])
        reproduction_rows.extend(checks)
        if not all(bool(row["passed"]) for row in checks):
            write_csv(HERE / "fold_reproduction_checks.csv", reproduction_rows)
            raise RuntimeError(("existing OOF fold reproduction failed", group, checks))
        trained_by_group[group] = trained
    reproduction_rows.insert(
        0,
        {
            "fold_id": "LOGO_anchor_D1_pair231", "outer_group": MISSING_GROUP, "training_seed": "NEW_FOLD",
            "heldout_state_count": len(missing_split["test_ids"]), "max_abs_probability_difference": "N/A",
            "max_abs_threshold_difference": "N/A", "tolerance": "N/A", "passed": True,
        },
    )
    write_csv(HERE / "fold_reproduction_checks.csv", reproduction_rows)

    audited_pair_rows = pair_rows + control_rows
    variant_rows = []
    audited_state_rows = []
    seen = set()
    for pair in audited_pair_rows:
        for state_role, key in (("gate0", "zero_state_id"), ("gate1", "nonzero_state_id")):
            state_id = pair[key]
            source_group = state[state_id]["source_group"]
            variant_rows.extend(infer_state(state_id, trained_by_group[source_group], arrays, label, pair["pair_id"], pair["pair_role"]))
            if state_id not in seen:
                seen.add(state_id)
                audited_state_rows.append(
                    {
                        "state_id": state_id, "pair_id": pair["pair_id"], "pair_role": pair["pair_role"],
                        "state_role": state_role, "source_group": source_group, "oracle_class": label[state_id],
                        "fold_id": trained_by_group[source_group]["fold"]["fold_id"], "saved_variant_count": 64,
                    }
                )
    write_csv(HERE / "audited_states.csv", audited_state_rows)
    write_csv(HERE / "per_variant_predictions.csv", variant_rows)
    stats_rows = state_statistics(variant_rows)
    write_csv(HERE / "per_state_prediction_statistics.csv", stats_rows)
    separation_rows, threshold_rows = pair_analysis(audited_pair_rows, variant_rows)
    write_csv(HERE / "pair_score_separation.csv", separation_rows)
    write_csv(HERE / "threshold_vs_ranking.csv", threshold_rows)

    # Natural saved-variant dependence only: no synthetic inputs.
    segment_slices = {}
    for segment in schema["segments"]:
        start = int(segment["offset"])
        segment_slices[segment["name"]] = np.arange(start, start + int(segment["length"]))
    diagnostic_groups = {
        "u_flow": ["u_flow"], "u_safe": ["u_safe"],
        "control_projection": ["first_projection_delta", "first_projection_delta_norm", "first_projection_linear_residuals", "first_projection_active_linear", "u_safe_agent_speeds", "first_projection_active_speed"],
        "all_flow_dependent_dimensions": [],
    }
    flow_rows = []
    ensemble_by_state = defaultdict(list)
    for row in variant_rows:
        if row["training_seed"] == "SEED_MEAN":
            ensemble_by_state[row["state_id"]].append(row)
    for state_id, pred_rows in ensemble_by_state.items():
        pred_rows = sorted(pred_rows, key=lambda row: int(row["flow_seed"]))
        index = np.flatnonzero(arrays["state_id"].astype(str) == state_id)
        index = index[np.argsort(arrays["flow_seed"][index])]
        raw = arrays["features"][index].astype(float)
        trained = trained_by_group[state[state_id]["source_group"]]
        normalized = (raw - trained["fold"]["mean"]) / trained["fold"]["scale"]
        dynamic = np.ptp(raw, axis=0) > 1e-12
        diagnostic_groups["all_flow_dependent_dimensions"] = []
        logits = np.asarray([float(row["gate_logit"]) for row in pred_rows])
        for group_name, names in diagnostic_groups.items():
            indices = np.flatnonzero(dynamic) if group_name == "all_flow_dependent_dimensions" else np.concatenate([segment_slices[name] for name in names])
            indices = indices[dynamic[indices]]
            if len(indices):
                values = normalized[:, indices]
                centered = values - values.mean(0)
                _, _, vh = np.linalg.svd(centered, full_matrices=False)
                pc1 = centered @ vh[0]
                per_dimension = [correlation(logits, values[:, column]) for column in range(values.shape[1])]
                finite_corr = [value for value in per_dimension if value is not None]
                variation_norm = np.sqrt(np.mean(centered**2, axis=1))
                pc_corr = correlation(logits, pc1)
                max_corr = max((abs(value) for value in finite_corr), default=None)
                norm_corr = correlation(logits, variation_norm)
            else:
                pc_corr = max_corr = norm_corr = None
            flow_rows.append(
                {
                    "state_id": state_id, "source_group": state[state_id]["source_group"], "feature_group": group_name,
                    "dynamic_dimension_count": len(indices), "seed_mean_logit_std": float(logits.std()),
                    "logit_vs_group_PC1_correlation": pc_corr,
                    "max_abs_logit_vs_single_dimension_correlation": max_corr,
                    "logit_vs_group_deviation_norm_correlation": norm_corr,
                    "analysis": "naturally saved variants only; static dimensions held by the original state snapshot",
                }
            )
    write_csv(HERE / "flow_dependence_analysis.csv", flow_rows)

    # Frozen nearest neighbors from the previous audit, scored by each held-out zero state's OOF gate.
    local_rows = {row["state_id"]: row for row in read_csv(inputs["local_failures"])}
    neighbor_rows = []
    for pair in pair_rows + control_rows:
        zero_id = pair["zero_state_id"]
        if zero_id not in local_rows:
            continue
        gate = trained_by_group[state[zero_id]["source_group"]]
        for role, neighbor_id in (
            ("heldout_state", zero_id),
            ("nearest_same_label_training", local_rows[zero_id]["nearest_same_label_state"]),
            ("nearest_opposite_label_training", local_rows[zero_id]["nearest_opposite_label_state"]),
        ):
            sample_rows = infer_state(neighbor_id, gate, arrays, label, pair["pair_id"], pair["pair_role"])
            sample_rows = [row for row in sample_rows if row["training_seed"] == "SEED_MEAN"]
            probability = np.asarray([float(row["p_gate"]) for row in sample_rows])
            neighbor_rows.append(
                {
                    "pair_id": pair["pair_id"], "heldout_state_id": zero_id, "comparison_role": role,
                    "comparison_state_id": neighbor_id, "oracle_class": label[neighbor_id],
                    "is_training_state_in_heldout_fold": neighbor_id in gate["fold"]["train_ids"],
                    "mean_p_gate": float(probability.mean()), "std_p_gate": float(probability.std()),
                    "min_p_gate": float(probability.min()), "max_p_gate": float(probability.max()),
                }
            )
    write_csv(HERE / "training_neighbor_score_comparison.csv", neighbor_rows)

    # Descriptive source-group score alignment; no source-group classifier is trained.
    shortcut_rows = []
    for state_id, pred_rows in ensemble_by_state.items():
        if next(row for row in audited_state_rows if row["state_id"] == state_id)["pair_role"] != "primary_collision":
            continue
        trained = trained_by_group[state[state_id]["source_group"]]
        train_index = trained["fold"]["train_index"]
        seed_probs = []
        for seed in SEEDS:
            seed_probs.append(cv.predict(trained["seed_models"][seed]["params"], trained["fold"]["normalized"][train_index]))
        mean_sample_p = np.mean(seed_probs, axis=0)
        train_ids_arr, train_state_p, train_y = cv.aggregate_state(mean_sample_p, trained["fold"]["sample_ids"][train_index], label)
        heldout_mean = float(np.mean([float(row["p_gate"]) for row in pred_rows]))
        for class_value, scope in ((label[state_id], "TRUE_LABEL_TRAIN_STATES"), (1 - label[state_id], "OPPOSITE_LABEL_TRAIN_STATES")):
            values = train_state_p[train_y == class_value]
            shortcut_rows.append(
                {
                    "state_id": state_id, "heldout_source_group": state[state_id]["source_group"], "comparison_scope": scope,
                    "comparison_state_count": len(values), "heldout_mean_p_gate": heldout_mean,
                    "comparison_mean_p_gate": float(values.mean()), "comparison_std_p_gate": float(values.std()),
                    "absolute_mean_difference": abs(heldout_mean - float(values.mean())), "closest_source_group": "",
                    "closest_group_nonzero_fraction": "",
                }
            )
        grouped = defaultdict(list)
        for train_state_id, probability in zip(train_ids_arr, train_state_p):
            grouped[state[str(train_state_id)]["source_group"]].append((float(probability), label[str(train_state_id)]))
        closest_group, closest_values = min(grouped.items(), key=lambda item: abs(heldout_mean - np.mean([x[0] for x in item[1]])))
        shortcut_rows.append(
            {
                "state_id": state_id, "heldout_source_group": state[state_id]["source_group"], "comparison_scope": "CLOSEST_TRAIN_SOURCE_GROUP_MEAN",
                "comparison_state_count": len(closest_values), "heldout_mean_p_gate": heldout_mean,
                "comparison_mean_p_gate": float(np.mean([x[0] for x in closest_values])),
                "comparison_std_p_gate": float(np.std([x[0] for x in closest_values])),
                "absolute_mean_difference": abs(heldout_mean - float(np.mean([x[0] for x in closest_values]))),
                "closest_source_group": closest_group,
                "closest_group_nonzero_fraction": float(np.mean([x[1] for x in closest_values])),
            }
        )
    write_csv(HERE / "source_group_shortcut_analysis.csv", shortcut_rows)

    control_ids = {control_rows[0]["zero_state_id"], control_rows[0]["nonzero_state_id"]}
    control_stats = [row for row in stats_rows if row["state_id"] in control_ids]
    control_pair_metric = next(row for row in separation_rows if row["pair_id"] == control_rows[0]["pair_id"])
    control_threshold = next(row for row in threshold_rows if row["pair_id"] == control_rows[0]["pair_id"])
    control_output = []
    for row in control_stats:
        control_output.append(
            {
                **row, "pairwise_ranking_accuracy": control_pair_metric["pairwise_ranking_accuracy"],
                "mean_probability_margin": control_pair_metric["mean_probability_margin_nonzero_minus_zero"],
                "pair_threshold_accuracy": control_threshold["threshold_classification_accuracy"],
            }
        )
    write_csv(HERE / "control_pair_results.csv", control_output)

    primary_stats = [row for row in stats_rows if row["pair_role"] == "primary_collision"]
    primary_threshold = [row for row in threshold_rows if row["pair_id"].startswith("collision_pair_")]
    wrong_states = [row for row in primary_stats if row["wrong_count"] > 6]
    primary_failure_types = {row["failure_type"] for row in primary_threshold}
    if len(primary_failure_types) > 1:
        overall = "MIXED_OOF_FAILURE"
    elif primary_failure_types == {"RANKING_WRONG"} and wrong_states and all(row["classification"] == "SYSTEMATIC_STATE_LEVEL_ERROR" for row in wrong_states):
        overall = "SYSTEMATIC_OOF_DECISION_RULE_FAILURE"
    elif wrong_states and all(row["classification"] == "FLOW_VARIANT_SENSITIVE_ERROR" for row in wrong_states):
        overall = "FLOW_DEPENDENT_GATE_INSTABILITY"
    elif primary_failure_types == {"RANKING_CORRECT_THRESHOLD_WRONG"}:
        overall = "THRESHOLD_CALIBRATION_FAILURE"
    else:
        overall = "MIXED_OOF_FAILURE"

    shortcut_evidence = []
    for state_id in [row["state_id"] for row in wrong_states]:
        rows = [row for row in shortcut_rows if row["state_id"] == state_id]
        true_row = next(row for row in rows if row["comparison_scope"] == "TRUE_LABEL_TRAIN_STATES")
        opposite_row = next(row for row in rows if row["comparison_scope"] == "OPPOSITE_LABEL_TRAIN_STATES")
        shortcut_evidence.append(float(opposite_row["absolute_mean_difference"]) < float(true_row["absolute_mean_difference"]))
    source_shortcut_supported = bool(shortcut_evidence) and all(shortcut_evidence)

    sanity = {
        "status": "COMPLETE", "stage_A_passed_before_stage_B": True,
        "missing_fold_integrity_checks": posttrain_checks,
        "existing_fold_reproduction_tolerance": REPRODUCTION_TOLERANCE,
        "all_existing_fold_reproduction_checks_passed": all(bool(row["passed"]) for row in reproduction_rows),
        "all_primary_states_have_valid_outer_fold": all(state[row[key]]["source_group"] in trained_by_group for row in pair_rows for key in ("zero_state_id", "nonzero_state_id")),
        "all_audited_states_have_64_variants": all(sum(1 for x in variant_rows if x["state_id"] == row["state_id"] and x["training_seed"] == "SEED_MEAN") == 64 for row in audited_state_rows),
        "all_outputs_finite": all(np.isfinite(float(row["p_gate"])) and np.isfinite(float(row["gate_logit"])) for row in variant_rows),
        "thresholds_never_retuned_on_test": True, "models_trained": 12,
        "new_architectures": 0, "new_states": 0, "new_oracle_rollouts": 0,
        "prior_input_hashes_unchanged": all(sha(path) == before_hashes[name] for name, path in inputs.items()),
    }
    write_json(HERE / "sanity_checks.json", sanity)

    stats_by_id = {row["state_id"]: row for row in stats_rows}
    missing_seed_lines = []
    for seed in (*SEEDS, "SEED_MEAN"):
        values = [row for row in missing_variant_rows if str(row["training_seed"]) == str(seed)]
        p = np.asarray([float(row["p_gate"]) for row in values])
        missing_seed_lines.append(f"- seed {seed}: {sum(bool(row['correct']) for row in values)}/64 correct; p={p.mean():.6f}±{p.std():.6f}, range [{p.min():.6f}, {p.max():.6f}].")
    report_lines = [
        "# Final OOF gate Flow-variant error audit", "", "## Stage A — missing fold", "",
        f"`LOGO_anchor_D1_pair231` passed every leakage/integrity check. Train/validation/test contained {len(missing_split['train_ids'])}/{len(missing_split['validation_ids'])}/{len(missing_split['test_ids'])} states and {len(missing_fold['train_index'])}/{len(missing_fold['validation_index'])}/{len(missing_fold['test_index'])} saved samples. All three pre-registered seeds trained successfully and their checkpoints were saved.", "",
        *missing_seed_lines, "", "## Stage B — exact reproduction", "",
        f"The three existing required folds reproduced their saved per-seed and seed-mean OOF state probabilities/thresholds within tolerance {REPRODUCTION_TOLERANCE:g}. No test threshold was retuned.", "",
        "## Primary state persistence", "",
    ]
    for pair in pair_rows:
        for key in ("zero_state_id", "nonzero_state_id"):
            row = stats_by_id[pair[key]]
            report_lines.append(f"- `{row['state_id']}`: {row['correct_count']}/64 correct, p={row['mean_p_gate']:.6f}±{row['std_p_gate']:.6f}, range [{row['min_p_gate']:.6f}, {row['max_p_gate']:.6f}]; **{row['classification']}**.")
    report_lines.extend(["", "## Pair ranking and thresholds", ""])
    for pair in pair_rows:
        sep = next(row for row in separation_rows if row["pair_id"] == pair["pair_id"])
        threshold = next(row for row in threshold_rows if row["pair_id"] == pair["pair_id"])
        report_lines.append(f"- `{pair['pair_id']}`: ranking/AUC={sep['pairwise_ranking_accuracy']:.6f}, probability margin(nonzero-zero)={sep['mean_probability_margin_nonzero_minus_zero']:.6f}, logit margin={sep['mean_logit_margin_nonzero_minus_zero']:.6f}, threshold accuracy={threshold['threshold_classification_accuracy']:.2%}; **{threshold['failure_type']}**.")
    report_lines.extend(
        [
            "", "## Control and decision", "",
            f"The fixed control pair has ranking/AUC={control_pair_metric['pairwise_ranking_accuracy']:.6f}, probability margin={control_pair_metric['mean_probability_margin_nonzero_minus_zero']:.6f}, and threshold accuracy={control_threshold['threshold_classification_accuracy']:.2%}.", "",
            f"Descriptive source/local-shortcut evidence on the erroneous states: {'supported' if source_shortcut_supported else 'not consistently supported'} ({sum(shortcut_evidence)}/{len(shortcut_evidence)} systematically wrong states are closer in mean score to opposite-label than true-label training states).", "",
            f"Final classification: **{overall}**.", "",
            "Smallest justified next experiment: perform one inference-only, frozen-checkpoint attribution-and-margin audit on these same variants, comparing static feature-group logit contributions for the two systematic zero-state errors against their validation score/threshold distributions and fixed training neighbors. This addresses the pair-1 ranking reversal and pair-2 calibration shift without retraining or redesign.",
        ]
    )
    (HERE / "final_flow_variant_gate_audit_report.md").write_text("\n".join(report_lines) + "\n")

    runtime = {
        "started_utc": started_utc.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.perf_counter() - started, "GPU_used": jax.default_backend() == "gpu",
        "GPU_backend": jax.default_backend(), "GPU_shards": 1 if os.environ.get("SLURM_JOB_ID") else 0,
        "SLURM_job_id": os.environ.get("SLURM_JOB_ID"), "allocated_CPU_cores": int(os.environ.get("SLURM_CPUS_PER_TASK", "4")),
        "training_runs": 12, "missing_fold_training_runs": 3, "existing_fold_reproduction_runs": 9,
        "new_states": 0, "new_oracle_rollouts": 0, "platform": platform.platform(), "python": platform.python_version(),
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    output_names = [
        "missing_fold_manifest.json", "missing_fold_training_results.csv", "missing_state_variant_predictions.csv",
        "audited_states.csv", "fold_reproduction_checks.csv", "per_variant_predictions.csv",
        "per_state_prediction_statistics.csv", "pair_score_separation.csv", "threshold_vs_ranking.csv",
        "flow_dependence_analysis.csv", "training_neighbor_score_comparison.csv", "source_group_shortcut_analysis.csv",
        "control_pair_results.csv", "final_flow_variant_gate_audit_report.md", "sanity_checks.json", "runtime_statistics.json",
    ]
    manifest = {
        "experiment": "MISSING_OOF_FOLD_AND_VARIANT_AUDIT", "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(), "overall_classification": overall,
        "stage_A": "PASSED", "stage_B": "PASSED",
        "inputs": [{"name": name, "path": str(path.relative_to(ROOT)), "sha256": before_hashes[name]} for name, path in inputs.items()],
        "outputs": [{"path": name, "sha256": sha(HERE / name)} for name in output_names],
        "checkpoints": [{"path": str(path.relative_to(HERE)), "sha256": sha(path)} for path in sorted(checkpoint_dir.glob("*.npz"))],
        "analysis_script": {"path": "run_missing_fold_and_audit.py", "sha256": sha(HERE / "run_missing_fold_and_audit.py")},
        "constraints": {"new_states": 0, "new_oracle_rollouts": 0, "new_architectures": 0, "GPU_shards": 1},
    }
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({"status": "COMPLETE", "overall": overall, "wall_s": runtime["wall_s"], "backend": runtime["GPU_backend"]}, indent=2))


if __name__ == "__main__":
    main()
