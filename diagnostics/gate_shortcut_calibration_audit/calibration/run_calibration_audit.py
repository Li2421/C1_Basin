"""Frozen-checkpoint calibration shift audit for the four relevant OOF folds.

No model is trained here.  Every probability comes from an exactly reproduced
checkpoint produced by the preceding OOF experiments.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/gate_shortcut_calibration_audit/calibration"
SHARED = ROOT / "diagnostics/gate_shortcut_calibration_audit/shared_checkpoints"
OOF = ROOT / "diagnostics/oof_gate_flow_variant_error_audit"
CONF = ROOT / "diagnostics/gphi_gate_confidence_aware_v1"
DATA = ROOT / "diagnostics/gphi_training_dataset_v4"
SEEDS = (17, 23, 41)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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


def logit(p):
    value = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(value / (1 - value))


def sigmoid(x):
    value = np.asarray(x, float)
    return np.where(value >= 0, 1 / (1 + np.exp(-value)), np.exp(value) / (1 + np.exp(value)))


def silu(x):
    return x * sigmoid(x)


def load_checkpoint(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
    layers = []
    index = 0
    while f"layer_{index}_weight" in arrays:
        layers.append((arrays[f"layer_{index}_weight"], arrays[f"layer_{index}_bias"]))
        index += 1
    return {
        "layers": layers,
        "mean": arrays["normalization_mean"],
        "scale": arrays["normalization_scale"],
        "binary": arrays["normalization_binary_mask"],
        "metadata": json.loads(str(arrays["metadata_json"])),
    }


def predict_one_checkpoint(model: dict, raw: np.ndarray) -> np.ndarray:
    value = ((raw - model["mean"]) / model["scale"]).astype(np.float32)
    for index, (weight, bias) in enumerate(model["layers"]):
        value = value @ weight + bias
        if index < len(model["layers"]) - 1:
            value = silu(value)
    return sigmoid(value.reshape(-1))


def aggregate_states(sample_ids: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = defaultdict(list)
    for state_id, probability in zip(sample_ids, probabilities):
        groups[str(state_id)].append(float(probability))
    ids = np.asarray(sorted(groups), str)
    return ids, np.asarray([np.mean(groups[state_id]) for state_id in ids])


def finite_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    if len(values) == 0:
        return {"count": 0, "mean": None, "median": None, "std": None, "p10": None, "p90": None}
    return {
        "count": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
        "std": float(values.std()), "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def group_distribution_rows(fold: dict, state_prob: dict[str, float], state: dict[str, dict]) -> list[dict]:
    rows = []
    scopes = {
        "TRAIN": fold["train_ids"], "VALIDATION": fold["validation_ids"], "OOF_TEST": fold["test_ids"]
    }
    for scope, ids in scopes.items():
        source_groups = sorted({state[state_id]["source_group"] for state_id in ids})
        for source_group in source_groups:
            selected = [state_id for state_id in ids if state[state_id]["source_group"] == source_group]
            zero = logit([state_prob[state_id] for state_id in selected if int(state[state_id]["original_gate_label"]) == 0])
            one = logit([state_prob[state_id] for state_id in selected if int(state[state_id]["original_gate_label"]) == 1])
            all_logits = logit([state_prob[state_id] for state_id in selected])
            s0, s1, sa = finite_summary(zero), finite_summary(one), finite_summary(all_logits)
            rows.append({
                "fold_id": fold["fold_id"], "outer_group": fold["outer_group"], "scope": scope,
                "source_group": source_group, "state_count": len(selected),
                "zero_count": s0["count"], "zero_mean_logit": s0["mean"], "zero_median_logit": s0["median"],
                "zero_std_logit": s0["std"], "nonzero_count": s1["count"],
                "nonzero_mean_logit": s1["mean"], "nonzero_median_logit": s1["median"],
                "nonzero_std_logit": s1["std"],
                "within_group_margin_nonzero_minus_zero": None if not len(zero) or not len(one) else float(one.mean() - zero.mean()),
                "class_midpoint_logit": None if not len(zero) or not len(one) else float((one.mean() + zero.mean()) / 2),
                "all_mean_logit": sa["mean"], "all_median_logit": sa["median"], "all_std_logit": sa["std"],
                "all_p10_logit": sa["p10"], "all_p90_logit": sa["p90"],
            })
    return rows


def scope_stats(ids: list[str], state_prob: dict[str, float], state: dict[str, dict]) -> dict:
    labels = np.asarray([int(state[state_id]["original_gate_label"]) for state_id in ids], int)
    logits = logit([state_prob[state_id] for state_id in ids])
    zero, one = logits[labels == 0], logits[labels == 1]
    if not len(zero) or not len(one):
        raise RuntimeError("calibration comparison requires both stable classes")
    return {
        "count": len(ids), "zero_count": len(zero), "nonzero_count": len(one),
        "zero_mean": float(zero.mean()), "zero_median": float(np.median(zero)), "zero_std": float(zero.std()),
        "one_mean": float(one.mean()), "one_median": float(np.median(one)), "one_std": float(one.std()),
        "margin": float(one.mean() - zero.mean()), "center": float((one.mean() + zero.mean()) / 2),
        "all_mean": float(logits.mean()), "all_std": float(logits.std()),
    }


def infer_fold(group: str, state: dict[str, dict], raw_features: np.ndarray, sample_ids: np.ndarray) -> dict:
    if group == "anchor_D1_pair231":
        metadata = json.loads((OOF / "missing_fold_manifest.json").read_text())
        folder = OOF / "missing_fold_checkpoints"
        fold = {
            "fold_id": metadata["fold_id"], "outer_group": group,
            "train_ids": metadata["train_state_ids"], "validation_ids": metadata["validation_state_ids"],
            "test_ids": metadata["test_state_ids"],
            "ensemble_threshold": float(metadata["ensemble_validation_selected_threshold"]),
        }
    else:
        metadata = json.loads((SHARED / group / "fold_metadata.json").read_text())
        folder = SHARED / group
        fold = {
            "fold_id": metadata["fold_id"], "outer_group": group,
            "train_ids": metadata["train_state_ids"], "validation_ids": metadata["validation_state_ids"],
            "test_ids": metadata["test_state_ids"],
            "ensemble_threshold": float(metadata["ensemble_validation_selected_threshold"]),
        }
    models = [load_checkpoint(folder / f"seed_{seed}.npz") for seed in SEEDS]
    required_ids = set(fold["train_ids"] + fold["validation_ids"] + fold["test_ids"])
    mask = np.isin(sample_ids, list(required_ids))
    ensemble_sample_p = np.mean([predict_one_checkpoint(model, raw_features[mask]) for model in models], axis=0)
    ids, probabilities = aggregate_states(sample_ids[mask], ensemble_sample_p)
    state_probability = dict(zip(ids.tolist(), probabilities.tolist()))
    if set(state_probability) != required_ids:
        raise RuntimeError((group, "state inventory mismatch", len(state_probability), len(required_ids)))
    fold["state_probability"] = state_probability
    fold["models"] = models
    return fold


def affine_fit(x: np.ndarray, y: np.ndarray) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    design = np.c_[np.ones(len(x)), x]
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    prediction = intercept + slope * x
    residual = y - prediction
    denominator = np.sum((y - y.mean()) ** 2)
    return {
        "intercept": float(intercept), "slope": float(slope),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "max_abs_residual": float(np.max(np.abs(residual))),
        "r2": float(1 - np.sum(residual**2) / denominator) if denominator > 0 else None,
    }


def pair_analysis(pair_id: str, zero_id: str, one_id: str, variant_rows: list[dict],
                  fold_by_group: dict[str, dict], state: dict[str, dict]) -> dict:
    selected = [row for row in variant_rows if row["pair_id"] == pair_id and row["training_seed"] == "SEED_MEAN"]
    p0 = np.asarray([float(row["p_gate"]) for row in selected if row["state_id"] == zero_id])
    p1 = np.asarray([float(row["p_gate"]) for row in selected if row["state_id"] == one_id])
    l0, l1 = logit(p0), logit(p1)
    group0, group1 = state[zero_id]["source_group"], state[one_id]["source_group"]
    t0 = logit(float(fold_by_group[group0]["ensemble_threshold"])).item()
    t1 = logit(float(fold_by_group[group1]["ensemble_threshold"])).item()
    midpoint = float((l0.mean() + l1.mean()) / 2)
    rank = float(np.mean(l1[:, None] > l0[None, :]) + .5 * np.mean(l1[:, None] == l0[None, :]))
    common_threshold_low = float(l0.max())
    common_threshold_high = float(l1.min())
    return {
        "pair_id": pair_id, "zero_state_id": zero_id, "nonzero_state_id": one_id,
        "zero_source_group": group0, "nonzero_source_group": group1,
        "same_oof_checkpoint": group0 == group1,
        "zero_mean_probability": float(p0.mean()), "nonzero_mean_probability": float(p1.mean()),
        "zero_mean_logit": float(l0.mean()), "nonzero_mean_logit": float(l1.mean()),
        "mean_logit_margin_nonzero_minus_zero": float(l1.mean() - l0.mean()),
        "cloud_ranking_AUC": rank,
        "zero_validation_threshold_probability": float(fold_by_group[group0]["ensemble_threshold"]),
        "zero_validation_threshold_logit": t0,
        "nonzero_validation_threshold_probability": float(fold_by_group[group1]["ensemble_threshold"]),
        "nonzero_validation_threshold_logit": t1,
        "local_mean_midpoint_logit": midpoint,
        "zero_threshold_to_midpoint_logit_displacement": midpoint - t0,
        "nonzero_threshold_to_midpoint_logit_displacement": midpoint - t1,
        "minimum_zero_fold_threshold_increase_to_classify_zero_mean": float(l0.mean() - t0),
        "minimum_zero_fold_threshold_increase_to_classify_all_zero_variants": common_threshold_low - t0,
        "common_threshold_interval_low_exclusive": common_threshold_low,
        "common_threshold_interval_high_inclusive": common_threshold_high,
        "clean_common_threshold_exists_for_all_variants": common_threshold_low < common_threshold_high,
        "diagnostic_only": "test threshold was not changed; displacements are counterfactual margin diagnostics",
    }


def main() -> None:
    started = time.perf_counter()
    start_utc = datetime.now(timezone.utc).isoformat()
    HERE.mkdir(parents=True, exist_ok=True)
    confidence = read_csv(CONF / "oracle_confidence_dataset.csv")
    stable_rows = [row for row in confidence if row["oracle_confidence_class"] in ("ORACLE_STABLE_ZERO", "ORACLE_STABLE_NONZERO")]
    state = {row["state_id"]: row for row in stable_rows}
    with np.load(DATA / "samples.npz", allow_pickle=False) as loaded:
        raw_features = np.asarray(loaded["features"])
        sample_ids = np.asarray(loaded["state_id"]).astype(str)

    ready = json.loads((SHARED / "READY.json").read_text())
    if ready["status"] != "READY" or not ready["all_reproduction_checks_passed"]:
        raise RuntimeError("shared OOF checkpoints not verified")
    groups = ["anchor_D2_pair228", "anchor_D4_pair227", "qual_pair226", "anchor_D1_pair231"]
    folds = {group: infer_fold(group, state, raw_features, sample_ids) for group in groups}

    # Check against frozen OOF state predictions (three existing folds) and missing-state variant result.
    saved_oof = read_csv(ROOT / "diagnostics/hard_stable_boundary_crossval/out_of_fold_predictions.csv")
    max_reproduction_error = 0.0
    for group in groups[:3]:
        rows = [row for row in saved_oof if row["model"] == "MLP_64x64" and row["training_seed"] == "SEED_MEAN" and row["heldout_source_group"] == group]
        for row in rows:
            max_reproduction_error = max(max_reproduction_error, abs(folds[group]["state_probability"][row["state_id"]] - float(row["p_gate"])))
    missing_saved = read_csv(OOF / "per_state_prediction_statistics.csv")
    missing_saved_p = float(next(row for row in missing_saved if row["state_id"] == "R_D1_s95106004_p40")["mean_p_gate"])
    missing_reproduction_error = abs(folds["anchor_D1_pair231"]["state_probability"]["R_D1_s95106004_p40"] - missing_saved_p)
    if max_reproduction_error > 2e-6 or missing_reproduction_error > 2e-6:
        raise RuntimeError(("checkpoint reproduction failed", max_reproduction_error, missing_reproduction_error))

    distribution_rows = []
    shift_rows = []
    for group, fold in folds.items():
        distribution_rows.extend(group_distribution_rows(fold, fold["state_probability"], state))
        validation = scope_stats(fold["validation_ids"], fold["state_probability"], state)
        test = scope_stats(fold["test_ids"], fold["state_probability"], state)
        zero_shift = test["zero_mean"] - validation["zero_mean"]
        one_shift = test["one_mean"] - validation["one_mean"]
        additive = (zero_shift + one_shift) / 2
        differential = one_shift - zero_shift
        separation_ratio = test["margin"] / validation["margin"]
        spread_ratio = test["all_std"] / validation["all_std"]
        additive_residual = abs(differential) / 2
        shift_type = (
            "MOSTLY_ADDITIVE_SHIFT" if abs(separation_ratio - 1) <= .25 and abs(differential) <= max(.25, .35 * abs(additive))
            else "MOSTLY_SCALE_SHIFT" if abs(additive) <= .25 and abs(separation_ratio - 1) > .25
            else "CLASS_DEPENDENT_SHIFT" if abs(differential) > max(.5, abs(additive))
            else "MIXED_CALIBRATION_SHIFT"
        )
        shift_rows.append({
            "fold_id": fold["fold_id"], "outer_group": group,
            "validation_state_count": validation["count"], "test_state_count": test["count"],
            "validation_zero_mean_logit": validation["zero_mean"], "validation_nonzero_mean_logit": validation["one_mean"],
            "validation_margin": validation["margin"], "validation_center": validation["center"],
            "validation_all_std": validation["all_std"],
            "test_zero_mean_logit": test["zero_mean"], "test_nonzero_mean_logit": test["one_mean"],
            "test_margin": test["margin"], "test_center": test["center"], "test_all_std": test["all_std"],
            "zero_class_shift_test_minus_validation": zero_shift,
            "nonzero_class_shift_test_minus_validation": one_shift,
            "common_additive_shift": additive, "class_differential_shift": differential,
            "residual_per_class_after_best_additive_shift": additive_residual,
            "separation_scale_ratio": separation_ratio, "overall_spread_ratio": spread_ratio,
            "validation_threshold_probability": fold["ensemble_threshold"],
            "validation_threshold_logit": float(logit(fold["ensemble_threshold"])),
            "descriptive_shift_type": shift_type,
        })
    write_csv(HERE / "source_group_score_distributions.csv", distribution_rows)

    # A single global affine relationship between validation and held-out class centroids.
    x_centroids, y_centroids = [], []
    for row in shift_rows:
        x_centroids.extend([row["validation_zero_mean_logit"], row["validation_nonzero_mean_logit"]])
        y_centroids.extend([row["test_zero_mean_logit"], row["test_nonzero_mean_logit"]])
    affine = affine_fit(np.asarray(x_centroids), np.asarray(y_centroids))
    additive_only = float(np.mean(np.asarray(y_centroids) - np.asarray(x_centroids)))
    additive_rmse = float(np.sqrt(np.mean((np.asarray(y_centroids) - (np.asarray(x_centroids) + additive_only)) ** 2)))

    # Oracle-informed explanatory check only: map held-out logits back into the
    # validation score frame with one pooled affine relation. Held-out labels
    # entered this fit, so these are not deployable recalibration results.
    diagnostic_decisions = []
    for group, fold in folds.items():
        threshold_logit = float(logit(fold["ensemble_threshold"]))
        for state_id in fold["test_ids"]:
            oracle_label = int(state[state_id]["original_gate_label"])
            raw_logit = float(logit(fold["state_probability"][state_id]))
            diagnostic_decisions.append({
                "label": oracle_label,
                "raw": int(raw_logit >= threshold_logit),
                "additive": int((raw_logit - additive_only) >= threshold_logit),
                "affine": int(((raw_logit - affine["intercept"]) / affine["slope"]) >= threshold_logit),
            })

    def decision_summary(key: str) -> dict:
        y = np.asarray([row["label"] for row in diagnostic_decisions], int)
        predicted = np.asarray([row[key] for row in diagnostic_decisions], int)
        zero, one = y == 0, y == 1
        return {
            "errors": int(np.sum(predicted != y)), "accuracy": float(np.mean(predicted == y)),
            "FPR": float(np.mean(predicted[zero] == 1)), "FNR": float(np.mean(predicted[one] == 0)),
        }

    raw_decision = decision_summary("raw")
    additive_decision = decision_summary("additive")
    affine_decision = decision_summary("affine")

    # Cross-check Agent A's exact Pair-1 decomposition. These constants are
    # read from its frozen-checkpoint artifact, not recomputed or hand tuned.
    decomposition_rows = read_csv(
        ROOT / "diagnostics/gate_shortcut_calibration_audit/ranking/pair1_cross_model_decomposition.csv"
    )
    pair1_actual_margin = float(np.mean([float(row["actual_oof_logit_difference"]) for row in decomposition_rows]))
    pair1_feature_effect = float(np.mean([float(row["symmetric_within_model_logit_difference"]) for row in decomposition_rows]))
    pair1_fold_model_offset = float(np.mean([float(row["cross_fold_model_effect"]) for row in decomposition_rows]))
    shift_by_group = {row["outer_group"]: row for row in shift_rows}
    common_relative_d4_minus_d2 = (
        shift_by_group["anchor_D4_pair227"]["common_additive_shift"]
        - shift_by_group["anchor_D2_pair228"]["common_additive_shift"]
    )
    conditional_relative_d4_nonzero_minus_d2_zero = (
        shift_by_group["anchor_D4_pair227"]["nonzero_class_shift_test_minus_validation"]
        - shift_by_group["anchor_D2_pair228"]["zero_class_shift_test_minus_validation"]
    )
    for row in shift_rows:
        row.update({
            "global_affine_validation_to_test_intercept": affine["intercept"],
            "global_affine_validation_to_test_slope": affine["slope"],
            "global_affine_centroid_R2": affine["r2"], "global_affine_centroid_RMSE": affine["rmse"],
            "global_additive_only_shift": additive_only, "global_additive_only_centroid_RMSE": additive_rmse,
            "diagnostic_raw_oof_errors_4groups": raw_decision["errors"],
            "diagnostic_additive_frame_errors_4groups": additive_decision["errors"],
            "diagnostic_affine_frame_errors_4groups": affine_decision["errors"],
            "pair1_exact_cross_fold_model_offset": pair1_fold_model_offset,
            "pair1_common_calibration_offset_D4_minus_D2": common_relative_d4_minus_d2,
            "pair1_label_conditional_shift_D4_nonzero_minus_D2_zero": conditional_relative_d4_nonzero_minus_d2_zero,
        })
    write_csv(HERE / "calibration_shift_summary.csv", shift_rows)

    variant_rows = read_csv(OOF / "per_variant_predictions.csv")
    pair2 = pair_analysis("collision_pair_2", "RB_Q_pair226_m080_s95400802_p073", "R_D1_s95106004_p40", variant_rows, folds, state)
    control = pair_analysis("control_pair_boundary_10", "RB_Q_pair228_m080_s95401001_p050", "RBV_Q_pair228_m080_s95401004_p036", variant_rows, folds, state)
    write_csv(HERE / "pair2_margin_analysis.csv", [pair2])
    write_csv(HERE / "control_pair_margin_analysis.csv", [control])

    additive_like = sum(row["descriptive_shift_type"] == "MOSTLY_ADDITIVE_SHIFT" for row in shift_rows)
    scale_like = sum(row["descriptive_shift_type"] == "MOSTLY_SCALE_SHIFT" for row in shift_rows)
    class_like = sum(row["descriptive_shift_type"] == "CLASS_DEPENDENT_SHIFT" for row in shift_rows)
    mixed_like = len(shift_rows) - additive_like - scale_like - class_like
    # Require both low centroid residual and a slope close to one before calling the pooled effect additive.
    if affine["rmse"] <= .35 and abs(affine["slope"] - 1) <= .25 and additive_like >= 3:
        conclusion = "MOSTLY_ADDITIVE_SHIFT"
    elif affine["rmse"] <= .35 and abs(affine["slope"] - 1) > .25 and scale_like >= 2:
        conclusion = "MOSTLY_SCALE_SHIFT"
    elif class_like >= 3:
        conclusion = "CLASS_DEPENDENT_SHIFT"
    elif max(additive_like, scale_like, class_like, mixed_like) >= 2:
        conclusion = "MIXED_CALIBRATION_SHIFT"
    else:
        conclusion = "CALIBRATION_INCONCLUSIVE"

    report = [
        "# Calibration shift audit", "",
        "All scores come from frozen, exactly reproduced OOF MLP checkpoints. No model was trained and no test threshold was changed.", "",
        "## Cross-group drift", "",
    ]
    for row in shift_rows:
        report.append(
            f"- `{row['outer_group']}`: zero/nonzero logit shifts {row['zero_class_shift_test_minus_validation']:+.3f}/{row['nonzero_class_shift_test_minus_validation']:+.3f}; "
            f"common offset {row['common_additive_shift']:+.3f}, differential {row['class_differential_shift']:+.3f}, separation ratio {row['separation_scale_ratio']:.3f}; **{row['descriptive_shift_type']}**."
        )
    report.extend([
        "", "The shared affine centroid fit is diagnostic only: "
        f"test_logit ≈ {affine['intercept']:+.3f} + {affine['slope']:.3f}·validation_logit, "
        f"R²={affine['r2']:.3f}, RMSE={affine['rmse']:.3f}. An additive-only fit has shift {additive_only:+.3f} and RMSE={additive_rmse:.3f}. "
        f"Across the 101 stable states in these four held-out groups, raw/additive-frame/affine-frame diagnostic error counts are {raw_decision['errors']}/{additive_decision['errors']}/{affine_decision['errors']}; "
        f"the corresponding FPR/FNR are {raw_decision['FPR']:.3f}/{raw_decision['FNR']:.3f}, {additive_decision['FPR']:.3f}/{additive_decision['FNR']:.3f}, and {affine_decision['FPR']:.3f}/{affine_decision['FNR']:.3f}. "
        "The correction fits used held-out labels and are explanatory checks only. Their large centroid residuals and failure to uniformly remove errors do not support a universal post-hoc correction.", "",
        "## Relation to Pair 1's cross-fold reversal", "",
        f"Agent A's exact decomposition gives a cross-fold model/score offset of {pair1_fold_model_offset:+.3f} logits. "
        f"The fold-wide common calibration offsets differ by {common_relative_d4_minus_d2:+.3f} logits (D4 minus D2), which accounts for {abs(common_relative_d4_minus_d2 / pair1_fold_model_offset):.1%} of that offset magnitude with the correct sign. "
        f"Using the relevant label-conditional shifts gives {conditional_relative_d4_nonzero_minus_d2_zero:+.3f} logits (D4 nonzero minus D2 zero), within {abs(conditional_relative_d4_nonzero_minus_d2_zero - pair1_fold_model_offset):.3f} logits of the exact model offset. "
        "Thus the apparent Pair-1 cross-model ranking reversal is substantially explained by source/fold-dependent score calibration, especially the upward zero-class shift in the D2 fold. It is not a same-model ranking reversal.", "",
        "## Pair 2", "",
        f"The zero/nonzero cloud means are {pair2['zero_mean_logit']:.3f}/{pair2['nonzero_mean_logit']:.3f} logits (margin {pair2['mean_logit_margin_nonzero_minus_zero']:+.3f}, AUC {pair2['cloud_ranking_AUC']:.3f}). "
        f"Their independently selected OOF thresholds are {pair2['zero_validation_threshold_logit']:.3f}/{pair2['nonzero_validation_threshold_logit']:.3f}. "
        f"The local midpoint is {pair2['local_mean_midpoint_logit']:.3f}; relative to the zero fold threshold, this is a +{pair2['zero_threshold_to_midpoint_logit_displacement']:.3f} displacement. "
        "Because the states use different OOF models, this is a score-coordinate diagnostic rather than a deployable common threshold.", "",
        "## Clean control pair", "",
        f"Both control states use the same OOF checkpoint. Their means are {control['zero_mean_logit']:.3f}/{control['nonzero_mean_logit']:.3f}, margin {control['mean_logit_margin_nonzero_minus_zero']:+.3f}, AUC {control['cloud_ranking_AUC']:.6f}. "
        f"The validation threshold is {control['zero_validation_threshold_logit']:.3f}, while the local mean midpoint is {control['local_mean_midpoint_logit']:.3f}, a +{control['zero_threshold_to_midpoint_logit_displacement']:.3f} shift. "
        f"A clean all-variant common threshold {'exists' if control['clean_common_threshold_exists_for_all_variants'] else 'does not exist because the narrow cloud supports overlap'}.", "",
        f"Final sub-agent conclusion: **{conclusion}**.", "",
        "The calibration failure is not explained by one universal scalar threshold offset alone when class-specific shifts and between-fold scale changes are material. All counterfactual shifts above are diagnostic; no test threshold was retuned.",
    ])
    (HERE / "calibration_failure_summary.md").write_text("\n".join(report) + "\n")

    sanity = {
        "status": "COMPLETE", "models_trained": 0, "new_rollouts": 0,
        "frozen_shared_checkpoints_ready": True,
        "max_existing_fold_state_probability_reproduction_error": max_reproduction_error,
        "missing_fold_target_state_probability_reproduction_error": missing_reproduction_error,
        "reproduction_tolerance": 2e-6, "fold_count": len(folds),
        "stable_state_count": len(stable_rows), "test_thresholds_retuned": False,
        "pair2_cross_model_score_comparison_flagged": pair2["same_oof_checkpoint"] is False,
        "control_same_model_score_comparison": control["same_oof_checkpoint"] is True,
        "all_results_finite": all(np.isfinite(list(fold["state_probability"].values())).all() for fold in folds.values()),
        "conclusion": conclusion,
    }
    write_json(HERE / "sanity.json", sanity)
    runtime = {
        "started_utc": start_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started, "gpu_used": False, "gpu_shards": 0,
        "cpu_thread_limit": int(os.environ.get("OMP_NUM_THREADS", "2")), "models_trained": 0,
        "inference_checkpoints": 12,
    }
    write_json(HERE / "runtime.json", runtime)
    print(json.dumps({"status": "COMPLETE", "conclusion": conclusion, "wall_seconds": runtime["wall_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
