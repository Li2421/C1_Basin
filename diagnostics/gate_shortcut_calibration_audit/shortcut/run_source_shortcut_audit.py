"""Read-only source-shortcut diagnostic on existing oracle-stable states.

No gate is trained here.  The only fitted objects are explicitly diagnostic,
fixed-regularization linear decoders used to quantify information already in
the frozen 214-D representation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "diagnostics/gphi_training_dataset_v4"
CONF = ROOT / "diagnostics/gphi_gate_confidence_aware_v1"
CV = ROOT / "diagnostics/hard_stable_boundary_crossval"
LOCAL = ROOT / "diagnostics/hard_stable_local_feature_audit"
SCHEMA_EXPANDED = ROOT / "diagnostics/stable_oracle_feature_audit/feature_schema_expanded.csv"


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    n1 = int(np.sum(y == 1))
    n0 = int(np.sum(y == 0))
    if not n1 or not n0:
        return float("nan")
    ranks = rankdata(score, method="average")
    return float((np.sum(ranks[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0))


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    recall = np.mean(pred[y == 1] == 1) if np.any(y == 1) else np.nan
    specificity = np.mean(pred[y == 0] == 0) if np.any(y == 0) else np.nan
    return float((recall + specificity) / 2)


def select_threshold(y: np.ndarray, score: np.ndarray) -> float:
    unique = np.unique(score)
    candidates = np.r_[np.nextafter(unique[0], -np.inf), (unique[:-1] + unique[1:]) / 2,
                       np.nextafter(unique[-1], np.inf), 0.5]
    return float(max(np.unique(candidates), key=lambda t: (balanced_accuracy(y, score >= t), -abs(t - 0.5))))


def fit_logistic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fixed L2 diagnostic logistic regression; preprocessing is train-only."""
    mean = x.mean(0)
    scale = x.std(0)
    scale[scale < 1e-8] = 1.0
    z = np.clip((x - mean) / scale, -25, 25)
    y = y.astype(float)

    def objective(theta):
        w, b = theta[:-1], theta[-1]
        logits = z @ w + b
        loss = np.mean(np.logaddexp(0.0, logits) - y * logits) + 0.5 * np.sum(w * w) / len(y)
        p = 1 / (1 + np.exp(-np.clip(logits, -50, 50)))
        grad_w = z.T @ (p - y) / len(y) + w / len(y)
        grad_b = np.mean(p - y)
        return float(loss), np.r_[grad_w, grad_b]

    result = minimize(objective, np.zeros(z.shape[1] + 1), jac=True, method="L-BFGS-B",
                      options={"maxiter": 300, "ftol": 1e-10})
    if not result.success and result.status not in (1, 2):
        raise RuntimeError(("logistic diagnostic failed", result.message))
    return result.x, mean, scale


def logistic_predict(model, x: np.ndarray) -> np.ndarray:
    theta, mean, scale = model
    z = np.clip((x - mean) / scale, -25, 25)
    logits = z @ theta[:-1] + theta[-1]
    return 1 / (1 + np.exp(-np.clip(logits, -50, 50)))


def eta_squared(values: np.ndarray, categories: np.ndarray) -> np.ndarray:
    grand = values.mean(0)
    total = np.sum((values - grand) ** 2, axis=0)
    between = np.zeros(values.shape[1])
    for category in np.unique(categories):
        subset = values[categories == category]
        between += len(subset) * (subset.mean(0) - grand) ** 2
    return np.divide(between, total, out=np.zeros_like(between), where=total > 1e-15)


def residualize(values: np.ndarray, categories: np.ndarray) -> np.ndarray:
    result = values.copy()
    for category in np.unique(categories):
        keep = categories == category
        result[keep] -= values[keep].mean(0)
    return result


def discrete_mi(xbin: np.ndarray, category: np.ndarray) -> float:
    xb, xinv = np.unique(xbin, return_inverse=True)
    cb, cinv = np.unique(category, return_inverse=True)
    counts = np.zeros((len(xb), len(cb)), float)
    np.add.at(counts, (xinv, cinv), 1)
    joint = counts / counts.sum()
    px = joint.sum(1, keepdims=True)
    pc = joint.sum(0, keepdims=True)
    nz = joint > 0
    return float(np.sum(joint[nz] * np.log(joint[nz] / (px @ pc)[nz])))


def quantile_bins(x: np.ndarray, bins: int = 5) -> np.ndarray:
    if np.ptp(x) < 1e-15:
        return np.zeros(len(x), int)
    cuts = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.digitize(x, cuts, right=True)


def normalized_mi_excess(values: np.ndarray, category: np.ndarray, rng: np.random.Generator,
                         permutations: int = 64) -> np.ndarray:
    _, counts = np.unique(category, return_counts=True)
    probabilities = counts / counts.sum()
    hc = -np.sum(probabilities * np.log(probabilities))
    output = np.zeros(values.shape[1])
    for dimension in range(values.shape[1]):
        binned = quantile_bins(values[:, dimension])
        _, xbcounts = np.unique(binned, return_counts=True)
        xp = xbcounts / xbcounts.sum()
        hx = -np.sum(xp * np.log(xp))
        denominator = min(hx, hc)
        if denominator <= 1e-15:
            continue
        observed = discrete_mi(binned, category)
        null = np.mean([discrete_mi(binned, rng.permutation(category)) for _ in range(permutations)])
        output[dimension] = max(0.0, (observed - null) / denominator)
    return output


def deterministic_source_decoder(x: np.ndarray, source: np.ndarray, state_ids: np.ndarray,
                                 folds: int = 5) -> tuple[float, int]:
    """Fixed ridge least-squares multiclass decoder with within-group held-out states."""
    classes = np.unique(source)
    class_index = {name: index for index, name in enumerate(classes)}
    assignment = np.empty(len(source), int)
    for name in classes:
        indices = np.flatnonzero(source == name)
        ordered = sorted(indices, key=lambda i: hashlib.sha256(state_ids[i].encode()).hexdigest())
        for position, index in enumerate(ordered):
            assignment[index] = position % folds
    predictions, truths = [], []
    for fold in range(folds):
        test = assignment == fold
        train = ~test
        if not np.any(test):
            continue
        mean = x[train].mean(0)
        scale = x[train].std(0)
        scale[scale < 1e-8] = 1.0
        ztrain = np.c_[np.clip((x[train] - mean) / scale, -25, 25), np.ones(np.sum(train))]
        ztest = np.c_[np.clip((x[test] - mean) / scale, -25, 25), np.ones(np.sum(test))]
        target = np.zeros((np.sum(train), len(classes)))
        for row, name in enumerate(source[train]):
            target[row, class_index[name]] = 1.0
        penalty = np.eye(ztrain.shape[1])
        penalty[-1, -1] = 0.0
        weights = np.linalg.solve(ztrain.T @ ztrain + penalty, ztrain.T @ target)
        predictions.extend(classes[np.argmax(ztest @ weights, axis=1)].tolist())
        truths.extend(source[test].tolist())
    return float(np.mean(np.asarray(predictions) == np.asarray(truths))), len(truths)


def loso_label_diagnostic(x: np.ndarray, y: np.ndarray, source: np.ndarray) -> tuple[float, float, int]:
    probabilities = np.full(len(y), np.nan)
    predictions = np.full(len(y), -1)
    for group in np.unique(source):
        test = source == group
        train = ~test
        model = fit_logistic(x[train], y[train])
        train_probability = logistic_predict(model, x[train])
        threshold = select_threshold(y[train], train_probability)
        probabilities[test] = logistic_predict(model, x[test])
        predictions[test] = probabilities[test] >= threshold
    return balanced_accuracy(y, predictions), auc(y, probabilities), int(np.sum(predictions == y))


def exact_fold_normalization(features: np.ndarray, sample_ids: np.ndarray, train_ids: list[str], schema: dict):
    keep = np.isin(sample_ids, train_ids)
    mean = features[keep].mean(0)
    scale = features[keep].std(0)
    scale[scale < 1e-8] = 1.0
    for segment in schema["segments"]:
        if segment["unit"] == "boolean":
            start, length = int(segment["offset"]), int(segment["length"])
            mean[start:start + length] = 0.0
            scale[start:start + length] = 1.0
    return mean, scale


def main() -> None:
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    HERE.mkdir(parents=True, exist_ok=True)
    inputs = {
        "samples": DATA / "samples.npz",
        "schema": DATA / "feature_schema.json",
        "expanded_schema": SCHEMA_EXPANDED,
        "confidence": CONF / "oracle_confidence_dataset.csv",
        "fold_manifest": CV / "fold_manifest.json",
        "oof_predictions": CV / "out_of_fold_predictions.csv",
        "robust_mistakes": LOCAL / "robust_mistakes.csv",
        "failure_classification": LOCAL / "per_state_failure_classification.csv",
        "neighbor_comparisons": LOCAL / "nearest_neighbor_comparisons.csv",
    }
    hashes_before = {key: sha256(path) for key, path in inputs.items()}

    confidence = read_csv(inputs["confidence"])
    stable_rows = [row for row in confidence if row["oracle_confidence_class"] in
                   ("ORACLE_STABLE_ZERO", "ORACLE_STABLE_NONZERO")]
    stable = {row["state_id"]: row for row in stable_rows}
    labels = {sid: int(row["original_gate_label"]) for sid, row in stable.items()}
    ambiguous = {row["state_id"] for row in confidence if row["oracle_confidence_class"] == "ORACLE_AMBIGUOUS"}

    with np.load(inputs["samples"], allow_pickle=False) as loaded:
        features = np.asarray(loaded["features"], float)
        sample_ids = loaded["state_id"].astype(str)
    schema = json.loads(inputs["schema"].read_text())
    expanded = read_csv(inputs["expanded_schema"])
    if features.shape[1] != 214 or len(expanded) != 214:
        raise RuntimeError(("feature schema mismatch", features.shape, len(expanded)))
    dimension_group = np.asarray([row["semantic_group"] for row in expanded])
    dimension_segment = np.asarray([row["segment"] for row in expanded])
    semantic_groups = sorted(set(dimension_group))
    group_indices = {name: np.flatnonzero(dimension_group == name) for name in semantic_groups}

    # One centroid per independent augmented state; no Flow variant is treated as independent.
    centroids = {}
    variant_counts = {}
    for sid in stable:
        indices = np.flatnonzero(sample_ids == sid)
        if len(indices) == 0:
            raise RuntimeError(("missing stable state samples", sid))
        centroids[sid] = features[indices].mean(0)
        variant_counts[sid] = len(indices)
    state_ids = np.asarray(sorted(stable))
    x = np.asarray([centroids[sid] for sid in state_ids])
    y = np.asarray([labels[sid] for sid in state_ids], int)
    source = np.asarray([stable[sid]["source_group"] for sid in state_ids])

    counts = Counter(source)
    multi_names = sorted(name for name, count in counts.items() if count >= 2)
    multi = np.isin(source, multi_names)
    xm, ym, gm, idm = x[multi], y[multi], source[multi], state_ids[multi]
    if any(len(set(ym[gm == name])) != 2 for name in multi_names):
        raise RuntimeError("multi-state source group without both stable labels")

    # Dimension-level source versus oracle-label association.
    source_eta = eta_squared(xm, gm)
    source_eta_after_label = eta_squared(residualize(xm, ym), gm)
    label_eta = eta_squared(xm, ym)
    label_eta_within_source = eta_squared(residualize(xm, gm), ym)
    label_corr = np.zeros(214)
    label_auc = np.full(214, np.nan)
    for d in range(214):
        if np.std(xm[:, d]) > 1e-15:
            label_corr[d] = np.corrcoef(xm[:, d], ym)[0, 1]
            raw_auc = auc(ym, xm[:, d])
            label_auc[d] = max(raw_auc, 1 - raw_auc)
    rng = np.random.default_rng(20260924)
    source_mi = normalized_mi_excess(xm, gm, rng)
    label_mi = normalized_mi_excess(xm, ym, rng)

    source_output = []
    label_output = []
    for d in range(214):
        source_output.append({
            "row_type": "DIMENSION", "dimension": d, "segment": dimension_segment[d],
            "semantic_group": dimension_group[d], "source_eta_squared": source_eta[d],
            "label_adjusted_source_eta_squared": source_eta_after_label[d],
            "source_normalized_MI_excess_5bin": source_mi[d],
            "multi_group_state_count": len(xm), "multi_source_group_count": len(multi_names),
        })
        label_output.append({
            "row_type": "DIMENSION", "dimension": d, "segment": dimension_segment[d],
            "semantic_group": dimension_group[d], "label_eta_squared": label_eta[d],
            "source_adjusted_label_eta_squared": label_eta_within_source[d],
            "absolute_point_biserial_correlation": abs(label_corr[d]),
            "orientation_free_univariate_AUROC": label_auc[d],
            "label_normalized_MI_excess_5bin": label_mi[d],
        })

    # Group-level fixed diagnostic decoders.
    group_metrics = {}
    for name, indices in group_indices.items():
        source_accuracy, decoded = deterministic_source_decoder(xm[:, indices], gm, idm)
        loso_ba, loso_auc, loso_correct = loso_label_diagnostic(xm[:, indices], ym, gm)
        group_metrics[name] = {
            "dimension_count": len(indices),
            "source_decoder_accuracy": source_accuracy,
            "source_decoder_chance": 1 / len(multi_names),
            "source_decoder_test_states": decoded,
            "label_LOSO_balanced_accuracy": loso_ba,
            "label_LOSO_AUROC": loso_auc,
            "label_LOSO_correct": loso_correct,
        }
        source_output.append({
            "row_type": "SEMANTIC_GROUP", "dimension": "", "segment": "",
            "semantic_group": name,
            "source_eta_squared_median": np.median(source_eta[indices]),
            "source_eta_squared_q90": np.quantile(source_eta[indices], .9),
            "label_adjusted_source_eta_squared_median": np.median(source_eta_after_label[indices]),
            "source_normalized_MI_excess_median": np.median(source_mi[indices]),
            **group_metrics[name],
        })
        label_output.append({
            "row_type": "SEMANTIC_GROUP", "dimension": "", "segment": "",
            "semantic_group": name,
            "label_eta_squared_median": np.median(label_eta[indices]),
            "source_adjusted_label_eta_squared_median": np.median(label_eta_within_source[indices]),
            "absolute_point_biserial_median": np.median(np.abs(label_corr[indices])),
            "orientation_free_univariate_AUROC_median": np.nanmedian(label_auc[indices]),
            "label_normalized_MI_excess_median": np.median(label_mi[indices]),
            **group_metrics[name],
        })
    write_csv(HERE / "feature_source_association.csv", source_output)
    write_csv(HERE / "feature_label_association.csv", label_output)

    # Direction consistency of class association across source groups.
    sign_rows = []
    group_sign_summary = {}
    for d in range(214):
        global_difference = float(xm[ym == 1, d].mean() - xm[ym == 0, d].mean())
        effects = {}
        standardized = {}
        for name in multi_names:
            values = xm[gm == name, d]
            target = ym[gm == name]
            difference = float(values[target == 1].mean() - values[target == 0].mean())
            spread = float(values.std())
            effects[name] = difference
            standardized[name] = difference / spread if spread > 1e-12 else 0.0
        global_sign = np.sign(global_difference)
        informative = [name for name in multi_names if abs(standardized[name]) >= 0.2]
        disagree = [name for name in informative if np.sign(effects[name]) not in (0, global_sign)]
        positive = sum(effects[name] > 0 for name in multi_names)
        negative = sum(effects[name] < 0 for name in multi_names)
        sign_rows.append({
            "row_type": "DIMENSION", "dimension": d, "segment": dimension_segment[d],
            "semantic_group": dimension_group[d], "global_nonzero_minus_zero": global_difference,
            "positive_source_groups": positive, "negative_source_groups": negative,
            "informative_source_groups_abs_std_effect_ge_0p2": len(informative),
            "sign_disagreeing_informative_groups": len(disagree),
            "sign_disagreement_fraction": len(disagree) / len(informative) if informative else "",
            "per_group_standardized_effect_json": json.dumps(standardized, sort_keys=True),
        })
    for name, indices in group_indices.items():
        rows = [sign_rows[d] for d in indices]
        fractions = [float(row["sign_disagreement_fraction"]) for row in rows
                     if row["sign_disagreement_fraction"] != ""]
        mixed = [row for row in rows if row["positive_source_groups"] > 0 and row["negative_source_groups"] > 0]
        summary = {
            "row_type": "SEMANTIC_GROUP", "dimension": "", "segment": "", "semantic_group": name,
            "dimension_count": len(indices),
            "dimensions_with_both_positive_and_negative_group_effects": len(mixed),
            "mixed_sign_dimension_fraction": len(mixed) / len(indices),
            "median_informative_sign_disagreement_fraction": np.median(fractions) if fractions else "",
        }
        group_sign_summary[name] = summary
        sign_rows.append(summary)
    write_csv(HERE / "cross_group_correlation_signs.csv", sign_rows)

    # Robust-error shortcut matches using exact OOF scores and each fold's frozen training membership.
    robust = read_csv(inputs["robust_mistakes"])
    failure = {row["state_id"]: row for row in read_csv(inputs["failure_classification"])}
    folds = json.loads(inputs["fold_manifest"].read_text())["folds"]
    fold_by_group = {row["heldout_source_group"]: row for row in folds}
    oof = [row for row in read_csv(inputs["oof_predictions"])
           if row["model"] == "MLP_64x64" and row["training_seed"] == "SEED_MEAN"]
    oof_by_state = {row["state_id"]: row for row in oof}
    oof_by_group = defaultdict(list)
    for row in oof:
        oof_by_group[row["heldout_source_group"]].append(row)

    shortcut_rows = []
    for row in robust:
        sid = row["state_id"]
        group = row["source_group"]
        label = int(row["oracle_gate_label"])
        fold = fold_by_group[group]
        train_ids = fold["normalization_fit_state_ids"]
        mean, scale = exact_fold_normalization(features, sample_ids, train_ids, schema)
        train_state_ids = np.asarray([candidate for candidate in train_ids if candidate in centroids])
        train_x = np.asarray([(centroids[candidate] - mean) / scale for candidate in train_state_ids])
        train_y = np.asarray([labels[candidate] for candidate in train_state_ids])
        query = (centroids[sid] - mean) / scale
        class_distances = {}
        nearest_group_records = {}
        opposite_source_majority_count = 0
        for feature_group, indices in {**group_indices, "all_214D": np.arange(214)}.items():
            class_distances[feature_group] = {
                str(target): float(np.sqrt(np.mean((query[indices] - train_x[train_y == target][:, indices].mean(0)) ** 2)))
                for target in (0, 1)
            }
            candidates = []
            for source_name in sorted({stable[candidate]["source_group"] for candidate in train_state_ids}):
                members = np.asarray([stable[candidate]["source_group"] == source_name for candidate in train_state_ids])
                if np.sum(members) < 2:
                    continue
                centroid = train_x[members][:, indices].mean(0)
                distance = float(np.sqrt(np.mean((query[indices] - centroid) ** 2)))
                rate = float(np.mean(train_y[members]))
                candidates.append((distance, source_name, rate, int(np.sum(members))))
            nearest = min(candidates)
            nearest_majority = int(nearest[2] >= 0.5)
            opposite_source_majority_count += nearest_majority != label
            nearest_group_records[feature_group] = {
                "source_group": nearest[1], "distance": nearest[0], "stable_nonzero_rate": nearest[2],
                "state_count": nearest[3], "majority_label": nearest_majority,
            }

        prediction = oof_by_state[sid]
        probability = float(prediction["p_gate"])
        heldout = oof_by_group[group]
        score_medians = {
            target: float(np.median([float(item["p_gate"]) for item in heldout if int(item["oracle_label"]) == target]))
            for target in (0, 1)
        }
        score_resembles = min((abs(probability - score_medians[target]), target) for target in (0, 1))[1]
        item = failure[sid]
        shortcut_rows.append({
            "state_id": sid, "source_group": group, "oracle_label": label,
            "OOF_p_gate": probability, "OOF_threshold": float(prediction["validation_selected_threshold"]),
            "OOF_predicted_label": int(prediction["predicted_label"]),
            "heldout_group_zero_score_median": score_medians[0],
            "heldout_group_nonzero_score_median": score_medians[1],
            "score_resembles_heldout_class": score_resembles,
            "score_resemblance_is_wrong_class": score_resembles != label,
            "nearest_same_label_training_state": item["nearest_same_label_state"],
            "nearest_same_label_distance": float(item["nearest_same_label_distance"]),
            "nearest_opposite_label_training_state": item["nearest_opposite_label_state"],
            "nearest_opposite_label_distance": float(item["nearest_opposite_label_distance"]),
            "nearest_training_neighbor_is_opposite": float(item["nearest_opposite_label_distance"]) < float(item["nearest_same_label_distance"]),
            "fold_train_class_centroid_distances_json": json.dumps(class_distances, sort_keys=True),
            "nearest_training_source_groups_json": json.dumps(nearest_group_records, sort_keys=True),
            "feature_groups_whose_nearest_source_majority_is_opposite": opposite_source_majority_count,
            "feature_group_count_including_all214": len(group_indices) + 1,
        })
    write_csv(HERE / "robust_error_shortcut_matches.csv", shortcut_rows)

    hashes_after = {key: sha256(path) for key, path in inputs.items()}
    runtime = time.monotonic() - started
    sanity = {
        "passed": hashes_before == hashes_after and len(stable) == 277 and len(ambiguous) == 47
                  and len(robust) == 6 and all(variant_counts[sid] == 64 for sid in stable),
        "input_hashes_unchanged": hashes_before == hashes_after,
        "feature_dimension": features.shape[1],
        "oracle_stable_state_count": len(stable),
        "oracle_ambiguous_excluded_count": len(ambiguous),
        "stable_zero": int(np.sum(y == 0)),
        "stable_nonzero": int(np.sum(y == 1)),
        "source_group_count": len(counts),
        "multi_state_mixed_label_source_group_count": len(multi_names),
        "states_in_multi_state_groups": len(xm),
        "robust_error_count": len(robust),
        "all_stable_states_have_64_variants": all(variant_counts[sid] == 64 for sid in stable),
        "diagnostic_only_no_gate_training": True,
        "gpu_used": False,
    }
    write_json(HERE / "sanity.json", sanity)
    write_json(HERE / "runtime.json", {
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": runtime,
        "cpu_processes": 1,
        "cpu_thread_cap": 2,
        "gpu_count": 0,
        "gpu_shards": 0,
        "new_rollouts": 0,
        "models_trained": 0,
        "diagnostic_linear_fits": "fixed logistic/ridge probes only",
    })
    write_json(HERE / "core_results.json", {
        "semantic_group_metrics": group_metrics,
        "group_sign_summary": group_sign_summary,
        "robust_shortcut_rows": shortcut_rows,
        "source_groups_used_for_primary_association": multi_names,
    })
    print(json.dumps({"runtime_s": runtime, "sanity_passed": sanity["passed"],
                      "group_metrics": group_metrics}, indent=2))


if __name__ == "__main__":
    main()
