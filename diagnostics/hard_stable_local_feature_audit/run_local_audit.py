"""Read-only local feature audit for six robust hard-boundary gate mistakes."""

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

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CV = ROOT / "diagnostics/hard_stable_boundary_crossval"
FEATURE = ROOT / "diagnostics/stable_oracle_feature_audit"
ORACLE = ROOT / "diagnostics/oracle_boundary_confidence_audit"
CONF = ROOT / "diagnostics/gphi_gate_confidence_aware_v1"
DATA = ROOT / "diagnostics/gphi_training_dataset_v4"


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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    keys = fieldnames or list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def compact(value: np.ndarray):
    value = np.asarray(value)
    if value.ndim == 0:
        scalar = value.item()
        return bool(scalar) if isinstance(scalar, (bool, np.bool_)) else scalar
    flat = value.astype(float).reshape(-1)
    return {
        "shape": list(value.shape),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "first": flat[: min(4, len(flat))].tolist(),
        "last": flat[-min(4, len(flat)) :].tolist(),
    }


def auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, int)
    pos = int(y.sum())
    neg = len(y) - pos
    if not pos or not neg:
        return float("nan")
    ranks = rankdata(score, method="average")
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def select_univariate_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    unique = np.unique(score)
    candidates = np.r_[
        np.nextafter(unique[0], -np.inf),
        (unique[:-1] + unique[1:]) / 2,
        np.nextafter(unique[-1], np.inf),
    ]
    best = None
    for threshold in candidates:
        prediction = score >= threshold
        recall = float(np.mean(prediction[y == 1]))
        specificity = float(np.mean(~prediction[y == 0]))
        value = (recall + specificity) / 2
        candidate = (value, -abs(float(threshold)), float(threshold))
        if best is None or candidate > best:
            best = candidate
    return best[2], best[0]


def fit_logistic(x: np.ndarray, y: np.ndarray, C: float = 10.0) -> np.ndarray:
    def objective(theta):
        weights = theta[:-1]
        z = x @ weights + theta[-1]
        p = 1 / (1 + np.exp(-np.clip(z, -40, 40)))
        loss = np.logaddexp(0, z).sum() - y @ z + 0.5 / C * (weights @ weights)
        gradient = np.r_[x.T @ (p - y) + weights / C, np.sum(p - y)]
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(x.shape[1] + 1),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success and result.status not in (1, 2):
        raise RuntimeError(("diagnostic logistic failed", result.message))
    return np.asarray(result.x)


def silu(x):
    clipped = np.clip(x, -40, 40)
    return x / (1 + np.exp(-clipped))


def silu_derivative(x):
    sigmoid = 1 / (1 + np.exp(-np.clip(x, -40, 40)))
    return sigmoid + x * sigmoid * (1 - sigmoid)


def frozen_gate_probability_and_gradient(checkpoint, features: np.ndarray):
    mean = checkpoint["normalization_mean"]
    scale = checkpoint["normalization_scale"]
    x = (features - mean) / scale
    w0 = checkpoint["layer_0_weight"]
    b0 = checkpoint["layer_0_bias"]
    w1 = checkpoint["layer_1_weight"]
    b1 = checkpoint["layer_1_bias"]
    w2 = checkpoint["layer_2_weight"]
    b2 = checkpoint["layer_2_bias"]
    z0 = x @ w0 + b0
    a0 = silu(z0)
    z1 = a0 @ w1 + b1
    a1 = silu(z1)
    logit = (a1 @ w2 + b2).reshape(-1)
    probability = 1 / (1 + np.exp(-np.clip(logit, -40, 40)))
    gradients = []
    for index in range(len(x)):
        grad_a1 = w2[:, 0]
        grad_z1 = grad_a1 * silu_derivative(z1[index])
        grad_a0 = w1 @ grad_z1
        grad_z0 = grad_a0 * silu_derivative(z0[index])
        gradients.append(w0 @ grad_z0)
    return probability, np.asarray(gradients)


def snapshot_vector(snapshot: dict[str, np.ndarray], fields: list[str]) -> np.ndarray:
    return np.concatenate([np.asarray(snapshot[field], float).reshape(-1) for field in fields])


def main() -> None:
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    HERE.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "crossval_repeated_errors": CV / "repeated_errors.csv",
        "crossval_predictions": CV / "out_of_fold_predictions.csv",
        "crossval_seed_sensitivity": CV / "seed_sensitivity.csv",
        "crossval_fold_manifest": CV / "fold_manifest.json",
        "crossval_manifest": CV / "manifest.json",
        "expanded_schema": FEATURE / "feature_schema_expanded.csv",
        "feature_audit_manifest": FEATURE / "manifest.json",
        "oracle_stability": ORACLE / "b63_resampling_stability.csv",
        "confidence_dataset": CONF / "oracle_confidence_dataset.csv",
        "confidence_checkpoint": CONF / "best_gate_checkpoint.npz",
        "dataset_samples": DATA / "samples.npz",
        "dataset_state_manifest": DATA / "state_manifest.jsonl",
        "dataset_manifest": DATA / "manifest.json",
    }
    before = {name: sha(path) for name, path in source_paths.items()}

    repeated = list(csv.DictReader(source_paths["crossval_repeated_errors"].open()))
    robust_ids = sorted(row["state_id"] for row in repeated)
    sensitivity = list(csv.DictReader(source_paths["crossval_seed_sensitivity"].open()))
    reproduced = sorted(
        row["state_id"]
        for row in sensitivity
        if row["model"] == "MLP_64x64" and row["sensitivity_class"] == "ROBUST_MISTAKE"
    )
    if robust_ids != reproduced or len(robust_ids) != 6:
        raise RuntimeError(("robust mistake reproduction failed", robust_ids, reproduced))

    confidence_rows = list(csv.DictReader(source_paths["confidence_dataset"].open()))
    stable_rows = [row for row in confidence_rows if row["oracle_confidence_class"] != "ORACLE_AMBIGUOUS"]
    stable = {row["state_id"]: row for row in stable_rows}
    labels = {state_id: int(row["original_gate_label"]) for state_id, row in stable.items()}
    if not set(robust_ids) <= set(stable):
        raise RuntimeError("robust mistakes not stable")
    manifest_rows = read_jsonl(source_paths["dataset_state_manifest"])
    state_manifest = {row["state_id"]: row for row in manifest_rows}
    fold_manifest = read_json(source_paths["crossval_fold_manifest"])
    fold_by_group = {fold["heldout_source_group"]: fold for fold in fold_manifest["folds"]}
    oof = list(csv.DictReader(source_paths["crossval_predictions"].open()))

    schema_rows = list(csv.DictReader(source_paths["expanded_schema"].open()))
    dimension_name = {
        int(row["dimension"]): f"{row['segment']}.{row['segment_component']}"
        for row in schema_rows
    }
    semantic_indices = defaultdict(list)
    segment_indices = defaultdict(list)
    for row in schema_rows:
        index = int(row["dimension"])
        semantic_indices[row["semantic_group"]].append(index)
        segment_indices[row["segment"]].append(index)
    semantic_indices["all_214D"] = list(range(214))

    with np.load(source_paths["dataset_samples"], allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    features = arrays["features"]
    sample_state_ids = arrays["state_id"].astype(str)
    if features.shape != (20736, 214):
        raise RuntimeError(features.shape)
    state_mean = {
        state_id: features[sample_state_ids == state_id].mean(axis=0)
        for state_id in stable
    }
    sample_indices = {
        state_id: np.flatnonzero(sample_state_ids == state_id) for state_id in stable
    }

    snapshots = {}
    for state_id, row in state_manifest.items():
        if state_id not in stable:
            continue
        with np.load(DATA / row["state_file"], allow_pickle=False) as source:
            snapshots[state_id] = {key: np.asarray(source[key]).copy() for key in source.files}
    snapshot_fields = list(next(iter(snapshots.values())))
    omitted_fields = {
        "first_success_step",
        "first_deadlock_step",
        "first_wall_collision_step",
        "first_agent_collision_step",
        "done",
    }
    field_encoding = {
        "positions": "ENCODED_214D",
        "velocities": "ENCODED_214D",
        "step": "ENCODED_214D",
        "error_history": "ENCODED_214D",
        "history_start_step": "ENCODED_214D",
        "candidate_since": "ENCODED_214D",
        "stuck_timer": "ENCODED_214D",
        "max_stuck_timer": "ENCODED_214D",
        "ever_candidate_deadlock": "ENCODED_214D",
        **{field: "OMITTED_214D_EVENT_LATCH" for field in omitted_fields},
    }

    with np.load(source_paths["confidence_checkpoint"], allow_pickle=False) as loaded:
        checkpoint = {key: np.asarray(loaded[key]) for key in loaded.files}
    frozen_gate_threshold = float(checkpoint["threshold"])
    existing_gate_probability = {}
    robust_gate_gradient = {}
    for state_id in stable:
        index = sample_indices[state_id]
        probability, gradient = frozen_gate_probability_and_gradient(checkpoint, features[index])
        existing_gate_probability[state_id] = float(np.mean(probability))
        if state_id in robust_ids:
            robust_gate_gradient[state_id] = gradient

    robust_output = []
    for state_id in robust_ids:
        row = stable[state_id]
        meta = state_manifest[state_id]
        snapshot = snapshots[state_id]
        seed_rows = sorted(
            (
                item
                for item in oof
                if item["model"] == "MLP_64x64"
                and item["state_id"] == state_id
                and item["training_seed"] != "SEED_MEAN"
            ),
            key=lambda item: int(item["training_seed"]),
        )
        if len(seed_rows) != 3 or any(item["correct"] != "False" for item in seed_rows):
            raise RuntimeError(("not robustly wrong", state_id, seed_rows))
        mean_feature = state_mean[state_id]
        robust_output.append(
            {
                "state_id": state_id,
                "source_group": row["source_group"],
                "source_trajectory": row["source_trajectory"],
                "category": row["category"],
                "recovery_phase": meta.get("recovery_phase", ""),
                "oracle_gate_label": labels[state_id],
                "enlarged_Q0": float(row["Q0"]),
                "oracle_label_reproduction_probability": float(row["original_label_reproduction_probability"]),
                "oracle_label_flip_probability": float(row["original_label_flip_probability"]),
                "continuation_count": int(row["n"]),
                "training_seeds": "|".join(item["training_seed"] for item in seed_rows),
                "p_gate_three_seeds": "|".join(f"{float(item['p_gate']):.12g}" for item in seed_rows),
                "predictions_three_seeds": "|".join(item["predicted_label"] for item in seed_rows),
                "validation_thresholds_three_seeds": "|".join(f"{float(item['validation_selected_threshold']):.12g}" for item in seed_rows),
                "existing_frozen_gate_p": existing_gate_probability[state_id],
                "existing_frozen_gate_threshold": frozen_gate_threshold,
                "existing_frozen_gate_prediction": int(existing_gate_probability[state_id] >= frozen_gate_threshold),
                "existing_frozen_gate_correct": int(existing_gate_probability[state_id] >= frozen_gate_threshold) == labels[state_id],
                "step": int(np.asarray(snapshot["step"])),
                "positions": json.dumps(np.asarray(snapshot["positions"]).tolist()),
                "last_executed_velocities": json.dumps(np.asarray(snapshot["velocities"]).tolist()),
                "goal_errors": json.dumps(mean_feature[60:62].tolist()),
                "inter_agent_distance": float(meta["inter_agent_distance"]),
                "relative_velocity_norm": float(meta["relative_velocity_norm"]),
                "candidate_since": int(np.asarray(snapshot["candidate_since"])),
                "stuck_timer": float(np.asarray(snapshot["stuck_timer"])),
                "max_stuck_timer": float(np.asarray(snapshot["max_stuck_timer"])),
                "ever_candidate_deadlock": bool(np.asarray(snapshot["ever_candidate_deadlock"])),
                "source_local_step": meta.get("source_local_step", ""),
                "source_eta": json.dumps(meta.get("source_eta", [])),
            }
        )
    write_csv(HERE / "robust_mistakes.csv", robust_output)
    frozen_gate_correct = sum(row["existing_frozen_gate_correct"] is True for row in robust_output)

    neighbor_rows = []
    group_distance_rows = []
    knn_rows = []
    snapshot_rows = []
    state_evidence = {}
    heldout_leakage = []

    for bad_id in robust_ids:
        bad_group = stable[bad_id]["source_group"]
        fold = fold_by_group[bad_group]
        train_ids = [state_id for state_id in fold["normalization_fit_state_ids"] if state_id in stable]
        if any(stable[state_id]["source_group"] == bad_group for state_id in train_ids):
            heldout_leakage.append(bad_id)
        train_sample_index = np.flatnonzero(np.isin(sample_state_ids, train_ids))
        mean = features[train_sample_index].mean(axis=0)
        scale = features[train_sample_index].std(axis=0)
        scale[scale < 1e-8] = 1.0
        z_bad = (state_mean[bad_id] - mean) / scale
        z_train = np.stack([(state_mean[state_id] - mean) / scale for state_id in train_ids])
        y_train = np.asarray([labels[state_id] for state_id in train_ids], int)
        distances = np.sqrt(np.mean((z_train - z_bad) ** 2, axis=1))

        snapshot_matrix = np.stack([snapshot_vector(snapshots[state_id], snapshot_fields) for state_id in train_ids])
        snapshot_mean = snapshot_matrix.mean(axis=0)
        snapshot_scale = snapshot_matrix.std(axis=0)
        snapshot_scale[snapshot_scale < 1e-8] = 1.0
        snapshot_bad = snapshot_vector(snapshots[bad_id], snapshot_fields)

        nearest_by_label = {}
        for target_label, relation in ((labels[bad_id], "SAME_LABEL"), (1 - labels[bad_id], "OPPOSITE_LABEL")):
            candidates = np.flatnonzero(y_train == target_label)
            ordered = candidates[np.argsort(distances[candidates], kind="stable")]
            nearest_by_label[target_label] = train_ids[int(ordered[0])]
            for rank, index in enumerate(ordered[:5], start=1):
                neighbor_id = train_ids[int(index)]
                full_distance = float(
                    np.sqrt(
                        np.mean(
                            (
                                (snapshot_bad - snapshot_mean) / snapshot_scale
                                - (snapshot_vector(snapshots[neighbor_id], snapshot_fields) - snapshot_mean) / snapshot_scale
                            )
                            ** 2
                        )
                    )
                )
                neighbor_rows.append(
                    {
                        "robust_state_id": bad_id,
                        "pool": "FOLD_TRAIN_CROSS_SOURCE",
                        "label_relation": relation,
                        "neighbor_rank": rank,
                        "neighbor_state_id": neighbor_id,
                        "neighbor_oracle_label": labels[neighbor_id],
                        "neighbor_source_group": stable[neighbor_id]["source_group"],
                        "normalized_214D_RMS_distance": float(distances[index]),
                        "normalized_full_snapshot_RMS_distance": full_distance,
                        "neighbor_Q0": float(stable[neighbor_id]["Q0"]),
                        "neighbor_existing_gate_p": existing_gate_probability[neighbor_id],
                        "eligible_for_knn": True,
                    }
                )

        own_source_ids = [
            state_id
            for state_id in stable
            if state_id != bad_id and stable[state_id]["source_group"] == bad_group
        ]
        own_source_nearest = {}
        for target_label, relation in ((labels[bad_id], "SAME_LABEL"), (1 - labels[bad_id], "OPPOSITE_LABEL")):
            eligible = [state_id for state_id in own_source_ids if labels[state_id] == target_label]
            ordered = sorted(
                eligible,
                key=lambda state_id: float(
                    np.sqrt(np.mean((((state_mean[state_id] - mean) / scale) - z_bad) ** 2))
                ),
            )
            if ordered:
                own_source_nearest[target_label] = (
                    ordered[0],
                    float(np.sqrt(np.mean((((state_mean[ordered[0]] - mean) / scale) - z_bad) ** 2))),
                )
            for rank, neighbor_id in enumerate(ordered[:3], start=1):
                feature_distance = float(np.sqrt(np.mean((((state_mean[neighbor_id] - mean) / scale) - z_bad) ** 2)))
                full_distance = float(
                    np.sqrt(
                        np.mean(
                            (
                                (snapshot_bad - snapshot_mean) / snapshot_scale
                                - (snapshot_vector(snapshots[neighbor_id], snapshot_fields) - snapshot_mean) / snapshot_scale
                            )
                            ** 2
                        )
                    )
                )
                neighbor_rows.append(
                    {
                        "robust_state_id": bad_id,
                        "pool": "OWN_SOURCE_DIAGNOSTIC_ONLY",
                        "label_relation": relation,
                        "neighbor_rank": rank,
                        "neighbor_state_id": neighbor_id,
                        "neighbor_oracle_label": labels[neighbor_id],
                        "neighbor_source_group": stable[neighbor_id]["source_group"],
                        "normalized_214D_RMS_distance": feature_distance,
                        "normalized_full_snapshot_RMS_distance": full_distance,
                        "neighbor_Q0": float(stable[neighbor_id]["Q0"]),
                        "neighbor_existing_gate_p": existing_gate_probability[neighbor_id],
                        "eligible_for_knn": False,
                    }
                )

        same_id = nearest_by_label[labels[bad_id]]
        opposite_id = nearest_by_label[1 - labels[bad_id]]
        same_index = train_ids.index(same_id)
        opposite_index = train_ids.index(opposite_id)
        d_same = float(distances[same_index])
        d_opposite = float(distances[opposite_index])

        class_mean = {
            label_value: z_train[y_train == label_value].mean(axis=0) for label_value in (0, 1)
        }
        separation = np.abs(class_mean[1] - class_mean[0])
        true_closer = np.abs(z_bad - class_mean[labels[bad_id]]) < np.abs(z_bad - class_mean[1 - labels[bad_id]])
        class_consistent = (separation >= 0.5) & true_closer

        all_group_indices = [("SEMANTIC", name, indices) for name, indices in semantic_indices.items()]
        all_group_indices += [("SEGMENT", name, indices) for name, indices in segment_indices.items()]
        collision_by_semantic = {}
        for level, group, indices_list in all_group_indices:
            indices = np.asarray(indices_list, int)
            group_bad = z_bad[indices]
            same_distances = np.sqrt(np.mean((z_train[y_train == labels[bad_id]][:, indices] - group_bad) ** 2, axis=1))
            opposite_distances = np.sqrt(np.mean((z_train[y_train != labels[bad_id]][:, indices] - group_bad) ** 2, axis=1))
            group_same = float(np.min(same_distances))
            group_opposite = float(np.min(opposite_distances))
            ratio = group_opposite / max(group_same, 1e-12)
            if level == "SEMANTIC":
                collision_by_semantic[group] = ratio
            consistent_indices = indices[class_consistent[indices]]
            group_distance_rows.append(
                {
                    "row_type": "LOCAL_COLLISION",
                    "robust_state_id": bad_id,
                    "comparison_neighbor_id": "",
                    "comparison_relation": "",
                    "group_level": level,
                    "feature_group": group,
                    "dimension_count": len(indices),
                    "normalized_L2_distance": "",
                    "normalized_RMS_distance": "",
                    "maximum_abs_dimension_difference": "",
                    "maximum_difference_dimension": "",
                    "almost_identical_dimension_count_abs_lt_0.05": "",
                    "nearest_same_label_distance": group_same,
                    "nearest_opposite_label_distance": group_opposite,
                    "opposite_over_same_ratio": ratio,
                    "true_class_consistent_strong_dimension_count": len(consistent_indices),
                    "true_class_consistent_strong_dimensions": "|".join(dimension_name[int(index)] for index in consistent_indices[:10]),
                }
            )
            for relation, neighbor_id in (("NEAREST_SAME", same_id), ("NEAREST_OPPOSITE", opposite_id)):
                neighbor_z = (state_mean[neighbor_id] - mean) / scale
                delta = z_bad[indices] - neighbor_z[indices]
                max_local = int(np.argmax(np.abs(delta)))
                max_index = int(indices[max_local])
                group_distance_rows.append(
                    {
                        "row_type": "PAIR_DISTANCE",
                        "robust_state_id": bad_id,
                        "comparison_neighbor_id": neighbor_id,
                        "comparison_relation": relation,
                        "group_level": level,
                        "feature_group": group,
                        "dimension_count": len(indices),
                        "normalized_L2_distance": float(np.linalg.norm(delta)),
                        "normalized_RMS_distance": float(np.sqrt(np.mean(delta**2))),
                        "maximum_abs_dimension_difference": float(np.max(np.abs(delta))),
                        "maximum_difference_dimension": f"{max_index}:{dimension_name[max_index]}",
                        "almost_identical_dimension_count_abs_lt_0.05": int(np.sum(np.abs(delta) < 0.05)),
                        "nearest_same_label_distance": "",
                        "nearest_opposite_label_distance": "",
                        "opposite_over_same_ratio": "",
                        "true_class_consistent_strong_dimension_count": len(consistent_indices),
                        "true_class_consistent_strong_dimensions": "|".join(dimension_name[int(index)] for index in consistent_indices[:10]),
                    }
                )

        sorted_neighbors = np.argsort(distances, kind="stable")
        for k in (3, 5, 7):
            selected = sorted_neighbors[:k]
            selected_y = y_train[selected]
            selected_d = distances[selected]
            same_mask = selected_y == labels[bad_id]
            knn_rows.append(
                {
                    "robust_state_id": bad_id,
                    "heldout_source_group": bad_group,
                    "k": k,
                    "true_label": labels[bad_id],
                    "true_label_neighbor_count": int(np.sum(same_mask)),
                    "opposite_label_neighbor_count": int(np.sum(~same_mask)),
                    "true_label_fraction": float(np.mean(same_mask)),
                    "predicted_knn_label": int(np.mean(selected_y) >= 0.5),
                    "knn_correct": int(np.mean(selected_y) >= 0.5) == labels[bad_id],
                    "average_same_label_distance_in_k": float(np.mean(selected_d[same_mask])) if np.any(same_mask) else "",
                    "average_opposite_label_distance_in_k": float(np.mean(selected_d[~same_mask])) if np.any(~same_mask) else "",
                    "neighbor_state_ids": "|".join(train_ids[int(index)] for index in selected),
                    "neighbor_labels": "|".join(str(int(value)) for value in selected_y),
                    "heldout_group_excluded": True,
                }
            )

        same_inside_opposite = [
            train_ids[index]
            for index in range(len(train_ids))
            if y_train[index] == labels[bad_id] and distances[index] <= d_opposite + 1e-12
        ]
        support_groups = sorted({stable[state_id]["source_group"] for state_id in same_inside_opposite})
        own_same = own_source_nearest.get(labels[bad_id], ("", float("inf")))
        own_opposite = own_source_nearest.get(1 - labels[bad_id], ("", float("inf")))
        own_source_ratio = own_opposite[1] / max(own_same[1], 1e-12)

        opposite_snapshot = snapshots[opposite_id]
        for field in snapshot_fields:
            bad_value = np.asarray(snapshots[bad_id][field])
            neighbor_value = np.asarray(opposite_snapshot[field])
            numeric_bad = bad_value.astype(float)
            numeric_neighbor = neighbor_value.astype(float)
            delta = numeric_bad - numeric_neighbor
            differs = not np.array_equal(bad_value, neighbor_value)
            snapshot_rows.append(
                {
                    "robust_state_id": bad_id,
                    "nearest_opposite_state_id": opposite_id,
                    "field": field,
                    "encoding_status": field_encoding[field],
                    "deployment_available_at_t": True,
                    "differs": differs,
                    "max_abs_difference": float(np.max(np.abs(delta))),
                    "RMS_difference": float(np.sqrt(np.mean(delta**2))),
                    "robust_value_summary": json.dumps(compact(bad_value), sort_keys=True),
                    "opposite_value_summary": json.dumps(compact(neighbor_value), sort_keys=True),
                    "potentially_useful_omitted_distinction": field in omitted_fields and differs,
                }
            )

        state_evidence[bad_id] = {
            "train_ids": train_ids,
            "mean": mean,
            "scale": scale,
            "z_bad": z_bad,
            "z_train": z_train,
            "y_train": y_train,
            "d_same": d_same,
            "d_opposite": d_opposite,
            "ratio": d_opposite / max(d_same, 1e-12),
            "nearest_same": same_id,
            "nearest_opposite": opposite_id,
            "knn5_true_fraction": next(row["true_label_fraction"] for row in knn_rows if row["robust_state_id"] == bad_id and row["k"] == 5),
            "same_support_count": len(same_inside_opposite),
            "same_support_groups": support_groups,
            "own_source_nearest_same": own_same[0],
            "own_source_nearest_same_distance": own_same[1],
            "own_source_nearest_opposite": own_opposite[0],
            "own_source_nearest_opposite_distance": own_opposite[1],
            "own_source_opposite_over_same_ratio": own_source_ratio,
            "collision_by_semantic": collision_by_semantic,
        }

    write_csv(HERE / "nearest_neighbor_comparisons.csv", neighbor_rows)
    write_csv(HERE / "feature_group_distances.csv", group_distance_rows)
    write_csv(HERE / "local_knn_geometry.csv", knn_rows)
    write_csv(HERE / "augmented_state_field_comparison.csv", snapshot_rows)

    # Training-only feature selection and fixed tiny L2 logistic probes.
    probe_rows = []
    probe_summary = defaultdict(list)
    for bad_id in robust_ids:
        evidence = state_evidence[bad_id]
        x = evidence["z_train"]
        y = evidence["y_train"]
        x_bad = evidence["z_bad"]
        univariate = []
        for dimension in range(214):
            raw_auc = auc(y, x[:, dimension])
            orientation = 1.0 if raw_auc >= 0.5 else -1.0
            oriented_auc = max(raw_auc, 1 - raw_auc)
            threshold, train_balanced = select_univariate_threshold(y, orientation * x[:, dimension])
            prediction = int(orientation * x_bad[dimension] >= threshold)
            univariate.append((oriented_auc, train_balanced, dimension, orientation, threshold, prediction))
        top = sorted(univariate, key=lambda item: (-item[0], -item[1], item[2]))[:5]
        for rank, (train_auc, train_balanced, dimension, orientation, threshold, prediction) in enumerate(top, start=1):
            correct = prediction == labels[bad_id]
            probe_rows.append(
                {
                    "robust_state_id": bad_id,
                    "probe_type": "UNIVARIATE_TOP_TRAIN_ONLY",
                    "feature_set": f"dimension_{dimension}:{dimension_name[dimension]}",
                    "feature_count": 1,
                    "selection_data": "fold training source groups only",
                    "train_oriented_AUROC": train_auc,
                    "train_balanced_accuracy": train_balanced,
                    "heldout_probability": "",
                    "heldout_score": float(orientation * x_bad[dimension]),
                    "decision_threshold": threshold,
                    "heldout_prediction": prediction,
                    "heldout_correct": correct,
                    "rank": rank,
                }
            )
        top_dimensions = np.asarray([item[2] for item in sorted(univariate, key=lambda item: (-item[0], -item[1], item[2]))[:8]], int)
        probe_sets = [(name, np.asarray(indices, int)) for name, indices in semantic_indices.items() if name != "all_214D"]
        probe_sets.append(("TOP8_UNIVARIATE_TRAIN_ONLY", top_dimensions))
        for name, indices in probe_sets:
            theta = fit_logistic(x[:, indices], y, C=10.0)
            logit = float(x_bad[indices] @ theta[:-1] + theta[-1])
            probability = float(1 / (1 + math.exp(-max(min(logit, 40), -40))))
            prediction = int(probability >= 0.5)
            correct = prediction == labels[bad_id]
            probe_rows.append(
                {
                    "robust_state_id": bad_id,
                    "probe_type": "GROUP_LOGISTIC_C10",
                    "feature_set": name,
                    "feature_count": len(indices),
                    "selection_data": "fixed existing group; fit on fold training source groups only",
                    "train_oriented_AUROC": "",
                    "train_balanced_accuracy": "",
                    "heldout_probability": probability,
                    "heldout_score": logit,
                    "decision_threshold": 0.5,
                    "heldout_prediction": prediction,
                    "heldout_correct": correct,
                    "rank": "",
                }
            )
            probe_summary[bad_id].append(correct)
    write_csv(HERE / "local_separability_probes.csv", probe_rows)

    sensitivity_rows = []
    sensitivity_evidence = {}
    for bad_id in robust_ids:
        gradient = np.mean(np.abs(robust_gate_gradient[bad_id]), axis=0)
        rows = []
        for group, indices_list in semantic_indices.items():
            if group == "all_214D":
                continue
            indices = np.asarray(indices_list, int)
            rows.append(
                {
                    "robust_state_id": bad_id,
                    "feature_group": group,
                    "dimension_count": len(indices),
                    "mean_abs_logit_gradient_normalized_input": float(np.mean(gradient[indices])),
                    "L2_logit_gradient_normalized_input": float(np.linalg.norm(gradient[indices])),
                    "fraction_total_abs_gradient": float(np.sum(gradient[indices]) / max(np.sum(gradient), 1e-12)),
                    "local_opposite_over_same_ratio": state_evidence[bad_id]["collision_by_semantic"][group],
                    "top_sensitive_dimensions": "|".join(
                        f"{int(index)}:{dimension_name[int(index)]}"
                        for index in indices[np.argsort(-gradient[indices])[: min(5, len(indices))]]
                    ),
                }
            )
        rows.sort(key=lambda row: -row["fraction_total_abs_gradient"])
        for rank, row in enumerate(rows, start=1):
            row["sensitivity_rank"] = rank
        sensitivity_rows.extend(rows)
        separation_groups = [group for group in semantic_indices if group != "all_214D"]
        best_separating = max(
            separation_groups,
            key=lambda group: state_evidence[bad_id]["collision_by_semantic"][group],
        )
        rank_of_best = next(row["sensitivity_rank"] for row in rows if row["feature_group"] == best_separating)
        sensitivity_evidence[bad_id] = {
            "top_sensitive_group": rows[0]["feature_group"],
            "best_separating_group": best_separating,
            "sensitivity_rank_of_best_separating_group": rank_of_best,
            "attention_mismatch": rank_of_best >= 4,
        }
    write_csv(HERE / "gate_sensitivity.csv", sensitivity_rows)

    classifications = []
    for bad_id in robust_ids:
        evidence = state_evidence[bad_id]
        omitted_useful = any(
            row["robust_state_id"] == bad_id and row["potentially_useful_omitted_distinction"] is True
            for row in snapshot_rows
        )
        probe_fraction = float(np.mean(probe_summary[bad_id]))
        local_signal = (
            evidence["ratio"] >= 1.20
            or evidence["own_source_opposite_over_same_ratio"] >= 1.20
            or evidence["knn5_true_fraction"] >= 0.60
            or probe_fraction >= 0.50
        )
        # Three local peers spanning two independent training groups is treated as
        # minimally adequate local coverage; anything less remains source-sparse.
        sparse_support = evidence["same_support_count"] < 3 or len(evidence["same_support_groups"]) < 2
        if omitted_useful:
            classification = "POSSIBLE_MISSING_STATE_INFORMATION"
        elif not local_signal:
            classification = "REPRESENTATION_COLLISION"
        elif sparse_support:
            classification = "INFORMATION_PRESENT_BUT_UNDERCOVERED"
        else:
            classification = "INFORMATION_PRESENT_BUT_NOT_LEARNED"
        classifications.append(
            {
                "state_id": bad_id,
                "oracle_label": labels[bad_id],
                "source_group": stable[bad_id]["source_group"],
                "nearest_same_label_state": evidence["nearest_same"],
                "nearest_same_label_distance": evidence["d_same"],
                "nearest_opposite_label_state": evidence["nearest_opposite"],
                "nearest_opposite_label_distance": evidence["d_opposite"],
                "opposite_over_same_ratio": evidence["ratio"],
                "own_source_nearest_same_label_state": evidence["own_source_nearest_same"],
                "own_source_nearest_same_label_distance": evidence["own_source_nearest_same_distance"],
                "own_source_nearest_opposite_label_state": evidence["own_source_nearest_opposite"],
                "own_source_nearest_opposite_label_distance": evidence["own_source_nearest_opposite_distance"],
                "own_source_opposite_over_same_ratio": evidence["own_source_opposite_over_same_ratio"],
                "knn5_true_label_fraction": evidence["knn5_true_fraction"],
                "same_label_support_inside_opposite_radius": evidence["same_support_count"],
                "same_label_support_source_group_count": len(evidence["same_support_groups"]),
                "same_label_support_source_groups": "|".join(evidence["same_support_groups"]),
                "semantic_group_probe_correct_fraction": probe_fraction,
                "top_sensitive_group": sensitivity_evidence[bad_id]["top_sensitive_group"],
                "best_locally_separating_group": sensitivity_evidence[bad_id]["best_separating_group"],
                "sensitivity_rank_of_best_separating_group": sensitivity_evidence[bad_id]["sensitivity_rank_of_best_separating_group"],
                "attention_mismatch": sensitivity_evidence[bad_id]["attention_mismatch"],
                "omitted_snapshot_field_distinguishes": omitted_useful,
                "failure_classification": classification,
                "classification_rule": "missing if omitted field differs; collision if no local signal; undercovered if signal but <3 supporting states or <2 source groups; otherwise present-not-learned",
            }
        )
    write_csv(HERE / "per_state_failure_classification.csv", classifications)

    failure_counts = Counter(row["failure_classification"] for row in classifications)
    if len(failure_counts) > 1:
        overall = "MIXED_LOCAL_FAILURE_MODES"
    elif failure_counts["REPRESENTATION_COLLISION"]:
        overall = "LOCAL_FEATURE_INSUFFICIENCY"
    elif failure_counts["INFORMATION_PRESENT_BUT_UNDERCOVERED"]:
        overall = "LOCAL_STABLE_DATA_COVERAGE_LIMIT"
    elif failure_counts["INFORMATION_PRESENT_BUT_NOT_LEARNED"]:
        overall = "FEATURES_PRESENT_BUT_NOT_LEARNED"
    elif failure_counts["POSSIBLE_MISSING_STATE_INFORMATION"]:
        overall = "LOCAL_FEATURE_INSUFFICIENCY"
    else:
        raise RuntimeError(failure_counts)

    semantic_probe_summary = []
    for name in [group for group in semantic_indices if group != "all_214D"] + ["TOP8_UNIVARIATE_TRAIN_ONLY"]:
        rows = [row for row in probe_rows if row["probe_type"] == "GROUP_LOGISTIC_C10" and row["feature_set"] == name]
        semantic_probe_summary.append((name, sum(row["heldout_correct"] is True for row in rows), len(rows)))
    semantic_probe_summary.sort(key=lambda item: (-item[1], item[0]))
    best_probe_text = ", ".join(f"{name} {correct}/{total}" for name, correct, total in semantic_probe_summary)
    classification_table = "\n".join(
        f"| `{row['state_id']}` | {row['oracle_label']} | {row['nearest_same_label_distance']:.3f} | {row['nearest_opposite_label_distance']:.3f} | {row['opposite_over_same_ratio']:.2f} | {row['own_source_opposite_over_same_ratio']:.2f} | {row['knn5_true_label_fraction']:.2f} | {row['same_label_support_source_group_count']} | {row['failure_classification']} |"
        for row in classifications
    )

    omitted_differences = [row for row in snapshot_rows if row["potentially_useful_omitted_distinction"] is True]
    smallest_next = (
        "Run one fixed, training-group-only low-dimensional gate probe using the best existing semantic block, with the same six LOGO folds; do not add features or collect states."
        if overall == "FEATURES_PRESENT_BUT_NOT_LEARNED"
        else "Audit additional pre-existing stable states from independent source groups for the same local patterns; do not target or collect new states in this step."
        if overall == "LOCAL_STABLE_DATA_COVERAGE_LIMIT"
        else "Using only the saved 64 Flow variants, compare sample-distribution overlap for the representation-collision states against their nearest opposite-label neighbors; do not retrain or collect data."
        if overall == "MIXED_LOCAL_FAILURE_MODES"
        else "Verify whether any already-recorded deployment-time memory outside the restorable snapshot exists before proposing feature changes."
    )

    runtime = {
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_s": time.monotonic() - started,
        "CPU_worker_limit": int(os.environ.get("LOCAL_AUDIT_CPU_LIMIT", "4")),
        "GPU_used": False,
        "GPU_shards": 0,
        "new_states": 0,
        "new_oracle_rollouts": 0,
        "gate_training_runs": 0,
        "diagnostic_logistic_fits": len([row for row in probe_rows if row["probe_type"] == "GROUP_LOGISTIC_C10"]),
        "python": sys.version,
        "platform": platform.platform(),
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    after = {name: sha(path) for name, path in source_paths.items()}
    sanity = {
        "passed": before == after
        and robust_ids == reproduced
        and len(robust_ids) == 6
        and not heldout_leakage
        and not omitted_differences,
        "prior_artifacts_unchanged": before == after,
        "robust_mistakes_exactly_reproduced": robust_ids == reproduced,
        "robust_mistake_count": len(robust_ids),
        "heldout_source_group_leakage": heldout_leakage,
        "ambiguous_states_used": False,
        "new_states_generated": 0,
        "new_oracle_rollouts": 0,
        "gate_retrained": False,
        "G_phi_trained": False,
        "feature_representation_modified": False,
        "oracle_modified": False,
        "closed_loop_run": False,
        "omitted_restorable_event_latches_that_differ": len(omitted_differences),
        "full_snapshot_field_count": len(snapshot_fields),
        "feature_dimension": 214,
        "overall_classification": overall,
    }
    write_json(HERE / "sanity_checks.json", sanity)
    if not sanity["passed"]:
        raise RuntimeError(sanity)

    report = f"""# Hard stable local feature audit

## Decision

**{overall}**

The exact six pre-defined robust mistakes were reproduced from the three-seed source-group-held-out predictions. No model was retrained, no state or rollout was generated, and the 214-D representation and oracle were unchanged.

## State-level evidence

| State | Oracle | cross-group d_same | cross-group d_opp | cross ratio | own-group ratio | k=5 true fraction | support groups | Classification |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{classification_table}

Distances use each state's original cross-validation fold normalization and only that fold's training source groups. The held-out source group is excluded from every primary neighbor, kNN, support, and probe calculation.

## Existing information and full snapshot

- Diagnostic group/top-eight logistic probes, selected and fitted only on training groups: {best_probe_text}.
- Full augmented snapshots contain {len(snapshot_fields)} stored fields. Every varying physical, monitor, timer, latch, and ordered error-history field is already encoded in the 214-D input.
- The only omitted stored fields are four first-terminal-event latches plus `done`; none differs in any robust-mistake/nearest-opposite comparison, so they provide no missing decision information.
- State classifications: {dict(failure_counts)}.

## Frozen gate sensitivity

Sensitivity is the input gradient of the frozen confidence-aware gate logit with respect to its normalized 214-D input, averaged over the state's 64 Flow variants. It is diagnostic only. Per-state comparisons between the most sensitive and most locally separating groups are recorded in `gate_sensitivity.csv` and `per_state_failure_classification.csv`.

The frozen gate classifies {frozen_gate_correct}/{len(robust_ids)} of these states correctly at its validation-selected threshold {frozen_gate_threshold:.6f}, whereas the strict source-group-held-out gates fail under all three seeds. Because the frozen checkpoint's original train/validation split contains some of these groups, its gradients diagnose learned attention but are not independent OOF attribution.

## Smallest next experiment

{smallest_next}

No controller redesign is proposed by this audit.
"""
    (HERE / "local_feature_audit_report.md").write_text(report)

    required = [
        "local_feature_audit_report.md",
        "robust_mistakes.csv",
        "nearest_neighbor_comparisons.csv",
        "feature_group_distances.csv",
        "augmented_state_field_comparison.csv",
        "local_knn_geometry.csv",
        "local_separability_probes.csv",
        "gate_sensitivity.csv",
        "per_state_failure_classification.csv",
        "sanity_checks.json",
        "runtime_statistics.json",
    ]
    manifest = {
        "study": "HARD_STABLE_LOCAL_FEATURE_AUDIT",
        "classification": overall,
        "robust_mistake_count": len(robust_ids),
        "new_states": 0,
        "new_oracle_rollouts": 0,
        "gate_training_runs": 0,
        "source_artifact_sha256": before,
        "files_sha256": {name: sha(HERE / name) for name in required},
    }
    write_json(HERE / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "classification": overall,
                "failure_counts": dict(failure_counts),
                "robust_states": robust_ids,
                "best_probe_summary": semantic_probe_summary,
                "runtime_s": runtime["wall_s"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
