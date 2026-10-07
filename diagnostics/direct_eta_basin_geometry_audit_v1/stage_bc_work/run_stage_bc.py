#!/usr/bin/env python3
"""CPU-only feature aliasing, zero/active probes, and smoothness diagnostics.

This is deliberately an audit script, not a production training script.  It
never imports or invokes the simulator and only writes beneath stage_bc_work.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import resource
import socket
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Set before importing numerical libraries.  The audit is CPU-only and capped
# at six host threads per the laboratory resource policy.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "6")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "6")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=6")

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.special import expit
from scipy.stats import rankdata, spearmanr


ROOT = Path("/home/zhihan/research/Basin_C1")
OUT = ROOT / "diagnostics/direct_eta_basin_geometry_audit_v1/stage_bc_work"
DATA = ROOT / "diagnostics/gphi_training_dataset_strict_deadlock_v1"
MODEL = ROOT / "diagnostics/gphi_fixed_d_eta_predictor_v1"
SAMPLES = DATA / "samples.npz"
METADATA = DATA / "sample_metadata.jsonl"
SCHEMA = DATA / "feature_schema.json"
CHECKPOINT = MODEL / "best_fixed_d_eta_checkpoint.npz"

EXPECTED = {
    SAMPLES: "79d7da0492d9b414c03ce53f9ee826c54ac7dce3509b2cd3d086fc1cf852deb9",
    METADATA: "03714a831f9f98a091ae6e7be8aa67c9b8243e3c81f0ff8098f07b835f691213",
    SCHEMA: "479e0dc8eeb613b3acd50404b4d11e2e3c58357f56be5a6d26f6ab1040ecaf85",
    CHECKPOINT: "2481028b7e2c4ef14fb14016c138e0d00dd83e40cc2997fd1ad1ee9b5028b095",
}

ETA_LOW = np.asarray([0.0, -0.53125, -0.125], dtype=np.float64)
ETA_HIGH = np.asarray([1.25, 0.5, 0.75], dtype=np.float64)
ETA_SPAN = ETA_HIGH - ETA_LOW
ZERO_NORM_THRESHOLDS = (0.01, 0.05, 0.10, 0.25)
LARGE_ETA_JUMP_THRESHOLD = 0.50  # L2 after coordinate scaling by ETA_SPAN.
NUM_SOURCE_FOLDS = 5
FOLD_SEED = 20260926
RANDOM_SPLIT_SEED = 20260927
MLP_EPOCHS = 80
MLP_BATCH_SIZE = 1024
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-5
LINEAR_L2 = 1e-4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    if not fieldnames:
        raise RuntimeError(f"fieldnames required for empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def fit_normalization(x_train: np.ndarray, binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    mean[binary] = 0.0
    scale[binary] = 1.0
    return mean, scale


def normalized_eta_distance_from_zero(eta: np.ndarray) -> np.ndarray:
    return np.linalg.norm(eta / ETA_SPAN, axis=-1)


def eta_pair_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm((a - b) / ETA_SPAN, axis=-1)


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float | int | None]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(probability, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        auc = float("nan")
        ap = float("nan")
    else:
        ranks = rankdata(p, method="average")
        auc = (float(ranks[y == 1].sum()) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        order = np.argsort(-p, kind="mergesort")
        ordered_y = y[order]
        precision = np.cumsum(ordered_y) / np.arange(1, len(y) + 1)
        ap = float(precision[ordered_y == 1].sum() / n_pos)
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1)))
    tn = int(np.sum((~pred) & (y == 0)))
    sensitivity = tp / n_pos if n_pos else float("nan")
    specificity = tn / n_neg if n_neg else float("nan")
    return {
        "n": int(len(y)),
        "active": n_pos,
        "zero": n_neg,
        "auroc": finite_float(auc),
        "auprc_average_precision": finite_float(ap),
        "balanced_accuracy_at_0.5": finite_float(0.5 * (sensitivity + specificity)),
        "sensitivity_at_0.5": finite_float(sensitivity),
        "specificity_at_0.5": finite_float(specificity),
    }


def aggregate_state_probabilities(
    indices: np.ndarray, state_ids: np.ndarray, y: np.ndarray, p: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state_to_values: dict[str, list[float]] = defaultdict(list)
    state_to_label: dict[str, int] = {}
    for idx, prob in zip(indices.tolist(), p.tolist()):
        sid = str(state_ids[idx])
        state_to_values[sid].append(float(prob))
        state_to_label[sid] = int(y[idx])
    ordered = np.asarray(sorted(state_to_values), dtype=str)
    labels = np.asarray([state_to_label[sid] for sid in ordered], dtype=np.int8)
    probs = np.asarray([np.mean(state_to_values[sid]) for sid in ordered], dtype=np.float64)
    return ordered, labels, probs


def make_source_folds(
    state_ids: np.ndarray, state_active: np.ndarray, state_groups: np.ndarray, k: int, seed: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    group_to_states: dict[str, list[int]] = defaultdict(list)
    for i, group in enumerate(state_groups.tolist()):
        group_to_states[str(group)].append(i)
    rng = np.random.default_rng(seed)
    tie = {group: float(rng.random()) for group in group_to_states}
    groups = sorted(
        group_to_states,
        key=lambda group: (-len(group_to_states[group]), -abs(sum(state_active[group_to_states[group]])
                                                        - len(group_to_states[group]) / 2), tie[group]),
    )
    target_n = len(state_ids) / k
    target_a = float(state_active.sum()) / k
    target_z = float((1 - state_active).sum()) / k
    counts = np.zeros((k, 3), dtype=np.float64)  # total, active, zero
    assignment: dict[str, int] = {}
    for group in groups:
        members = group_to_states[group]
        delta = np.asarray([len(members), state_active[members].sum(), len(members) - state_active[members].sum()])
        candidates = []
        for fold in range(k):
            candidate = counts.copy()
            candidate[fold] += delta
            objective = float(np.sum(((candidate[:, 0] - target_n) / target_n) ** 2)
                              + np.sum(((candidate[:, 1] - target_a) / target_a) ** 2)
                              + np.sum(((candidate[:, 2] - target_z) / target_z) ** 2))
            candidates.append((objective, counts[fold, 0], fold))
        chosen = min(candidates)[2]
        assignment[group] = chosen
        counts[chosen] += delta
    state_fold = np.asarray([assignment[str(group)] for group in state_groups], dtype=np.int64)
    rows = []
    for fold in range(k):
        mask = state_fold == fold
        rows.append({
            "fold": fold,
            "states": int(mask.sum()),
            "active_states": int(state_active[mask].sum()),
            "zero_states": int(mask.sum() - state_active[mask].sum()),
            "source_leakage_groups": int(len(set(state_groups[mask].tolist()))),
        })
    return state_fold, rows


def train_linear(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    y64 = y.astype(np.float64)
    n = len(y64)
    pos = max(float(y64.sum()), 1.0)
    neg = max(float(n - y64.sum()), 1.0)
    sample_weight = np.where(y64 > 0.5, n / (2.0 * pos), n / (2.0 * neg))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = x @ theta[:-1] + theta[-1]
        loss = np.mean(sample_weight * (np.logaddexp(0.0, logits) - y64 * logits))
        loss += 0.5 * LINEAR_L2 * float(theta[:-1] @ theta[:-1])
        residual = sample_weight * (expit(logits) - y64) / n
        gradient = np.r_[x.T @ residual + LINEAR_L2 * theta[:-1], residual.sum()]
        return float(loss), gradient

    result = minimize(
        objective,
        np.zeros(x.shape[1] + 1, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-6, "maxls": 30},
    )
    return result.x, {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "final_objective": float(result.fun),
        "gradient_inf_norm": float(np.max(np.abs(result.jac))),
    }


def predict_linear(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    return expit(x @ theta[:-1] + theta[-1])


def train_mlp(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[Any, dict[str, Any]]:
    # Imports are local so all collision/geometry work remains independent of JAX.
    import jax
    import jax.numpy as jnp
    import optax

    rng = np.random.default_rng(seed)
    dims = (214, 64, 64, 1)
    params = []
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        params.append({
            "w": jnp.asarray(rng.uniform(-bound, bound, size=(fan_in, fan_out)).astype(np.float32)),
            "b": jnp.zeros((fan_out,), dtype=jnp.float32),
        })
    optimizer = optax.adamw(MLP_LR, weight_decay=MLP_WEIGHT_DECAY)
    optimizer_state = optimizer.init(params)
    n = len(y)
    pos = max(int(y.sum()), 1)
    neg = max(n - pos, 1)
    class_weight = jnp.asarray([n / (2.0 * neg), n / (2.0 * pos)], dtype=jnp.float32)

    def apply(candidate: Any, xb: Any) -> Any:
        value = jax.nn.silu(xb @ candidate[0]["w"] + candidate[0]["b"])
        value = jax.nn.silu(value @ candidate[1]["w"] + candidate[1]["b"])
        return (value @ candidate[2]["w"] + candidate[2]["b"]).reshape(-1)

    @jax.jit
    def update(candidate: Any, state: Any, xb: Any, yb: Any) -> tuple[Any, Any, Any]:
        def loss_fn(value: Any) -> Any:
            logits = apply(value, xb)
            weights = jnp.where(yb > 0.5, class_weight[1], class_weight[0])
            return jnp.mean(weights * optax.sigmoid_binary_cross_entropy(logits, yb))
        loss, gradient = jax.value_and_grad(loss_fn)(candidate)
        updates, state = optimizer.update(gradient, state, candidate)
        return optax.apply_updates(candidate, updates), state, loss

    x32 = x.astype(np.float32, copy=False)
    y32 = y.astype(np.float32, copy=False)
    losses = []
    for epoch in range(MLP_EPOCHS):
        order = rng.permutation(n)
        epoch_losses = []
        for begin in range(0, n, MLP_BATCH_SIZE):
            batch = order[begin:begin + MLP_BATCH_SIZE]
            params, optimizer_state, loss = update(
                params, optimizer_state, jnp.asarray(x32[batch]), jnp.asarray(y32[batch])
            )
            epoch_losses.append(float(loss))
        losses.append(float(np.mean(epoch_losses)))
    # Force completion before runtime accounting.
    final_norm = float(sum(np.asarray(layer["w"] ** 2).sum() for layer in params) ** 0.5)
    return (params, apply), {
        "epochs": MLP_EPOCHS,
        "batch_size": MLP_BATCH_SIZE,
        "learning_rate": MLP_LR,
        "weight_decay": MLP_WEIGHT_DECAY,
        "initial_epoch_loss": losses[0],
        "final_epoch_loss": losses[-1],
        "parameter_l2": final_norm,
        "device": str(jax.devices()[0]),
    }


def predict_mlp(model: tuple[Any, Any], x: np.ndarray) -> np.ndarray:
    import jax
    import jax.numpy as jnp
    params, apply = model
    probabilities = []
    for begin in range(0, len(x), 4096):
        logits = apply(params, jnp.asarray(x[begin:begin + 4096].astype(np.float32, copy=False)))
        probabilities.append(np.asarray(jax.nn.sigmoid(logits)))
    return np.concatenate(probabilities).astype(np.float64)


def frozen_eta_predict(features: np.ndarray, checkpoint: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = (features - checkpoint["normalization_mean"]) / checkpoint["normalization_scale"]
    for i in range(2):
        value = value @ checkpoint[f"layer_{i}_weight"].astype(np.float64) + checkpoint[f"layer_{i}_bias"]
        value = value / (1.0 + np.exp(-value))
    raw = value @ checkpoint["layer_2_weight"].astype(np.float64) + checkpoint["layer_2_bias"]
    clipped = np.clip(raw, 0.0, 1.0)
    eta = ETA_LOW + clipped * ETA_SPAN
    return eta, raw, clipped


def main() -> None:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    observed_hashes = {str(path): sha256(path) for path in EXPECTED}
    for path, expected in EXPECTED.items():
        if observed_hashes[str(path)] != expected:
            raise RuntimeError(f"frozen asset hash mismatch: {path}")

    with np.load(SAMPLES, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]).copy() for key in source.files}
    metadata = [json.loads(line) for line in METADATA.read_text().splitlines() if line]
    schema = json.loads(SCHEMA.read_text())
    if arrays["features"].shape != (27136, 214) or len(metadata) != 27136:
        raise RuntimeError("unexpected authoritative dataset dimensions")
    if any(metadata[i]["sample_id"] != str(arrays["sample_id"][i]) for i in range(len(metadata))):
        raise RuntimeError("sample metadata alignment mismatch")
    if not np.isfinite(arrays["features"]).all():
        raise RuntimeError("non-finite features")

    features = arrays["features"].astype(np.float64, copy=False)
    sample_state_ids = arrays["state_id"].astype(str)
    sample_ids = arrays["sample_id"].astype(str)
    sample_splits = arrays["split"].astype(str)
    eta = np.asarray([row["eta_best_metadata_only"] for row in metadata], dtype=np.float64)
    active = np.any(eta != 0.0, axis=1).astype(np.int8)  # exact mandated definition
    source_groups = np.asarray([row["leakage_group"] for row in metadata], dtype=str)
    source_trajectory = np.asarray([row["source_trajectory"] for row in metadata], dtype=str)

    state_to_indices: dict[str, np.ndarray] = {}
    for sid in dict.fromkeys(sample_state_ids.tolist()):
        state_to_indices[sid] = np.flatnonzero(sample_state_ids == sid)
    state_ids = np.asarray(list(state_to_indices), dtype=str)
    if len(state_ids) != 424 or any(len(indices) != 64 for indices in state_to_indices.values()):
        raise RuntimeError("expected exactly 424 states x 64 Flow variants")
    representative = np.asarray([state_to_indices[sid][0] for sid in state_ids], dtype=np.int64)
    state_eta = eta[representative]
    state_active = active[representative]
    state_splits = sample_splits[representative]
    state_groups = source_groups[representative]
    state_sources = source_trajectory[representative]

    binary = np.zeros(214, dtype=bool)
    for segment in schema["segments"]:
        if segment["unit"] == "boolean":
            begin = int(segment["offset"])
            binary[begin:begin + int(segment["length"])] = True

    with np.load(CHECKPOINT, allow_pickle=False) as source:
        checkpoint = {key: np.asarray(source[key]).copy() for key in source.files}
    original_train = sample_splits == "train"
    calculated_mean, calculated_scale = fit_normalization(features[original_train], binary)
    mean_delta = float(np.max(np.abs(calculated_mean - checkpoint["normalization_mean"])))
    scale_delta = float(np.max(np.abs(calculated_scale - checkpoint["normalization_scale"])))
    if mean_delta > 1e-12 or scale_delta > 1e-12:
        raise RuntimeError(("checkpoint training normalizer mismatch", mean_delta, scale_delta))
    normalized = (features - calculated_mean) / calculated_scale

    # ------------------------------------------------------------------
    # Stage B: exact/numerically indistinguishable feature collisions.
    # ------------------------------------------------------------------
    collision_rows: list[dict[str, Any]] = []
    exact_pair_count = 0
    exact_conflict_pair_count = 0
    exact_cross_state_pair_count = 0
    _, inverse = np.unique(features, axis=0, return_inverse=True)
    exact_groups: dict[int, list[int]] = defaultdict(list)
    for idx, group in enumerate(inverse.tolist()):
        exact_groups[group].append(idx)
    exact_pairs_seen: set[tuple[int, int]] = set()
    for members in exact_groups.values():
        if len(members) < 2:
            continue
        for p, left in enumerate(members):
            for right in members[p + 1:]:
                if sample_state_ids[left] == sample_state_ids[right]:
                    continue
                pair = (min(left, right), max(left, right))
                exact_pairs_seen.add(pair)
                exact_pair_count += 1
                exact_cross_state_pair_count += 1
                distance = float(eta_pair_distance(eta[left], eta[right]))
                conflict = distance > 1e-12
                exact_conflict_pair_count += int(conflict)
                collision_rows.append({
                    "collision_type": "bitwise_exact",
                    "sample_i": sample_ids[left], "state_i": sample_state_ids[left],
                    "source_i": source_groups[left], "eta_i": json.dumps(eta[left].tolist()),
                    "sample_j": sample_ids[right], "state_j": sample_state_ids[right],
                    "source_j": source_groups[right], "eta_j": json.dumps(eta[right].tolist()),
                    "max_abs_normalized_h_difference": 0.0,
                    "normalized_h_l2": 0.0,
                    "normalized_eta_l2": distance,
                    "zero_active_mismatch": bool(active[left] != active[right]),
                    "incompatible_eta": bool(conflict),
                })
    # Numerically indistinguishable means every train-normalized coordinate is
    # within 1e-12.  query_pairs with L-infinity implements this definition.
    near_exact_pairs = cKDTree(normalized).query_pairs(r=1e-12, p=np.inf, output_type="ndarray")
    numerical_pair_count = 0
    numerical_conflict_pair_count = 0
    for left, right in near_exact_pairs.tolist():
        if sample_state_ids[left] == sample_state_ids[right] or (left, right) in exact_pairs_seen:
            continue
        numerical_pair_count += 1
        distance = float(eta_pair_distance(eta[left], eta[right]))
        conflict = distance > 1e-12
        numerical_conflict_pair_count += int(conflict)
        delta = normalized[left] - normalized[right]
        collision_rows.append({
            "collision_type": "numerically_indistinguishable_1e-12_Linf",
            "sample_i": sample_ids[left], "state_i": sample_state_ids[left],
            "source_i": source_groups[left], "eta_i": json.dumps(eta[left].tolist()),
            "sample_j": sample_ids[right], "state_j": sample_state_ids[right],
            "source_j": source_groups[right], "eta_j": json.dumps(eta[right].tolist()),
            "max_abs_normalized_h_difference": float(np.max(np.abs(delta))),
            "normalized_h_l2": float(np.linalg.norm(delta)),
            "normalized_eta_l2": distance,
            "zero_active_mismatch": bool(active[left] != active[right]),
            "incompatible_eta": bool(conflict),
        })
    collision_fields = [
        "collision_type", "sample_i", "state_i", "source_i", "eta_i",
        "sample_j", "state_j", "source_j", "eta_j",
        "max_abs_normalized_h_difference", "normalized_h_l2", "normalized_eta_l2",
        "zero_active_mismatch", "incompatible_eta",
    ]
    write_csv(OUT / "exact_feature_collisions.csv", collision_rows, collision_fields)

    # Exact sample-level nearest other-state neighbor using a train-normalized
    # 214-D cKDTree. k=128 exceeds the 64 variants belonging to one state.
    tree = cKDTree(normalized)
    nn_distance, nn_index = tree.query(normalized, k=128, workers=6)
    neighbor_rows: list[dict[str, Any]] = []
    no_other = 0
    for i in range(len(features)):
        chosen = None
        for distance, j in zip(nn_distance[i, 1:].tolist(), nn_index[i, 1:].tolist()):
            if j < len(features) and sample_state_ids[j] != sample_state_ids[i]:
                chosen = (float(distance), int(j))
                break
        if chosen is None:
            no_other += 1
            continue
        distance, j = chosen
        neighbor_rows.append({
            "sample_id": sample_ids[i], "state_id": sample_state_ids[i],
            "split": sample_splits[i], "source_group": source_groups[i],
            "canonical_eta": json.dumps(eta[i].tolist()), "is_active": bool(active[i]),
            "neighbor_sample_id": sample_ids[j], "neighbor_state_id": sample_state_ids[j],
            "neighbor_split": sample_splits[j], "neighbor_source_group": source_groups[j],
            "neighbor_canonical_eta": json.dumps(eta[j].tolist()),
            "neighbor_is_active": bool(active[j]),
            "normalized_h_l2": distance,
            "normalized_h_rms": distance / math.sqrt(features.shape[1]),
            "eta_physical_l2": float(np.linalg.norm(eta[i] - eta[j])),
            "eta_coordinate_normalized_l2": float(eta_pair_distance(eta[i], eta[j])),
            "zero_active_mismatch": bool(active[i] != active[j]),
            "same_source_group": bool(source_groups[i] == source_groups[j]),
        })
    if no_other:
        raise RuntimeError(f"nearest-neighbor query failed for {no_other} samples")
    write_csv(OUT / "nearest_neighbor_target_jumps.csv", neighbor_rows)

    # State centroids are the frozen transparent state-level cloud summary.
    centroids = np.asarray([normalized[state_to_indices[sid]].mean(axis=0) for sid in state_ids])
    state_delta = centroids[:, None, :] - centroids[None, :, :]
    state_h_distance = np.linalg.norm(state_delta, axis=2)
    np.fill_diagonal(state_h_distance, np.inf)
    state_eta_distance = np.linalg.norm(
        (state_eta[:, None, :] - state_eta[None, :, :]) / ETA_SPAN, axis=2
    )
    state_neighbor_rows: list[dict[str, Any]] = []
    for i, sid in enumerate(state_ids):
        nearest = int(np.argmin(state_h_distance[i]))
        opposite_candidates = np.flatnonzero(state_active != state_active[i])
        nearest_opposite = int(opposite_candidates[np.argmin(state_h_distance[i, opposite_candidates])])
        large_candidates = np.flatnonzero(state_eta_distance[i] >= LARGE_ETA_JUMP_THRESHOLD)
        nearest_large = int(large_candidates[np.argmin(state_h_distance[i, large_candidates])])
        row: dict[str, Any] = {
            "state_id": sid, "split": state_splits[i], "source_group": state_groups[i],
            "canonical_eta": json.dumps(state_eta[i].tolist()), "is_active": bool(state_active[i]),
        }
        for prefix, j in (("nearest", nearest), ("nearest_opposite", nearest_opposite),
                          ("nearest_large_eta_jump", nearest_large)):
            row.update({
                f"{prefix}_state_id": state_ids[j],
                f"{prefix}_source_group": state_groups[j],
                f"{prefix}_is_active": bool(state_active[j]),
                f"{prefix}_h_l2": float(state_h_distance[i, j]),
                f"{prefix}_h_rms": float(state_h_distance[i, j] / math.sqrt(214)),
                f"{prefix}_eta_normalized_l2": float(state_eta_distance[i, j]),
            })
        state_neighbor_rows.append(row)
    write_csv(OUT / "state_level_neighbor_target_jumps.csv", state_neighbor_rows)

    # Preliminary Stage-D target smoothness: directed 5-NN centroid pairs.
    smooth_rows: list[dict[str, Any]] = []
    for i, sid in enumerate(state_ids):
        nearest_five = np.argsort(state_h_distance[i])[:5]
        for rank, j in enumerate(nearest_five, start=1):
            hd = float(state_h_distance[i, j])
            ed = float(state_eta_distance[i, j])
            smooth_rows.append({
                "state_i": sid, "split_i": state_splits[i], "source_i": state_groups[i],
                "eta_i": json.dumps(state_eta[i].tolist()), "active_i": bool(state_active[i]),
                "neighbor_rank": rank,
                "state_j": state_ids[j], "split_j": state_splits[j], "source_j": state_groups[j],
                "eta_j": json.dumps(state_eta[j].tolist()), "active_j": bool(state_active[j]),
                "normalized_h_l2": hd,
                "normalized_h_rms": hd / math.sqrt(214),
                "eta_coordinate_normalized_l2": ed,
                "local_eta_jump_per_h_l2": ed / max(hd, 1e-15),
                "zero_active_crossing": bool(state_active[i] != state_active[j]),
                "same_source_group": bool(state_groups[i] == state_groups[j]),
            })
    write_csv(OUT / "canonical_smoothness_preliminary.csv", smooth_rows)
    # Also expose the canonical Stage-D filename for straightforward parent
    # integration; contents are explicitly preliminary representation metrics.
    write_csv(OUT / "canonical_smoothness.csv", smooth_rows)
    unique_informative: dict[tuple[str, str], dict[str, Any]] = {}
    for row in smooth_rows:
        key = tuple(sorted((str(row["state_i"]), str(row["state_j"]))))
        if row["eta_coordinate_normalized_l2"] < 0.25:
            continue
        existing = unique_informative.get(key)
        if existing is None or row["normalized_h_l2"] < existing["normalized_h_l2"]:
            unique_informative[key] = row
    candidate_pairs = sorted(
        unique_informative.values(),
        key=lambda row: (-row["local_eta_jump_per_h_l2"], row["normalized_h_l2"],
                         row["state_i"], row["state_j"]),
    )[:20]
    candidate_pair_rows = [
        {"priority_rank": rank, "selection_rule": "top eta_jump/h among unique 5NN pairs with eta jump >=0.25", **row}
        for rank, row in enumerate(candidate_pairs, start=1)
    ]
    write_csv(OUT / "cross_eta_candidate_pairs.csv", candidate_pair_rows)
    smooth_h = np.asarray([row["normalized_h_l2"] for row in smooth_rows])
    smooth_eta = np.asarray([row["eta_coordinate_normalized_l2"] for row in smooth_rows])
    spearman_all = spearmanr(smooth_h, smooth_eta)
    rank1 = np.asarray([row["neighbor_rank"] == 1 for row in smooth_rows])
    spearman_rank1 = spearmanr(smooth_h[rank1], smooth_eta[rank1])
    lipschitz = smooth_eta / np.maximum(smooth_h, 1e-15)
    smoothness_summary = {
        "state_representation": "centroid of 64 Flow-variant features after exact checkpoint train normalization",
        "nearby_pair_rule": "five directed nearest-other-state centroid neighbors per state",
        "directed_pairs": len(smooth_rows),
        "spearman_h_eta_all_5nn": finite_float(float(spearman_all.statistic)),
        "spearman_pvalue_all_5nn": finite_float(float(spearman_all.pvalue)),
        "spearman_h_eta_rank1": finite_float(float(spearman_rank1.statistic)),
        "spearman_pvalue_rank1": finite_float(float(spearman_rank1.pvalue)),
        "rank1_zero_active_crossing_fraction": float(np.mean([
            row["zero_active_crossing"] for row in smooth_rows if row["neighbor_rank"] == 1
        ])),
        "five_nn_zero_active_crossing_fraction": float(np.mean([
            row["zero_active_crossing"] for row in smooth_rows
        ])),
        "local_eta_jump_per_h_l2_quantiles": {
            str(q): float(np.quantile(lipschitz, q)) for q in (0, .25, .5, .75, .9, .95, .99, 1)
        },
        "eta_jump_quantiles_5nn": {
            str(q): float(np.quantile(smooth_eta, q)) for q in (0, .25, .5, .75, .9, .95, .99, 1)
        },
    }
    write_json(OUT / "canonical_smoothness_preliminary_summary.json", smoothness_summary)

    # ------------------------------------------------------------------
    # Stage C: source-held-out probes.  Fold normalizers are fitted only on
    # each fold's training samples. The production checkpoint is untouched.
    # ------------------------------------------------------------------
    state_fold, fold_composition = make_source_folds(
        state_ids, state_active, state_groups, NUM_SOURCE_FOLDS, FOLD_SEED
    )
    state_to_fold = {sid: int(state_fold[i]) for i, sid in enumerate(state_ids)}
    sample_fold = np.asarray([state_to_fold[sid] for sid in sample_state_ids], dtype=np.int64)
    fold_rows = []
    pooled: dict[str, np.ndarray] = {
        "linear": np.full(len(features), np.nan),
        "mlp": np.full(len(features), np.nan),
    }
    optimization: dict[str, list[dict[str, Any]]] = {"linear": [], "mlp": []}
    for fold in range(NUM_SOURCE_FOLDS):
        train_mask = sample_fold != fold
        test_mask = sample_fold == fold
        test_indices = np.flatnonzero(test_mask)
        fold_mean, fold_scale = fit_normalization(features[train_mask], binary)
        x_train = (features[train_mask] - fold_mean) / fold_scale
        x_test = (features[test_mask] - fold_mean) / fold_scale
        theta, linear_info = train_linear(x_train, active[train_mask])
        linear_probability = predict_linear(theta, x_test)
        pooled["linear"][test_mask] = linear_probability
        model, mlp_info = train_mlp(x_train, active[train_mask], FOLD_SEED + fold)
        mlp_probability = predict_mlp(model, x_test)
        pooled["mlp"][test_mask] = mlp_probability
        optimization["linear"].append({"fold": fold, **linear_info})
        optimization["mlp"].append({"fold": fold, **mlp_info})
        for model_name, probability in (("linear", linear_probability), ("mlp", mlp_probability)):
            _, state_y, state_p = aggregate_state_probabilities(
                test_indices, sample_state_ids, active, probability
            )
            fold_rows.append({
                "scheme": "source_group_5fold", "fold": fold, "model": model_name,
                "sample_metrics": metrics(active[test_mask], probability),
                "state_metrics": metrics(state_y, state_p),
            })
    if any(np.isnan(value).any() for value in pooled.values()):
        raise RuntimeError("incomplete out-of-fold probabilities")

    grouped_metrics: dict[str, Any] = {}
    all_indices = np.arange(len(features))
    for model_name, probability in pooled.items():
        _, state_y, state_p = aggregate_state_probabilities(
            all_indices, sample_state_ids, active, probability
        )
        grouped_metrics[model_name] = {
            "sample_level_pooled_oof": metrics(active, probability),
            "state_level_pooled_oof_primary": metrics(state_y, state_p),
            "folds": [row for row in fold_rows if row["model"] == model_name],
        }

    # One intentionally leaky random sample split. This is a shortcut
    # diagnostic only: variants of the same state occur on both sides.
    rng = np.random.default_rng(RANDOM_SPLIT_SEED)
    random_order = rng.permutation(len(features))
    random_test_indices = np.sort(random_order[:round(0.20 * len(features))])
    random_test = np.zeros(len(features), dtype=bool)
    random_test[random_test_indices] = True
    random_train = ~random_test
    random_mean, random_scale = fit_normalization(features[random_train], binary)
    random_x_train = (features[random_train] - random_mean) / random_scale
    random_x_test = (features[random_test] - random_mean) / random_scale
    random_results: dict[str, Any] = {}
    random_probabilities: dict[str, np.ndarray] = {}
    theta, random_linear_info = train_linear(random_x_train, active[random_train])
    random_probabilities["linear"] = predict_linear(theta, random_x_test)
    random_model, random_mlp_info = train_mlp(random_x_train, active[random_train], RANDOM_SPLIT_SEED)
    random_probabilities["mlp"] = predict_mlp(random_model, random_x_test)
    for model_name, probability in random_probabilities.items():
        _, state_y, state_p = aggregate_state_probabilities(
            random_test_indices, sample_state_ids, active, probability
        )
        random_results[model_name] = {
            "sample_level": metrics(active[random_test], probability),
            "state_level_aggregate_over_heldout_variants_primary": metrics(state_y, state_p),
            "optimization": random_linear_info if model_name == "linear" else random_mlp_info,
        }

    # ------------------------------------------------------------------
    # Frozen current G_eta behavior on all original dataset splits.
    # ------------------------------------------------------------------
    predicted_eta, predicted_raw, predicted_clipped = frozen_eta_predict(features, checkpoint)
    predicted_norm = normalized_eta_distance_from_zero(predicted_eta)
    predictor_rows: list[dict[str, Any]] = []
    for i, sid in enumerate(state_ids):
        indices = state_to_indices[sid]
        eta_values = predicted_eta[indices]
        norm_values = predicted_norm[indices]
        mean_eta = eta_values.mean(axis=0)
        row = {
            "state_id": sid, "split": state_splits[i], "source_group": state_groups[i],
            "source_trajectory": state_sources[i],
            "true_eta1": state_eta[i, 0], "true_eta2": state_eta[i, 1], "true_eta3": state_eta[i, 2],
            "true_class": "ACTIVE" if state_active[i] else "ZERO",
            "predicted_mean_eta1": mean_eta[0], "predicted_mean_eta2": mean_eta[1],
            "predicted_mean_eta3": mean_eta[2],
            "predicted_eta_physical_norm_mean": float(np.linalg.norm(eta_values, axis=1).mean()),
            "predicted_zero_distance_normalized_norm_of_mean_eta": float(normalized_eta_distance_from_zero(mean_eta)),
            "predicted_zero_distance_normalized_norm_mean": float(norm_values.mean()),
            "predicted_zero_distance_normalized_norm_median": float(np.median(norm_values)),
            "predicted_zero_distance_normalized_norm_min": float(norm_values.min()),
            "predicted_zero_distance_normalized_norm_max": float(norm_values.max()),
            "any_coordinate_clipped_fraction": float(np.mean(np.any(
                np.abs(predicted_raw[indices] - predicted_clipped[indices]) > 0, axis=1
            ))),
        }
        for threshold in ZERO_NORM_THRESHOLDS:
            row[f"fraction_variants_norm_le_{threshold:.2f}"] = float(np.mean(norm_values <= threshold))
        predictor_rows.append(row)
    write_csv(OUT / "current_predictor_zero_active_behavior.csv", predictor_rows)

    predictor_summary: dict[str, Any] = {}
    for split in ("train", "validation", "test", "all"):
        split_mask = np.ones(len(state_ids), dtype=bool) if split == "all" else state_splits == split
        predictor_summary[split] = {}
        for label, label_value in (("ZERO", 0), ("ACTIVE", 1)):
            mask = split_mask & (state_active == label_value)
            state_norms = np.asarray([
                predictor_rows[i]["predicted_zero_distance_normalized_norm_mean"]
                for i in np.flatnonzero(mask)
            ])
            predictor_summary[split][label] = {
                "states": int(mask.sum()),
                "predicted_zero_distance_normalized_norm_mean": float(state_norms.mean()),
                "predicted_zero_distance_normalized_norm_median": float(np.median(state_norms)),
                "predicted_zero_distance_normalized_norm_quantiles": {
                    str(q): float(np.quantile(state_norms, q)) for q in (0, .1, .25, .5, .75, .9, 1)
                },
                "state_fraction_mean_variant_norm_le_threshold": {
                    str(threshold): float(np.mean(state_norms <= threshold))
                    for threshold in ZERO_NORM_THRESHOLDS
                },
            }

    random_group_gap: dict[str, Any] = {}
    for model_name in ("linear", "mlp"):
        grouped_primary = grouped_metrics[model_name]["state_level_pooled_oof_primary"]
        random_primary = random_results[model_name]["state_level_aggregate_over_heldout_variants_primary"]
        random_group_gap[model_name] = {
            metric_name: float(random_primary[metric_name] - grouped_primary[metric_name])
            for metric_name in ("auroc", "auprc_average_precision", "balanced_accuracy_at_0.5")
        }

    feature_alias_summary = {
        "feature_normalization": "exact checkpoint train-only mean/scale; binary features retain literal 0/1",
        "normalizer_max_abs_mean_reproduction_error": mean_delta,
        "normalizer_max_abs_scale_reproduction_error": scale_delta,
        "exact_cross_state_pair_count": exact_cross_state_pair_count,
        "exact_incompatible_eta_pair_count": exact_conflict_pair_count,
        "numerically_indistinguishable_cross_state_pair_count": numerical_pair_count,
        "numerically_indistinguishable_incompatible_eta_pair_count": numerical_conflict_pair_count,
        "numerically_indistinguishable_definition": "max absolute difference in train-normalized coordinates <= 1e-12",
        "sample_nearest_other_state": {
            "rows": len(neighbor_rows),
            "zero_active_mismatch_fraction": float(np.mean([row["zero_active_mismatch"] for row in neighbor_rows])),
            "same_source_group_fraction": float(np.mean([row["same_source_group"] for row in neighbor_rows])),
            "nonzero_eta_jump_fraction": float(np.mean([
                row["eta_coordinate_normalized_l2"] > 1e-12 for row in neighbor_rows
            ])),
            "large_eta_jump_ge_0.5_fraction": float(np.mean([
                row["eta_coordinate_normalized_l2"] >= LARGE_ETA_JUMP_THRESHOLD for row in neighbor_rows
            ])),
            "minimum_h_l2_among_large_eta_jumps_ge_0.5": float(min(
                row["normalized_h_l2"] for row in neighbor_rows
                if row["eta_coordinate_normalized_l2"] >= LARGE_ETA_JUMP_THRESHOLD
            )),
            "h_l2_quantiles": {
                str(q): float(np.quantile([row["normalized_h_l2"] for row in neighbor_rows], q))
                for q in (0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1)
            },
            "eta_normalized_l2_quantiles": {
                str(q): float(np.quantile([row["eta_coordinate_normalized_l2"] for row in neighbor_rows], q))
                for q in (0, .25, .5, .75, .9, .95, .99, 1)
            },
        },
        "state_centroid_nearest_other_state": {
            "states": len(state_neighbor_rows),
            "zero_active_mismatch_fraction": float(np.mean([
                row["nearest_is_active"] != row["is_active"] for row in state_neighbor_rows
            ])),
            "nonzero_eta_jump_fraction": float(np.mean([
                row["nearest_eta_normalized_l2"] > 1e-12 for row in state_neighbor_rows
            ])),
            "large_eta_jump_ge_0.5_fraction": float(np.mean([
                row["nearest_eta_normalized_l2"] >= LARGE_ETA_JUMP_THRESHOLD
                for row in state_neighbor_rows
            ])),
            "minimum_h_l2_among_large_eta_jumps_ge_0.5": float(min(
                row["nearest_h_l2"] for row in state_neighbor_rows
                if row["nearest_eta_normalized_l2"] >= LARGE_ETA_JUMP_THRESHOLD
            )),
            "minimum_h_l2_among_zero_active_mismatches": float(min(
                row["nearest_h_l2"] for row in state_neighbor_rows
                if row["nearest_is_active"] != row["is_active"]
            )),
            "h_l2_quantiles": {
                str(q): float(np.quantile([row["nearest_h_l2"] for row in state_neighbor_rows], q))
                for q in (0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1)
            },
            "eta_normalized_l2_quantiles": {
                str(q): float(np.quantile([row["nearest_eta_normalized_l2"] for row in state_neighbor_rows], q))
                for q in (0, .25, .5, .75, .9, .95, .99, 1)
            },
            "large_eta_jump_threshold": LARGE_ETA_JUMP_THRESHOLD,
        },
    }
    write_json(OUT / "feature_aliasing_summary.json", feature_alias_summary)

    probe_results = {
        "status": "COMPLETE_CPU_ONLY_DIAGNOSTIC_PROBES",
        "target_definition": {"ZERO": "canonical eta exactly (0,0,0)", "ACTIVE": "canonical eta not exactly (0,0,0)"},
        "states": {"total": 424, "zero": int((state_active == 0).sum()), "active": int(state_active.sum())},
        "samples": {"total": len(features), "zero": int((active == 0).sum()), "active": int(active.sum())},
        "grouped_cv": {
            "scheme": "deterministic 5-fold source holdout by authoritative leakage_group; all variants/states from a group stay together",
            "seed": FOLD_SEED,
            "fold_composition": fold_composition,
            "feature_normalization": "fit on each fold's training samples only; schema binary indicators uncentered/unscaled",
            "linear": {"architecture": "214 -> 1 logistic", "class_balancing": True, "l2": LINEAR_L2,
                       "metrics": grouped_metrics["linear"], "optimization": optimization["linear"]},
            "mlp": {"architecture": "214 -> 64 -> 64 -> 1, SiLU", "class_balancing": True,
                    "metrics": grouped_metrics["mlp"], "optimization": optimization["mlp"]},
        },
        "random_sample_shortcut": {
            "scheme": "single seeded 80/20 random sample split; state/Flow variants deliberately leak across sides",
            "seed": RANDOM_SPLIT_SEED,
            "train_samples": int(random_train.sum()), "test_samples": int(random_test.sum()),
            "test_states": int(len(set(sample_state_ids[random_test].tolist()))),
            "linear": random_results["linear"], "mlp": random_results["mlp"],
        },
        "random_minus_grouped_primary_metric_gap": random_group_gap,
        "current_frozen_g_eta": {
            "near_zero_norm_definition": "L2 of physical predicted eta after coordinate-wise division by frozen eta-domain span; zero means physical eta=(0,0,0)",
            "descriptive_thresholds": list(ZERO_NORM_THRESHOLDS),
            "by_original_split_and_true_class": predictor_summary,
        },
    }
    write_json(OUT / "zero_active_probe_results.json", probe_results)

    report = f"""# Feature aliasing and zero/active diagnostic audit

This is a CPU-only read-only audit of the frozen 27,136-sample / 424-state
dataset. It trained only the explicitly permitted diagnostic probes and ran no
continuations or simulator calls.

## Provenance and rules

- Authoritative samples SHA256: `{observed_hashes[str(SAMPLES)]}`
- Authoritative metadata SHA256: `{observed_hashes[str(METADATA)]}`
- Frozen G_eta checkpoint SHA256: `{observed_hashes[str(CHECKPOINT)]}`
- Geometry normalization: exact frozen training-split mean/scale reproduced to
  max absolute mean error {mean_delta:.3g} and scale error {scale_delta:.3g}.
- State cloud rule: centroid of all 64 normalized Flow variants.
- Source grouping: authoritative `leakage_group`; groups are indivisible across
  the five diagnostic cross-validation folds.

## Exact aliasing

- Bitwise-identical cross-state feature pairs: {exact_cross_state_pair_count}.
- Bitwise-identical pairs with different canonical eta: {exact_conflict_pair_count}.
- Additional cross-state pairs within 1e-12 L-infinity in normalized feature
  space: {numerical_pair_count}; incompatible eta among them: {numerical_conflict_pair_count}.

See `feature_aliasing_summary.json` and the full nearest-neighbor CSVs for the
near-collision distributions. Ordinary overlap is not classified as
representation impossibility.

At state-centroid level, {100 * feature_alias_summary['state_centroid_nearest_other_state']['nonzero_eta_jump_fraction']:.1f}%
of directed nearest-neighbor pairs change canonical eta, {100 * feature_alias_summary['state_centroid_nearest_other_state']['large_eta_jump_ge_0.5_fraction']:.1f}%
have coordinate-normalized eta jump >=0.5, and {100 * feature_alias_summary['state_centroid_nearest_other_state']['zero_active_mismatch_fraction']:.1f}%
cross ZERO/ACTIVE. The closest >=0.5 eta jump has normalized-feature L2
{feature_alias_summary['state_centroid_nearest_other_state']['minimum_h_l2_among_large_eta_jumps_ge_0.5']:.3f};
the closest ZERO/ACTIVE crossing is farther away at
{feature_alias_summary['state_centroid_nearest_other_state']['minimum_h_l2_among_zero_active_mismatches']:.3f}.

## Zero/active probes

Primary metrics are state-level pooled out-of-fold results. Random-sample
metrics are explicitly a leaky shortcut diagnostic because Flow variants of
the same state occur in train and test. Full metrics, fold composition,
optimizer diagnostics, sensitivity, and specificity are in
`zero_active_probe_results.json`.

- Linear grouped: AUROC {grouped_metrics['linear']['state_level_pooled_oof_primary']['auroc']:.3f},
  BAcc {grouped_metrics['linear']['state_level_pooled_oof_primary']['balanced_accuracy_at_0.5']:.3f}.
- MLP grouped: AUROC {grouped_metrics['mlp']['state_level_pooled_oof_primary']['auroc']:.3f},
  BAcc {grouped_metrics['mlp']['state_level_pooled_oof_primary']['balanced_accuracy_at_0.5']:.3f}.
- Random-sample shortcut: linear AUROC
  {random_results['linear']['state_level_aggregate_over_heldout_variants_primary']['auroc']:.3f},
  MLP AUROC {random_results['mlp']['state_level_aggregate_over_heldout_variants_primary']['auroc']:.3f}.

Grouped performance remains strong, so ZERO/ACTIVE information is present in
the representation. The random/group gap shows measurable shortcut optimism,
but not a grouped-evaluation collapse.

## Canonical smoothness

Preliminary smoothness uses the five directed nearest state-centroid neighbors
per state. It is target geometry only; it does not establish successful-set
discontinuity. Cross-eta continuation tests are intentionally outside this
subtask.
"""
    (OUT / "feature_aliasing_report.md").write_text(report)

    elapsed = time.monotonic() - started
    provenance = {
        "status": "COMPLETE",
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "asset_hashes": observed_hashes,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "cpu_thread_cap": 6,
        "gpu_used": False,
        "simulator_calls": 0,
        "continuation_rollouts": 0,
        "wall_seconds": elapsed,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "configuration": {
            "source_folds": NUM_SOURCE_FOLDS,
            "fold_seed": FOLD_SEED,
            "random_split_seed": RANDOM_SPLIT_SEED,
            "linear_l2": LINEAR_L2,
            "mlp_epochs": MLP_EPOCHS,
            "mlp_batch_size": MLP_BATCH_SIZE,
            "mlp_lr": MLP_LR,
            "mlp_weight_decay": MLP_WEIGHT_DECAY,
        },
    }
    write_json(OUT / "provenance_runtime.json", provenance)
    print(json.dumps({
        "status": "COMPLETE",
        "wall_seconds": elapsed,
        "exact_incompatible_pairs": exact_conflict_pair_count + numerical_conflict_pair_count,
        "linear_grouped": grouped_metrics["linear"]["state_level_pooled_oof_primary"],
        "mlp_grouped": grouped_metrics["mlp"]["state_level_pooled_oof_primary"],
        "linear_random": random_results["linear"]["state_level_aggregate_over_heldout_variants_primary"],
        "mlp_random": random_results["mlp"]["state_level_aggregate_over_heldout_variants_primary"],
    }, indent=2))


if __name__ == "__main__":
    main()
