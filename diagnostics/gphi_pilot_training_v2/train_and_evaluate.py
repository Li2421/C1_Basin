"""Train and audit the first deterministic G_phi pilot.

This diagnostic is deliberately self-contained.  It reads the immutable v1
oracle dataset, fits train-only preprocessing, compares compact deterministic
regressors, and replays the unchanged hard projection for offline evaluation.
It never writes to the dataset or to frozen controller/environment sources.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "diagnostics" / "gphi_training_dataset_v2"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
VENV = SYSROOT / ".venv-c1"
VMAX = 0.5
EPS = 1e-12

# When this file is executed by path, Python otherwise exposes HERE but not the
# repository root.  Projection replay imports an isolated diagnostic helper.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def git_revision() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True))
        return {"available": True, "commit": commit, "dirty": dirty}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def dataset_audit() -> tuple[dict, dict, list[dict], dict, dict]:
    manifest = read_json(DATA / "manifest.json")
    schema = read_json(DATA / "feature_schema.json")
    split_manifest = read_json(DATA / "split_manifest.json")
    metadata = read_jsonl(DATA / "sample_metadata.jsonl")
    state_manifest = read_jsonl(DATA / "state_manifest.jsonl")

    expected_hash = manifest["files_sha256"]["samples.npz"]
    actual_hash = sha(DATA / "samples.npz")
    with np.load(DATA / "samples.npz", allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}

    required = {"features", "targets", "target_actions", "u_safe", "positions",
                "state_id", "split", "category", "sample_id", "flow_seed"}
    failures = []
    if not required.issubset(arrays):
        failures.append(f"missing npz fields: {sorted(required-set(arrays))}")
    expected_samples = int(manifest["supervised_samples"])
    if arrays["features"].shape != (expected_samples, 214):
        failures.append(f"features shape {arrays['features'].shape}")
    if arrays["targets"].shape != (expected_samples, 4):
        failures.append(f"targets shape {arrays['targets'].shape}")
    if len(metadata) != expected_samples:
        failures.append(f"metadata rows {len(metadata)}")
    if schema.get("feature_dimension") != 214 or schema.get("target_dimension") != 4:
        failures.append("feature schema dimensions do not equal 214 -> 4")
    if actual_hash != expected_hash:
        failures.append("samples.npz does not reproduce manifest SHA256")
    if not np.isfinite(arrays["features"]).all() or not np.isfinite(arrays["targets"]).all():
        failures.append("NaN/Inf in features or targets")
    if not np.allclose(arrays["target_actions"],
                       arrays["u_safe"].reshape(-1, 4) + arrays["targets"],
                       atol=1e-12, rtol=0):
        failures.append("target action != u_safe + target")

    ids_from_meta = np.asarray([row["sample_id"] for row in metadata])
    if not np.array_equal(ids_from_meta, arrays["sample_id"]):
        failures.append("sample_metadata order does not match samples.npz")
    if len(set(arrays["sample_id"].tolist())) != len(arrays["sample_id"]):
        failures.append("duplicate sample IDs")

    usable_ids = set(arrays["state_id"].tolist())
    expected_counts = {
        name: 64 * sum(row["split"] == name and row["state_id"] in usable_ids
                       for row in state_manifest)
        for name in ("train", "validation", "test")
    }
    actual_counts = {name: int(np.sum(arrays["split"] == name)) for name in expected_counts}
    if actual_counts != expected_counts:
        failures.append(f"split counts {actual_counts}")
    actual_states = {
        split: set(arrays["state_id"][arrays["split"] == split].tolist())
        for split in expected_counts
    }
    for split, expected in split_manifest["state_ids"].items():
        if not actual_states[split].issubset(set(expected)):
            failures.append(f"{split} state membership differs from split manifest")
    overlap = {
        f"{a}_{b}": sorted(actual_states[a] & actual_states[b])
        for a, b in combinations(expected_counts, 2)
    }
    if any(overlap.values()):
        failures.append(f"state leakage: {overlap}")
    meta_groups = {
        split: {row["leakage_group"] for row in metadata if row["split"] == split}
        for split in expected_counts
    }
    group_overlap = {
        f"{a}_{b}": sorted(meta_groups[a] & meta_groups[b])
        for a, b in combinations(expected_counts, 2)
    }
    if any(group_overlap.values()):
        failures.append(f"trajectory leakage: {group_overlap}")

    target_norm = np.linalg.norm(arrays["targets"], axis=1)
    zero_meta = np.asarray([row["zero_label"] for row in metadata], dtype=bool)
    zero_actual = target_norm <= 1e-14
    if not np.array_equal(zero_meta, zero_actual):
        failures.append("zero-label metadata mismatch")
    category_counts = Counter(row["category"] for row in state_manifest)
    label_classes = Counter(row["label_classification"] for row in metadata)
    audit = {
        "passed": not failures,
        "failures": failures,
        "dataset_manifest_sha256": sha(DATA / "manifest.json"),
        "samples_sha256_expected": expected_hash,
        "samples_sha256_actual": actual_hash,
        "sample_count": int(len(arrays["targets"])),
        "feature_dimension": int(arrays["features"].shape[1]),
        "target_dimension": int(arrays["targets"].shape[1]),
        "split_sample_counts": actual_counts,
        "split_state_counts": {key: len(value) for key, value in actual_states.items()},
        "split_state_overlap": overlap,
        "split_trajectory_overlap": group_overlap,
        "all_finite": bool(np.isfinite(arrays["features"]).all()
                           and np.isfinite(arrays["targets"]).all()),
        "unique_states": len(set(arrays["state_id"].tolist())),
        "category_state_counts": dict(category_counts),
        "zero_label_samples": int(zero_actual.sum()),
        "nonzero_label_samples": int((~zero_actual).sum()),
        "zero_label_states": len(set(arrays["state_id"][zero_actual].tolist())),
        "nonzero_label_states": len(set(arrays["state_id"][~zero_actual].tolist())),
        "label_classification_sample_counts": dict(label_classes),
        "target_norm": {
            "min": float(target_norm.min()), "mean": float(target_norm.mean()),
            "median": float(np.median(target_norm)), "max": float(target_norm.max()),
            "std": float(target_norm.std()),
        },
        "feature_schema_future_information": bool(schema.get("future_information_in_features", True)),
    }
    return arrays, audit, metadata, schema, manifest


def fit_normalization(x_train: np.ndarray, schema: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    binary = np.zeros(x_train.shape[1], dtype=bool)
    for segment in schema["segments"]:
        if segment["unit"] == "boolean":
            start = int(segment["offset"])
            binary[start:start + int(segment["length"])] = True
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    constant = scale < 1e-8
    scale[constant] = 1.0
    # Binary/latch indicators retain their literal 0/1 semantics.
    mean[binary] = 0.0
    scale[binary] = 1.0
    return mean, scale, binary


def normalize(x, mean, scale):
    return (x - mean) / scale


def subset_metrics(pred: np.ndarray, target: np.ndarray, state_id: np.ndarray) -> dict:
    if len(pred) == 0:
        return {name: float("nan") for name in (
            "mse", "rmse", "mean_l2", "median_l2", "max_l2",
            "mean_l2_over_vmax", "mean_relative_l2_nonzero",
            "state_grouped_mse", "state_grouped_rmse", "state_grouped_mean_l2")}
    error = pred - target
    l2 = np.linalg.norm(error, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    relative = l2[target_norm > EPS] / target_norm[target_norm > EPS]
    per_state = []
    for sid in sorted(set(state_id.tolist())):
        mask = state_id == sid
        per_state.append((float(np.mean(error[mask] ** 2)), float(np.mean(l2[mask]))))
    state_mse = float(np.mean([row[0] for row in per_state]))
    return {
        "sample_count": int(len(pred)), "state_count": int(len(per_state)),
        "mse": float(np.mean(error ** 2)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mean_l2": float(np.mean(l2)), "median_l2": float(np.median(l2)),
        "max_l2": float(np.max(l2)), "mean_l2_over_vmax": float(np.mean(l2) / VMAX),
        "mean_relative_l2_nonzero": float(np.mean(relative)) if len(relative) else 0.0,
        "median_relative_l2_nonzero": float(np.median(relative)) if len(relative) else 0.0,
        "state_grouped_mse": state_mse,
        "state_grouped_rmse": float(math.sqrt(state_mse)),
        "state_grouped_mean_l2": float(np.mean([row[1] for row in per_state])),
    }


def state_metric_rows(model: str, split: str, pred, target, arrays, indices,
                      metadata_by_sample) -> list[dict]:
    rows = []
    state_ids = arrays["state_id"][indices]
    for sid in sorted(set(state_ids.tolist())):
        local = state_ids == sid
        chosen = indices[local]
        error = pred[local] - target[local]
        l2 = np.linalg.norm(error, axis=1)
        norm = np.linalg.norm(target[local], axis=1)
        meta = metadata_by_sample[str(arrays["sample_id"][chosen[0]])]
        rows.append({
            "model": model, "split": split, "state_id": sid,
            "category": str(arrays["category"][chosen[0]]),
            "label_classification": meta["label_classification"],
            "zero_label_state": bool(np.all(norm <= 1e-14)),
            "mse": float(np.mean(error ** 2)), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "mean_l2": float(np.mean(l2)), "median_l2": float(np.median(l2)),
            "max_l2": float(np.max(l2)), "mean_l2_over_vmax": float(np.mean(l2) / VMAX),
            "mean_target_norm": float(np.mean(norm)),
            "mean_relative_l2_nonzero": (float(np.mean(l2[norm > EPS] / norm[norm > EPS]))
                                          if np.any(norm > EPS) else 0.0),
        })
    return rows


def ridge_fit(x, y, weight_decay: float):
    augmented = np.c_[x, np.ones(len(x))]
    gram = augmented.T @ augmented
    penalty = np.eye(gram.shape[0]) * weight_decay * len(x)
    penalty[-1, -1] = 0.0
    try:
        coef = np.linalg.solve(gram + penalty, augmented.T @ y)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(augmented, y, rcond=1e-10)[0]
    return coef[:-1], coef[-1]


def ridge_predict(model, x):
    return x @ model[0] + model[1]


def init_mlp(jax, layer_sizes, seed):
    key = jax.random.PRNGKey(seed)
    params = []
    for index, (fan_in, fan_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        key, subkey = jax.random.split(key)
        gain = math.sqrt(2.0 if index < len(layer_sizes) - 2 else 1.0)
        weight = jax.random.normal(subkey, (fan_in, fan_out)) * (gain / math.sqrt(fan_in))
        params.append({"w": weight, "b": np.zeros(fan_out, dtype=np.float32)})
    return params


def mlp_apply(jax, params, x):
    value = x
    for index, layer in enumerate(params):
        value = value @ layer["w"] + layer["b"]
        if index < len(params) - 1:
            value = jax.nn.silu(value)
    return value


def tree_to_numpy(jax, tree):
    return jax.tree.map(lambda value: np.asarray(value), tree)


def train_mlp(x_train, y_train, sid_train, x_val, y_val, sid_val, config):
    import jax
    import jax.numpy as jnp
    import optax

    sizes = [x_train.shape[1], *config["hidden"], y_train.shape[1]]
    params = init_mlp(jax, sizes, config["seed"])
    optimizer = optax.adamw(config["learning_rate"], weight_decay=config["weight_decay"])
    state = optimizer.init(params)

    @jax.jit
    def update(params, state, xb, yb):
        def objective(candidate):
            prediction = mlp_apply(jax, candidate, xb)
            return jnp.mean((prediction - yb) ** 2)
        loss, grad = jax.value_and_grad(objective)(params)
        updates, state = optimizer.update(grad, state, params)
        return optax.apply_updates(params, updates), state, loss

    @jax.jit
    def predict(params, x):
        return mlp_apply(jax, params, x)

    x_train_jax = jnp.asarray(x_train, dtype=jnp.float32)
    y_train_jax = jnp.asarray(y_train, dtype=jnp.float32)
    x_val_jax = jnp.asarray(x_val, dtype=jnp.float32)
    rng = np.random.default_rng(config["seed"])
    best_params = tree_to_numpy(jax, params)
    best_score = float("inf")
    best_epoch = 0
    evaluations_without_improvement = 0
    history = []
    start = time.monotonic()

    for epoch in range(1, config["max_epochs"] + 1):
        ordering = rng.permutation(len(x_train))
        losses = []
        for begin in range(0, len(ordering), config["batch_size"]):
            batch = ordering[begin:begin + config["batch_size"]]
            params, state, loss = update(params, state, x_train_jax[batch], y_train_jax[batch])
            losses.append(float(loss))
        if epoch == 1 or epoch % config["eval_interval"] == 0:
            val_prediction = np.asarray(predict(params, x_val_jax))
            train_prediction = np.asarray(predict(params, x_train_jax))
            val_metrics = subset_metrics(val_prediction, y_val, sid_val)
            train_metrics = subset_metrics(train_prediction, y_train, sid_train)
            score = val_metrics["state_grouped_mse"]
            history.append({
                "run": config["name"], "epoch": epoch,
                "minibatch_loss": float(np.mean(losses)),
                "train_state_grouped_mse": train_metrics["state_grouped_mse"],
                "validation_state_grouped_mse": score,
                "validation_state_grouped_mean_l2": val_metrics["state_grouped_mean_l2"],
                "elapsed_s": time.monotonic() - start,
            })
            if score < best_score - config["min_delta"]:
                best_score = score
                best_epoch = epoch
                best_params = tree_to_numpy(jax, params)
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
            if evaluations_without_improvement >= config["patience_evaluations"]:
                break
    # Synchronize device work before measuring runtime.
    np.asarray(predict(best_params, x_val_jax)).copy()
    runtime = time.monotonic() - start
    return best_params, history, {
        "best_epoch": best_epoch, "epochs_completed": epoch,
        "best_validation_state_grouped_mse": best_score,
        "runtime_s": runtime, "parameter_count": int(sum(
            np.asarray(layer["w"]).size + np.asarray(layer["b"]).size
            for layer in best_params)),
    }


def predict_mlp(params, x):
    import jax
    import jax.numpy as jnp
    return np.asarray(mlp_apply(jax, params, jnp.asarray(x, dtype=jnp.float32)))


def save_checkpoint(path: Path, params, config, mean, scale, binary):
    payload = {
        "architecture_json": np.asarray(json.dumps(config, sort_keys=True)),
        "normalization_mean": mean, "normalization_scale": scale,
        "normalization_binary_mask": binary.astype(np.uint8),
    }
    for index, layer in enumerate(params):
        payload[f"layer_{index}_weight"] = np.asarray(layer["w"])
        payload[f"layer_{index}_bias"] = np.asarray(layer["b"])
    np.savez_compressed(path, **payload)


def projection_replay(predictions: dict, arrays, indices_by_split, protocol) -> dict:
    sys.path.insert(0, str(SYSROOT))
    from single_integrator.cbf import CBFConfig, barrier_constraints
    from single_integrator.environment import Config, GiveWayEnv
    from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry

    config = Config(**protocol["environment"])
    cbf = CBFConfig()
    template = GiveWayEnv(config)
    walls = template.walls.copy()
    results = {}
    constraint_cache = {}
    for split in ("validation", "test"):
        indices = indices_by_split[split]
        prediction = predictions[split]
        exec_errors = []
        projection_changes = []
        already_feasible = []
        retry_count = 0
        failures = []
        oracle_reconstruction = []
        for local, index in enumerate(indices):
            sid = str(arrays["state_id"][index])
            if sid not in constraint_cache:
                snapshot = {
                    "positions": arrays["positions"][index], "walls": walls,
                    "config": config.to_dict(),
                }
                constraint_cache[sid] = barrier_constraints(snapshot, cbf)[:2]
            A, lower = constraint_cache[sid]
            safe = arrays["u_safe"][index].reshape(4)
            wanted = safe + prediction[local]
            linear_ok = float(np.min(A @ wanted - lower)) >= -cbf.feasibility_tol
            speed_ok = float(np.max(np.linalg.norm(wanted.reshape(2, 2), axis=1))) <= config.max_speed + cbf.speed_tol
            already_feasible.append(linear_ok and speed_ok)
            try:
                executed, _, retried, _ = project_velocity_with_retry(
                    wanted.reshape(2, 2), A, lower, config.max_speed, cbf)
                retry_count += int(retried)
                executed = executed.reshape(4)
                oracle_action = arrays["target_actions"][index].reshape(4)
                exec_errors.append(float(np.linalg.norm(executed - oracle_action)))
                projection_changes.append(float(np.linalg.norm(executed - wanted)))
                oracle_reconstruction.append(float(np.max(np.abs(
                    oracle_action - (safe + arrays["targets"][index])))))
            except Exception as exc:
                failures.append({"sample_id": str(arrays["sample_id"][index]), "error": repr(exc)})
        value = {
            "sample_count": len(indices), "projection_failures": failures,
            "retry_count": retry_count,
            "already_feasible_fraction": float(np.mean(already_feasible)),
            "projection_changed_fraction_over_1e-6": float(np.mean(np.asarray(projection_changes) > 1e-6)) if projection_changes else float("nan"),
            "executed_action_error": {
                "mean": float(np.mean(exec_errors)) if exec_errors else float("nan"),
                "std": float(np.std(exec_errors)) if exec_errors else float("nan"),
                "median": float(np.median(exec_errors)) if exec_errors else float("nan"),
                "max": float(np.max(exec_errors)) if exec_errors else float("nan"),
                "mean_over_vmax": float(np.mean(exec_errors) / VMAX) if exec_errors else float("nan"),
            },
            "projection_rewrite_norm": {
                "mean": float(np.mean(projection_changes)) if projection_changes else float("nan"),
                "median": float(np.median(projection_changes)) if projection_changes else float("nan"),
                "max": float(np.max(projection_changes)) if projection_changes else float("nan"),
            },
            "oracle_action_reconstruction_max_abs_error": float(np.max(oracle_reconstruction)) if oracle_reconstruction else float("nan"),
        }
        results[split] = value
    return results


def nearest_neighbor_audit(x_norm, arrays, indices_by_split, best_predictions) -> dict:
    centroids = {}
    for sid in sorted(set(arrays["state_id"].tolist())):
        centroids[sid] = x_norm[arrays["state_id"] == sid].mean(axis=0)
    train_states = sorted(set(arrays["state_id"][indices_by_split["train"]].tolist()))
    report = {}
    for split in ("validation", "test"):
        split_states = sorted(set(arrays["state_id"][indices_by_split[split]].tolist()))
        rows = []
        for sid in split_states:
            distances = [(other, float(np.linalg.norm(centroids[sid] - centroids[other]) /
                                        math.sqrt(x_norm.shape[1]))) for other in train_states]
            nearest = min(distances, key=lambda pair: pair[1])
            rows.append({"state_id": sid, "nearest_train_state": nearest[0],
                         "normalized_feature_rms_distance": nearest[1]})
        report[split] = {
            "per_state": rows,
            "mean_nearest_state_distance": float(np.mean([row["normalized_feature_rms_distance"] for row in rows])),
            "min_nearest_state_distance": float(np.min([row["normalized_feature_rms_distance"] for row in rows])),
            "max_nearest_state_distance": float(np.max([row["normalized_feature_rms_distance"] for row in rows])),
        }

    # Check for output collapse at the held-out state-mean level.
    split = "test"
    indices = indices_by_split[split]
    prediction = best_predictions[split]
    target = arrays["targets"][indices]
    ids = arrays["state_id"][indices]
    pred_means, target_means = [], []
    for sid in sorted(set(ids.tolist())):
        mask = ids == sid
        pred_means.append(prediction[mask].mean(axis=0))
        target_means.append(target[mask].mean(axis=0))
    pred_means, target_means = np.asarray(pred_means), np.asarray(target_means)
    def mean_pairwise(value):
        distances = [np.linalg.norm(value[i] - value[j])
                     for i, j in combinations(range(len(value)), 2)]
        return float(np.mean(distances)) if distances else 0.0
    report["prediction_collapse"] = {
        "test_state_mean_prediction_pairwise_l2": mean_pairwise(pred_means),
        "test_state_mean_target_pairwise_l2": mean_pairwise(target_means),
        "prediction_to_target_pairwise_spread_ratio": (
            mean_pairwise(pred_means) / max(mean_pairwise(target_means), EPS)),
        "unique_test_state_mean_predictions_rounded_1e-5": int(len(
            {tuple(np.round(row, 5)) for row in pred_means})),
    }
    return report


def stratified_state_order(arrays, train_indices: np.ndarray) -> list[str]:
    """Deterministic nested order balanced over category and zero/nonzero label."""
    buckets = defaultdict(list)
    state_ids = arrays["state_id"][train_indices]
    for sid in sorted(set(state_ids.tolist())):
        chosen = train_indices[state_ids == sid]
        category = str(arrays["category"][chosen[0]])
        zero = bool(np.all(np.linalg.norm(arrays["targets"][chosen], axis=1) <= 1e-14))
        digest = hashlib.sha256(sid.encode()).hexdigest()
        buckets[(category, zero)].append((digest, sid))
    for key in buckets:
        buckets[key].sort()
    order = []
    keys = sorted(buckets)
    total = sum(len(value) for value in buckets.values())
    while len(order) < total:
        for key in keys:
            if buckets[key]:
                order.append(buckets[key].pop(0)[1])
    return order


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    arrays, audit, metadata, schema, dataset_manifest = dataset_audit()
    write_json(HERE / "sanity_checks.json", audit)
    if not audit["passed"]:
        raise SystemExit("dataset audit failed: " + "; ".join(audit["failures"]))
    if args.audit_only:
        print(json.dumps(audit, indent=2))
        return

    # ML imports intentionally follow the hard dataset gate.
    import jax
    import flax
    import optax
    print(json.dumps({
        "stage": "deterministic G_phi pilot training",
        "samples": len(arrays["targets"]), "device": [str(x) for x in jax.devices()],
        "models": ["ZERO", "TRAIN_MEAN", "LINEAR", "MLP_128x128"],
        "selection": "validation state-grouped MSE only",
    }))

    indices_by_split = {
        split: np.flatnonzero(arrays["split"] == split)
        for split in ("train", "validation", "test")
    }
    train_indices = indices_by_split["train"]
    mean, scale, binary = fit_normalization(arrays["features"][train_indices], schema)
    x_norm = normalize(arrays["features"], mean, scale).astype(np.float32)
    y = arrays["targets"].astype(np.float32)
    normalization = {
        "fit_split": "train only", "epsilon": 1e-8,
        "rule": "continuous: (x-mu)/max(sigma,eps); schema boolean/latch: unchanged",
        "binary_feature_indices": np.flatnonzero(binary).tolist(),
        "constant_continuous_feature_indices": np.flatnonzero((scale == 1.0) & (~binary) &
                                                                (arrays["features"][train_indices].std(axis=0) < 1e-8)).tolist(),
        "mean": mean.tolist(), "scale": scale.tolist(),
    }
    write_json(HERE / "normalization.json", normalization)

    split_data = {}
    for split, indices in indices_by_split.items():
        split_data[split] = (x_norm[indices], y[indices], arrays["state_id"][indices])
    metadata_by_sample = {row["sample_id"]: row for row in metadata}

    model_predictions = defaultdict(dict)
    comparison = []
    state_rows = []
    # Required no-learning baselines.
    baseline_predictors = {
        "ZERO": np.zeros(4, dtype=np.float32),
        "TRAIN_MEAN": y[train_indices].mean(axis=0),
    }
    for name, constant in baseline_predictors.items():
        row = {"model": name, "kind": "baseline", "parameter_count": 0,
               "selected_by_validation": False}
        for split, (_, target, state_id) in split_data.items():
            prediction = np.broadcast_to(constant, target.shape).copy()
            model_predictions[name][split] = prediction
            metrics = subset_metrics(prediction, target, state_id)
            for key in ("mse", "state_grouped_mse", "state_grouped_mean_l2"):
                row[f"{split}_{key}"] = metrics[key]
            state_rows.extend(state_metric_rows(
                name, split, prediction, target, arrays, indices_by_split[split], metadata_by_sample))
        comparison.append(row)

    # Deterministic linear baseline; regularization selected only on validation.
    linear_candidates = []
    for wd in (0.0, 1e-4):
        linear = ridge_fit(split_data["train"][0], split_data["train"][1], wd)
        pred_val = ridge_predict(linear, split_data["validation"][0])
        score = subset_metrics(pred_val, split_data["validation"][1],
                               split_data["validation"][2])["state_grouped_mse"]
        linear_candidates.append((score, wd, linear))
    _, linear_wd, linear = min(linear_candidates, key=lambda row: row[0])
    linear_row = {"model": "LINEAR", "kind": "linear", "parameter_count": 860,
                  "weight_decay": linear_wd, "selected_by_validation": False}
    for split, (features, target, state_id) in split_data.items():
        prediction = ridge_predict(linear, features)
        model_predictions["LINEAR"][split] = prediction
        metrics = subset_metrics(prediction, target, state_id)
        for key in ("mse", "state_grouped_mse", "state_grouped_mean_l2"):
            linear_row[f"{split}_{key}"] = metrics[key]
        state_rows.extend(state_metric_rows(
            "LINEAR", split, prediction, target, arrays, indices_by_split[split], metadata_by_sample))
    comparison.append(linear_row)

    base_training = {
        "batch_size": 256, "max_epochs": 1200, "eval_interval": 5,
        "patience_evaluations": 35, "min_delta": 1e-9,
    }
    screening = [
        {"name": "MLP128_lr1e-3_wd1e-5_s17", "hidden": [128, 128],
         "learning_rate": 1e-3, "weight_decay": 1e-5, "seed": 17},
    ]
    histories = []
    screening_models = {}
    screening_summaries = []
    for candidate in screening:
        config = {**base_training, **candidate}
        params, history, summary = train_mlp(
            *split_data["train"], *split_data["validation"], config)
        histories.extend(history)
        screening_models[candidate["name"]] = (params, config, summary)
        pred_train = predict_mlp(params, split_data["train"][0])
        pred_val = predict_mlp(params, split_data["validation"][0])
        train_metrics = subset_metrics(pred_train, split_data["train"][1], split_data["train"][2])
        val_metrics = subset_metrics(pred_val, split_data["validation"][1], split_data["validation"][2])
        screening_summaries.append({
            "model": candidate["name"], "kind": "screening_mlp",
            "hidden": "x".join(map(str, candidate["hidden"])),
            "learning_rate": candidate["learning_rate"], "weight_decay": candidate["weight_decay"],
            "seed": candidate["seed"], **summary,
            "train_mse": train_metrics["mse"],
            "train_state_grouped_mse": train_metrics["state_grouped_mse"],
            "train_state_grouped_mean_l2": train_metrics["state_grouped_mean_l2"],
            "validation_mse": val_metrics["mse"],
            "validation_state_grouped_mse": val_metrics["state_grouped_mse"],
            "validation_state_grouped_mean_l2": val_metrics["state_grouped_mean_l2"],
            "test_mse": "NOT_EVALUATED_FOR_SELECTION",
            "selected_by_validation": False,
        })
    architecture_winner = min(
        screening_summaries, key=lambda row: row["validation_state_grouped_mse"])
    architecture_winner["selected_by_validation"] = True
    selected_name = architecture_winner["model"]
    selected_params, selected_config, selected_summary = screening_models[selected_name]

    # Three seeds for the final validation-selected architecture/hyperparameters.
    final_runs = []
    checkpoint_dir = HERE / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for seed in (17, 23, 41):
        if seed == 17:
            params, config, summary = selected_params, dict(selected_config), dict(selected_summary)
        else:
            config = dict(selected_config)
            config["seed"] = seed
            config["name"] = f"FINAL_{'x'.join(map(str, config['hidden']))}_s{seed}"
            params, history, summary = train_mlp(
                *split_data["train"], *split_data["validation"], config)
            histories.extend(history)
        pred_train = predict_mlp(params, split_data["train"][0])
        pred_val = predict_mlp(params, split_data["validation"][0])
        train_metrics = subset_metrics(pred_train, split_data["train"][1], split_data["train"][2])
        val_metrics = subset_metrics(pred_val, split_data["validation"][1], split_data["validation"][2])
        save_checkpoint(checkpoint_dir / f"final_seed{seed}.npz", params, config, mean, scale, binary)
        final_runs.append({
            "seed": seed, "params": params, "config": config, "summary": summary,
            "train_metrics": train_metrics, "validation_metrics": val_metrics,
        })
    # Seed choice is validation-only.  Test is opened exactly once below.
    best = min(final_runs, key=lambda row: row["validation_metrics"]["state_grouped_mse"])
    best_params, best_config = best["params"], best["config"]
    best_name = f"SELECTED_MLP_seed{best['seed']}"
    best_predictions = {}
    best_metrics = {}
    for split, (features, target, state_id) in split_data.items():
        prediction = predict_mlp(best_params, features)
        best_predictions[split] = prediction
        best_metrics[split] = subset_metrics(prediction, target, state_id)
        state_rows.extend(state_metric_rows(
            best_name, split, prediction, target, arrays, indices_by_split[split], metadata_by_sample))

    # Data-scaling diagnostic: same 128x128 architecture, optimizer, loss,
    # seed, early stopping, and fixed validation states.  Only the number of
    # independent training states changes; each subset fits its own train-only
    # normalization to prevent leakage from excluded states.
    state_order = stratified_state_order(arrays, train_indices)
    full_train_state_count = len(state_order)
    subset_sizes = sorted(set(min(size, full_train_state_count) for size in (46, 84, 126, full_train_state_count)))
    scaling_rows = []
    for state_count in subset_sizes:
        selected_states = set(state_order[:state_count])
        subset_indices = train_indices[np.isin(arrays["state_id"][train_indices], list(selected_states))]
        subset_mean, subset_scale, _ = fit_normalization(arrays["features"][subset_indices], schema)
        subset_x_train = normalize(arrays["features"][subset_indices], subset_mean, subset_scale).astype(np.float32)
        subset_y_train = y[subset_indices]
        subset_sid_train = arrays["state_id"][subset_indices]
        subset_x_val = normalize(arrays["features"][indices_by_split["validation"]], subset_mean, subset_scale).astype(np.float32)
        subset_config = {
            **base_training, "name": f"SCALING_{state_count}_states_seed23",
            "hidden": [128, 128], "learning_rate": 1e-3,
            "weight_decay": 1e-5, "seed": 23,
        }
        if state_count == full_train_state_count:
            run = next(item for item in final_runs if item["seed"] == 23)
            subset_params = run["params"]
            subset_summary = run["summary"]
            subset_train_prediction = predict_mlp(subset_params, subset_x_train)
            subset_val_prediction = predict_mlp(subset_params, subset_x_val)
        else:
            subset_params, subset_history, subset_summary = train_mlp(
                subset_x_train, subset_y_train, subset_sid_train,
                subset_x_val, y[indices_by_split["validation"]],
                arrays["state_id"][indices_by_split["validation"]], subset_config)
            histories.extend(subset_history)
            subset_train_prediction = predict_mlp(subset_params, subset_x_train)
            subset_val_prediction = predict_mlp(subset_params, subset_x_val)
        train_subset_metrics = subset_metrics(
            subset_train_prediction, subset_y_train, subset_sid_train)
        val_subset_metrics = subset_metrics(
            subset_val_prediction, y[indices_by_split["validation"]],
            arrays["state_id"][indices_by_split["validation"]])
        val_zero = np.linalg.norm(y[indices_by_split["validation"]], axis=1) <= 1e-14
        scaling_rows.append({
            "independent_training_states": state_count,
            "approx_total_dataset_state_equivalent": int(round(state_count * len(set(arrays["state_id"].tolist())) / full_train_state_count)),
            "training_samples": len(subset_indices),
            "zero_label_training_states": sum(
                np.all(np.linalg.norm(arrays["targets"][subset_indices][subset_sid_train == sid], axis=1) <= 1e-14)
                for sid in selected_states),
            "nonzero_label_training_states": state_count - sum(
                np.all(np.linalg.norm(arrays["targets"][subset_indices][subset_sid_train == sid], axis=1) <= 1e-14)
                for sid in selected_states),
            "seed": 23, "best_epoch": subset_summary["best_epoch"],
            "train_state_grouped_mean_l2": train_subset_metrics["state_grouped_mean_l2"],
            "validation_state_grouped_mean_l2": val_subset_metrics["state_grouped_mean_l2"],
            "validation_state_grouped_mse": val_subset_metrics["state_grouped_mse"],
            "validation_zero_label_false_intervention_mean": float(
                np.mean(np.linalg.norm(subset_val_prediction[val_zero], axis=1))) if np.any(val_zero) else float("nan"),
        })
    write_csv(HERE / "scaling_analysis.csv", scaling_rows)

    for row in screening_summaries:
        comparison.append(row)
    for run in final_runs:
        row = {
            "model": f"FINAL_seed{run['seed']}", "kind": "final_architecture_seed",
            "hidden": "x".join(map(str, run["config"]["hidden"])),
            "parameter_count": run["summary"]["parameter_count"],
            "learning_rate": run["config"]["learning_rate"],
            "weight_decay": run["config"]["weight_decay"], "seed": run["seed"],
            "best_epoch": run["summary"]["best_epoch"],
            "train_mse": run["train_metrics"]["mse"],
            "train_state_grouped_mse": run["train_metrics"]["state_grouped_mse"],
            "train_state_grouped_mean_l2": run["train_metrics"]["state_grouped_mean_l2"],
            "validation_mse": run["validation_metrics"]["mse"],
            "validation_state_grouped_mse": run["validation_metrics"]["state_grouped_mse"],
            "validation_state_grouped_mean_l2": run["validation_metrics"]["state_grouped_mean_l2"],
            "test_mse": (best_metrics["test"]["mse"] if run["seed"] == best["seed"]
                         else "NOT_EVALUATED"),
            "selected_by_validation": run["seed"] == best["seed"],
        }
        comparison.append(row)

    save_checkpoint(HERE / "best_checkpoint.npz", best_params, best_config, mean, scale, binary)
    write_csv(HERE / "training_history.csv", histories)
    write_csv(HERE / "model_comparison.csv", comparison)
    write_csv(HERE / "state_grouped_metrics.csv", state_rows)

    # Category, label-zero/nonzero, and ambiguity diagnostics for selected model.
    category_rows, ambiguity_rows = [], []
    for split, indices in indices_by_split.items():
        pred, target = best_predictions[split], y[indices]
        sid = arrays["state_id"][indices]
        for category in ("NORMAL", "PRE_DEADLOCK", "RECOVERY"):
            mask = arrays["category"][indices] == category
            row = {"split": split, "group_type": "category", "group": category,
                   **subset_metrics(pred[mask], target[mask], sid[mask])}
            category_rows.append(row)
        zero = np.linalg.norm(target, axis=1) <= 1e-14
        for label, mask in (("zero_label", zero), ("nonzero_label", ~zero)):
            metrics = subset_metrics(pred[mask], target[mask], sid[mask])
            row = {"split": split, "group_type": "label", "group": label, **metrics}
            if label == "zero_label" and np.any(mask):
                row["false_intervention_mean_norm"] = float(np.mean(np.linalg.norm(pred[mask], axis=1)))
                row["false_intervention_max_norm"] = float(np.max(np.linalg.norm(pred[mask], axis=1)))
            category_rows.append(row)
        classifications = np.asarray([
            metadata_by_sample[str(arrays["sample_id"][index])]["label_classification"]
            for index in indices
        ])
        for label in ("LABEL_STABLE", "LABEL_MILDLY_AMBIGUOUS"):
            mask = classifications == label
            ambiguity_rows.append({"split": split, "label_classification": label,
                                   **subset_metrics(pred[mask], target[mask], sid[mask])})
    write_csv(HERE / "category_metrics.csv", category_rows)
    write_csv(HERE / "label_ambiguity_metrics.csv", ambiguity_rows)

    test_indices = indices_by_split["test"]
    test_target = y[test_indices]
    test_prediction = best_predictions["test"]
    test_zero_mask = np.linalg.norm(test_target, axis=1) <= 1e-14
    zero_norms = np.linalg.norm(test_prediction[test_zero_mask], axis=1)
    nonzero_errors = np.linalg.norm(
        test_prediction[~test_zero_mask] - test_target[~test_zero_mask], axis=1)
    nonzero_norms = np.linalg.norm(test_target[~test_zero_mask], axis=1)
    write_json(HERE / "zero_label_metrics.json", {
        "split": "test", "state_count": len(set(arrays["state_id"][test_indices][test_zero_mask].tolist())),
        "sample_count": int(test_zero_mask.sum()), "semantics": "||G_phi(x)||_2 on exact-zero oracle labels",
        "mean": float(np.mean(zero_norms)), "median": float(np.median(zero_norms)),
        "p95": float(np.quantile(zero_norms, 0.95)), "max": float(np.max(zero_norms)),
        "mean_over_vmax": float(np.mean(zero_norms) / VMAX),
    })
    write_json(HERE / "nonzero_label_metrics.json", {
        "split": "test", "state_count": len(set(arrays["state_id"][test_indices][~test_zero_mask].tolist())),
        "sample_count": int((~test_zero_mask).sum()), "semantics": "L2 error to nonzero g*_exec",
        "mean_l2_error": float(np.mean(nonzero_errors)), "median_l2_error": float(np.median(nonzero_errors)),
        "p95_l2_error": float(np.quantile(nonzero_errors, 0.95)), "max_l2_error": float(np.max(nonzero_errors)),
        "mean_relative_error": float(np.mean(nonzero_errors / nonzero_norms)),
        "median_relative_error": float(np.median(nonzero_errors / nonzero_norms)),
        "mean_target_norm": float(np.mean(nonzero_norms)),
    })

    protocol = read_json(DATA / "protocol.json")
    projection = projection_replay(best_predictions, arrays, indices_by_split, protocol)
    write_json(HERE / "projection_replay_metrics.json", projection)
    nearest = nearest_neighbor_audit(x_norm, arrays, indices_by_split, best_predictions)

    # Required baseline comparison on the held-out test split.
    test_comparison = {}
    for name in ("ZERO", "TRAIN_MEAN", "LINEAR"):
        test_comparison[name] = subset_metrics(
            model_predictions[name]["test"], split_data["test"][1], split_data["test"][2])
    test_comparison[best_name] = best_metrics["test"]
    # Add selected baseline test rows only after model selection is frozen.
    for row in comparison:
        if row["model"] in ("ZERO", "TRAIN_MEAN", "LINEAR"):
            row["test_opened_after_selection"] = True
    write_csv(HERE / "model_comparison.csv", comparison)

    train_gap = best_metrics["validation"]["state_grouped_mse"] / max(
        best_metrics["train"]["state_grouped_mse"], EPS)
    test_gap = best_metrics["test"]["state_grouped_mse"] / max(
        best_metrics["train"]["state_grouped_mse"], EPS)
    zero_test = next(row for row in category_rows
                     if row["split"] == "test" and row["group"] == "zero_label")
    nonzero_test = next(row for row in category_rows
                        if row["split"] == "test" and row["group"] == "nonzero_label")
    nonlinear_beats = (
        nonzero_test["state_grouped_mean_l2"]
        < min(test_comparison["ZERO"]["state_grouped_mean_l2"],
              test_comparison["TRAIN_MEAN"]["state_grouped_mean_l2"])
    )
    projection_not_rewriting_most = (
        projection["test"]["projection_changed_fraction_over_1e-6"] < 0.5
        or projection["test"]["projection_rewrite_norm"]["mean"]
        < best_metrics["test"]["state_grouped_mean_l2"]
    )
    finite = all(np.isfinite(best_predictions[split]).all() for split in best_predictions)
    severe_memorization = bool(test_gap > 5.0 and train_gap > 3.0)
    false_intervention_reasonable = bool(zero_test["false_intervention_mean_norm"] < 0.1 * VMAX)
    v1 = {
        "train_state_grouped_mean_l2": 0.00301638,
        "validation_state_grouped_mean_l2": 0.0458845,
        "test_state_grouped_mean_l2": 0.0208517,
        "zero_label_false_intervention": 0.0349482,
        "nonzero_label_error": 0.0177192,
    }
    v1_val_gap = v1["validation_state_grouped_mean_l2"] - v1["train_state_grouped_mean_l2"]
    v2_val_gap = best_metrics["validation"]["state_grouped_mean_l2"] - best_metrics["train"]["state_grouped_mean_l2"]
    v1_test_gap = v1["test_state_grouped_mean_l2"] - v1["train_state_grouped_mean_l2"]
    v2_test_gap = best_metrics["test"]["state_grouped_mean_l2"] - best_metrics["train"]["state_grouped_mean_l2"]
    zero_reduction = 1.0 - zero_test["false_intervention_mean_norm"] / v1["zero_label_false_intervention"]
    scaling_validation_reduction = (
        1.0 - scaling_rows[-1]["validation_state_grouped_mean_l2"]
        / max(scaling_rows[0]["validation_state_grouped_mean_l2"], EPS)
    )
    material_gap_improvement = v2_val_gap <= 0.75 * v1_val_gap
    if (
        zero_reduction >= 0.30 and material_gap_improvement
        and best_metrics["test"]["state_grouped_mean_l2"] <= 1.10 * v1["test_state_grouped_mean_l2"]
    ):
        coverage_conclusion = "COVERAGE_FIXES_GENERALIZATION"
    elif scaling_validation_reduction >= 0.15 and (
        zero_reduction > 0.10 or v2_val_gap < v1_val_gap
    ):
        coverage_conclusion = "STILL_DATA_LIMITED"
    else:
        coverage_conclusion = "COVERAGE_NOT_THE_MAIN_PROBLEM"
    acceptance_evidence = {
        "nonlinear_materially_beats_zero_and_mean_on_nonzero_test_states": bool(nonlinear_beats),
        "test_nonzero_state_grouped_mean_l2": nonzero_test["state_grouped_mean_l2"],
        "test_nonzero_mean_target_relative_error": nonzero_test["mean_relative_l2_nonzero"],
        "test_zero_false_intervention_mean_norm": zero_test["false_intervention_mean_norm"],
        "test_zero_false_intervention_over_vmax": zero_test["false_intervention_mean_norm"] / VMAX,
        "false_intervention_scale_reasonable": false_intervention_reasonable,
        "train_to_validation_state_mse_ratio": train_gap,
        "train_to_test_state_mse_ratio": test_gap,
        "severe_memorization_signal": severe_memorization,
        "projection_does_not_completely_rewrite_most_outputs": bool(projection_not_rewriting_most),
        "finite_predictions": finite,
        "projection_failures": len(projection["validation"]["projection_failures"])
                               + len(projection["test"]["projection_failures"]),
        "v1_reference": v1,
        "zero_false_intervention_fractional_reduction_vs_v1": zero_reduction,
        "validation_mean_l2_gap_v1": v1_val_gap,
        "validation_mean_l2_gap_v2": v2_val_gap,
        "test_mean_l2_gap_v1": v1_test_gap,
        "test_mean_l2_gap_v2": v2_test_gap,
        "scaling_validation_fractional_reduction_smallest_to_full": scaling_validation_reduction,
        "coverage_conclusion": coverage_conclusion,
        "decision_note": "Evidence-based pilot gate; numeric comparisons are reported rather than presented as a theorem.",
    }
    ready_for_closed_loop = bool(
        coverage_conclusion == "COVERAGE_FIXES_GENERALIZATION"
        and zero_test["false_intervention_mean_norm"] <= 0.5 * v1["zero_label_false_intervention"]
        and best_metrics["test"]["state_grouped_mean_l2"] <= v1["test_state_grouped_mean_l2"]
        and nonlinear_beats and projection_not_rewriting_most and finite
        and acceptance_evidence["projection_failures"] == 0
    )
    readiness = (
        "READY_FOR_CLOSED_LOOP_PILOT"
        if ready_for_closed_loop
        else "TRAINING_NEEDS_REVISION"
    )

    test_metrics = {
        "selected_model": best_name, "selection_used_test": False,
        "selected_from_validation_seed": best["seed"],
        "metrics_by_split": best_metrics,
        "required_test_baseline_comparison": test_comparison,
        "overfitting_and_nearest_neighbor_audit": nearest,
        "acceptance_evidence": acceptance_evidence,
        "coverage_conclusion": coverage_conclusion,
        "ready_for_closed_loop_pilot": ready_for_closed_loop,
        "classification": readiness,
    }
    write_json(HERE / "test_metrics.json", test_metrics)

    config_output = {
        "task": "deterministic G_phi supervised pilot",
        "input_dimension": 214, "output_dimension": 4,
        "target": "g*_exec = u*_exec - u_safe",
        "primary_loss": "mean squared error to g*_exec",
        "selected_architecture": best_config["hidden"],
        "activation": "SiLU", "output_activation": "identity",
        "selected_training_seed": best["seed"],
        "final_architecture_seeds": [17, 23, 41],
        "optimizer": "AdamW", "learning_rate": best_config["learning_rate"],
        "weight_decay": best_config["weight_decay"],
        "batch_size": best_config["batch_size"], "early_stopping": {
            "metric": "validation state-grouped MSE", "patience_evaluations": best_config["patience_evaluations"],
            "eval_interval_epochs": best_config["eval_interval"], "best_epoch": best["summary"]["best_epoch"],
        },
        "model_selection": "validation only; test opened after architecture/hyperparameter/seed frozen",
        "parameter_count": best["summary"]["parameter_count"],
        "dataset_manifest_sha256": sha(DATA / "manifest.json"),
        "coverage_conclusion": coverage_conclusion,
        "ready_for_closed_loop_pilot": ready_for_closed_loop,
    }
    write_json(HERE / "config.json", config_output)
    (HERE / "model_summary.txt").write_text(
        "Deterministic G_phi\n"
        f"Architecture: 214 -> {' -> '.join(map(str, best_config['hidden']))} -> 4\n"
        "Hidden activation: SiLU; output: linear\n"
        f"Parameters: {best['summary']['parameter_count']}\n"
        "Primary loss: direct MSE to four-dimensional executed correction g*_exec\n"
        f"Selected seed: {best['seed']} (validation state-grouped MSE only)\n"
    )

    finished_utc = datetime.now(timezone.utc).isoformat()
    runtime = {
        "started_utc": started_utc, "finished_utc": finished_utc,
        "wall_clock_seconds": time.monotonic() - started,
        "screening_and_final_run_seconds_sum": float(sum(
            row["runtime_s"] for row in screening_summaries)
            + sum(run["summary"]["runtime_s"] for run in final_runs if run["seed"] != 17)),
        "jax_devices": [str(device) for device in jax.devices()],
        "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "xla_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "xla_memory_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "cpu_count_visible": os.cpu_count(),
        "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "observed_gpu_process_memory_mib": 720,
        "observed_gpu_memory_source": "nvidia-smi sample during Slurm training job",
        "python": sys.version, "platform": platform.platform(),
        "jax": jax.__version__, "flax": flax.__version__, "optax": optax.__version__,
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    dataset_hash_after = sha(DATA / "samples.npz")
    frozen_paths = {
        "environment": SYSROOT / "single_integrator" / "environment.py",
        "projection": SYSROOT / "single_integrator" / "cbf.py",
        "corrector": ROOT / "diagnostics" / "cl_fhcb" / "closed_loop.py",
        "retry": ROOT / "diagnostics" / "success_basin_multimodality" / "exact_projector.py",
        "checkpoint": Path(protocol["checkpoint"]),
    }
    frozen_actual = {key: sha(path) for key, path in frozen_paths.items()}
    frozen_expected = protocol["frozen_hashes"]
    frozen_unchanged = frozen_actual == frozen_expected
    sanity_final = dict(audit)
    sanity_final.update({
        "dataset_samples_sha256_after_training": dataset_hash_after,
        "dataset_unchanged": dataset_hash_after == audit["samples_sha256_actual"],
        "frozen_hashes_expected": frozen_expected,
        "frozen_hashes_after": frozen_actual,
        "frozen_hashes_unchanged": frozen_unchanged,
        "prediction_all_finite": finite,
        "checkpoint_exists": (HERE / "best_checkpoint.npz").is_file(),
        "test_not_used_for_selection": True,
    })
    write_json(HERE / "sanity_checks.json", sanity_final)

    report = f"""# Deterministic G_phi pilot training v2

## Coverage diagnosis

**{coverage_conclusion}**

Closed-loop pilot readiness: **{readiness}**.

The selected model predicts the four-dimensional instantaneous executed correction
`g*_exec = u*_exec - u_safe` directly.  It does not predict eta, a stochastic
distribution, or a trajectory.  No frozen controller, physics, event, oracle, or
dataset artifact was modified.

## Selected model

- Architecture: `214 -> {' -> '.join(map(str, best_config['hidden']))} -> 4`, SiLU hidden activation.
- Parameters: {best['summary']['parameter_count']}.
- Optimizer: AdamW, learning rate {best_config['learning_rate']}, weight decay {best_config['weight_decay']}.
- Selected training seed: {best['seed']} from seeds 17/23/41, using validation state-grouped MSE only.
- Checkpoint: `best_checkpoint.npz`.

## Held-out state results

| Split | state-grouped RMSE | state-grouped mean L2 | sample mean L2 |
|---|---:|---:|---:|
| Train | {best_metrics['train']['state_grouped_rmse']:.6g} | {best_metrics['train']['state_grouped_mean_l2']:.6g} | {best_metrics['train']['mean_l2']:.6g} |
| Validation | {best_metrics['validation']['state_grouped_rmse']:.6g} | {best_metrics['validation']['state_grouped_mean_l2']:.6g} | {best_metrics['validation']['mean_l2']:.6g} |
| Test | {best_metrics['test']['state_grouped_rmse']:.6g} | {best_metrics['test']['state_grouped_mean_l2']:.6g} | {best_metrics['test']['mean_l2']:.6g} |

Test nonzero-label state-grouped mean L2 is {nonzero_test['state_grouped_mean_l2']:.6g} m/s.
Test zero-label false intervention mean norm is {zero_test['false_intervention_mean_norm']:.6g} m/s
({zero_test['false_intervention_mean_norm']/VMAX:.3%} of vmax).

## Direct V1 comparison

| Metric | V1 | V2 | V2 - V1 |
|---|---:|---:|---:|
| Train state-grouped mean L2 | {v1['train_state_grouped_mean_l2']:.6g} | {best_metrics['train']['state_grouped_mean_l2']:.6g} | {best_metrics['train']['state_grouped_mean_l2']-v1['train_state_grouped_mean_l2']:+.6g} |
| Validation state-grouped mean L2 | {v1['validation_state_grouped_mean_l2']:.6g} | {best_metrics['validation']['state_grouped_mean_l2']:.6g} | {best_metrics['validation']['state_grouped_mean_l2']-v1['validation_state_grouped_mean_l2']:+.6g} |
| Test state-grouped mean L2 | {v1['test_state_grouped_mean_l2']:.6g} | {best_metrics['test']['state_grouped_mean_l2']:.6g} | {best_metrics['test']['state_grouped_mean_l2']-v1['test_state_grouped_mean_l2']:+.6g} |
| Zero-label false intervention | {v1['zero_label_false_intervention']:.6g} | {zero_test['false_intervention_mean_norm']:.6g} | {zero_test['false_intervention_mean_norm']-v1['zero_label_false_intervention']:+.6g} |
| Nonzero-label error | {v1['nonzero_label_error']:.6g} | {nonzero_test['state_grouped_mean_l2']:.6g} | {nonzero_test['state_grouped_mean_l2']-v1['nonzero_label_error']:+.6g} |

The fixed-seed scaling diagnostic changed validation state-grouped mean L2 from
{scaling_rows[0]['validation_state_grouped_mean_l2']:.6g} at {scaling_rows[0]['independent_training_states']}
training states to {scaling_rows[-1]['validation_state_grouped_mean_l2']:.6g} at
{scaling_rows[-1]['independent_training_states']} training states.

## Required baselines on test

| Model | state-grouped mean L2 | state-grouped RMSE |
|---|---:|---:|
| ZERO | {test_comparison['ZERO']['state_grouped_mean_l2']:.6g} | {test_comparison['ZERO']['state_grouped_rmse']:.6g} |
| TRAIN-MEAN | {test_comparison['TRAIN_MEAN']['state_grouped_mean_l2']:.6g} | {test_comparison['TRAIN_MEAN']['state_grouped_rmse']:.6g} |
| Linear | {test_comparison['LINEAR']['state_grouped_mean_l2']:.6g} | {test_comparison['LINEAR']['state_grouped_rmse']:.6g} |
| Selected MLP | {best_metrics['test']['state_grouped_mean_l2']:.6g} | {best_metrics['test']['state_grouped_rmse']:.6g} |

## Projection replay

On test samples, mean post-projection executed-action error is
{projection['test']['executed_action_error']['mean']:.6g} m/s.  The unchanged
second projection changes the raw predicted action by a mean of
{projection['test']['projection_rewrite_norm']['mean']:.6g} m/s; the fraction
changed above 1e-6 is {projection['test']['projection_changed_fraction_over_1e-6']:.3%}.
Projection failures: {len(projection['test']['projection_failures'])}.

## Generalization cautions

The {len(arrays['targets']):,} samples come from {len(set(arrays['state_id'].tolist()))} independent augmented states.  Splits remain
source-state/source-trajectory grouped.  Train/validation and train/test
state-MSE ratios are {train_gap:.3g} and {test_gap:.3g}.  Nearest-state distances
and output-collapse diagnostics are recorded in `test_metrics.json`; category,
zero/nonzero, and label-ambiguity results are in their dedicated CSV files.

This is an offline supervised pilot.  No large closed-loop benchmark was run.
"""
    (HERE / "training_report.md").write_text(report)

    required_outputs = (
        "training_report.md", "config.json", "model_summary.txt", "normalization.json",
        "training_history.csv", "model_comparison.csv", "test_metrics.json",
        "state_grouped_metrics.csv", "category_metrics.csv", "label_ambiguity_metrics.csv",
        "zero_label_metrics.json", "nonzero_label_metrics.json", "scaling_analysis.csv",
        "projection_replay_metrics.json", "best_checkpoint.npz", "sanity_checks.json",
        "runtime_statistics.json",
    )
    manifest_output = {
        "study": "deterministic G_phi state-coverage retraining v2",
        "generated_at_utc": finished_utc, "classification": readiness,
        "training_performed": True, "large_closed_loop_benchmark_performed": False,
        "dataset_directory": str(DATA), "dataset_manifest_sha256": sha(DATA / "manifest.json"),
        "dataset_samples_sha256": dataset_hash_after,
        "git": git_revision(), "resource_policy": "oracle used two GPU shards; training used one capped GPU process",
        "coverage_conclusion": coverage_conclusion,
        "ready_for_closed_loop_pilot": ready_for_closed_loop,
        "selected_checkpoint": "best_checkpoint.npz",
        "files_sha256": {name: sha(HERE / name) for name in required_outputs},
        "frozen_hashes_unchanged": frozen_unchanged,
    }
    write_json(HERE / "manifest.json", manifest_output)
    print(json.dumps({
        "classification": readiness, "architecture": best_config["hidden"],
        "parameters": best["summary"]["parameter_count"], "seed": best["seed"],
        "validation_state_grouped_mean_l2": best_metrics["validation"]["state_grouped_mean_l2"],
        "test_state_grouped_mean_l2": best_metrics["test"]["state_grouped_mean_l2"],
        "runtime_seconds": runtime["wall_clock_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
