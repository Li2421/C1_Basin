"""Merge, train, and offline-audit startup-complete deterministic G_phi.

The two source datasets are treated as immutable.  All generated data and
reports are written beneath this diagnostic's ``artifacts`` directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V3 = ROOT / "diagnostics/gphi_training_dataset_v3"
STARTUP = ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1"
PREVIOUS_TRAINING = ROOT / "diagnostics/gphi_pilot_training_v3"
DEFAULT_OUTPUT = HERE / "artifacts"
REQUIRED_FILES = (
    "samples.npz", "state_manifest.jsonl", "sample_metadata.jsonl",
    "feature_schema.json", "protocol.json", "manifest.json",
)
SPLITS = ("train", "validation", "test")
EPS = 1e-12
VMAX = 0.5

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import diagnostics.gphi_pilot_training_v2.train_and_evaluate as core


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def assert_fresh_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(
            f"refusing existing output path (prevents stale/partial result mixing): {path}")


def dataset_file_hashes(path: Path) -> dict[str, str]:
    return {name: sha256(path / name) for name in REQUIRED_FILES if (path / name).exists()}


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.asarray(source[key]).copy() for key in source.files}


def validate_sample_alignment(arrays: dict[str, np.ndarray], states: list[dict],
                              metadata: list[dict], dataset_name: str) -> dict:
    """Verify sample arrays, metadata, and state rows agree exactly."""
    state_by_id = {str(row["state_id"]): row for row in states}
    failures = []
    for index, meta in enumerate(metadata):
        state_id = str(arrays["state_id"][index])
        state = state_by_id[state_id]
        checks = {
            "sample_id": (str(arrays["sample_id"][index]), str(meta.get("sample_id"))),
            "state_id": (state_id, str(meta.get("state_id"))),
            "split_metadata": (str(arrays["split"][index]), str(meta.get("split"))),
            "category_metadata": (str(arrays["category"][index]), str(meta.get("category"))),
            "state_index_metadata": (int(arrays["state_index"][index]), int(meta.get("state_index", -1))),
            "split_state": (str(arrays["split"][index]), str(state.get("split"))),
            "category_state": (str(arrays["category"][index]), str(state.get("category"))),
            "state_index_state": (int(arrays["state_index"][index]), int(state.get("state_index", -1))),
        }
        mismatch = {name: values for name, values in checks.items() if values[0] != values[1]}
        if mismatch:
            failures.append({"index": index, "sample_id": str(arrays["sample_id"][index]),
                             "mismatch": mismatch})
            if len(failures) >= 10:
                break
    if failures:
        raise ValueError(f"{dataset_name}: NPZ/metadata/state alignment failed: {failures}")
    return {
        "dataset": dataset_name, "sample_count": len(metadata),
        "fields_checked": ["sample_id", "state_id", "split", "category", "state_index"],
        "passed": True,
    }


def _split_counts(n: int, ratios: dict[str, float]) -> dict[str, int]:
    if n < 3:
        raise ValueError("startup dataset needs at least three source episodes")
    raw = {split: n * ratios[split] for split in SPLITS}
    counts = {split: int(math.floor(raw[split])) for split in SPLITS}
    for split in sorted(SPLITS, key=lambda item: (raw[item] - counts[item], item), reverse=True)[:n - sum(counts.values())]:
        counts[split] += 1
    for split in SPLITS:
        if counts[split] == 0:
            donor = max(SPLITS, key=lambda item: counts[item])
            counts[donor] -= 1
            counts[split] += 1
    assert sum(counts.values()) == n and all(counts[split] > 0 for split in SPLITS)
    return counts


def deterministic_episode_split(episodes: list[str], seed: int,
                                ratios: dict[str, float],
                                fixed: dict[str, str] | None = None) -> dict[str, str]:
    unique = sorted(set(episodes), key=lambda value: hashlib.sha256(
        f"{seed}:{value}".encode()).hexdigest())
    target = _split_counts(len(unique), ratios)
    fixed = fixed or {}
    assignment = {episode: fixed[episode] for episode in unique if episode in fixed}
    counts = Counter(assignment.values())
    for episode in (value for value in unique if value not in assignment):
        # Fill target deficits first.  If inherited V3 memberships already
        # exceed a target, choose the split with the smallest normalized
        # post-assignment excess.  Hash ordering makes ties deterministic.
        deficits = {split: target[split] - counts[split] for split in SPLITS}
        positive = [split for split in SPLITS if deficits[split] > 0]
        if positive:
            chosen = max(positive, key=lambda split: (deficits[split], -SPLITS.index(split)))
        else:
            chosen = min(SPLITS, key=lambda split: (
                (counts[split] + 1 - target[split]) / max(target[split], 1),
                SPLITS.index(split)))
        assignment[episode] = chosen
        counts[chosen] += 1
    return assignment


def _manifest_hash(manifest: dict, name: str) -> str | None:
    hashes = manifest.get("files_sha256", {})
    return hashes.get(name) or manifest.get(f"{name}_sha256")


def audit_source_dataset(path: Path, *, require_source_episode: bool) -> tuple[
        dict[str, np.ndarray], list[dict], list[dict], dict, dict, dict]:
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{path}: missing required files {missing}")
    manifest = read_json(path / "manifest.json")
    schema = read_json(path / "feature_schema.json")
    protocol = read_json(path / "protocol.json")
    states = read_jsonl(path / "state_manifest.jsonl")
    metadata = read_jsonl(path / "sample_metadata.jsonl")
    arrays = load_npz(path / "samples.npz")
    expected_hash = _manifest_hash(manifest, "samples.npz")
    actual_hash = sha256(path / "samples.npz")
    if not expected_hash or expected_hash != actual_hash:
        raise ValueError(f"{path}: samples.npz manifest hash mismatch")
    required_arrays = {
        "features", "targets", "target_actions", "u_safe", "positions",
        "state_id", "state_index", "sample_id", "split", "category",
    }
    if not required_arrays.issubset(arrays):
        raise ValueError(f"{path}: missing arrays {sorted(required_arrays - set(arrays))}")
    n = len(arrays["features"])
    if arrays["features"].shape != (n, 214) or arrays["targets"].shape != (n, 4):
        raise ValueError(f"{path}: expected features/targets shaped N x 214 and N x 4")
    if any(len(value) != n for value in arrays.values()):
        raise ValueError(f"{path}: inconsistent sample-array lengths")
    if len(metadata) != n:
        raise ValueError(f"{path}: metadata length {len(metadata)} != {n}")
    if not np.isfinite(arrays["features"]).all() or not np.isfinite(arrays["targets"]).all():
        raise ValueError(f"{path}: nonfinite features or targets")
    if not np.allclose(arrays["target_actions"], arrays["u_safe"].reshape(-1, 4) + arrays["targets"], atol=1e-12, rtol=0):
        raise ValueError(f"{path}: target_actions != u_safe + targets")
    if schema.get("feature_dimension") != 214 or schema.get("target_dimension") != 4:
        raise ValueError(f"{path}: schema is not 214 -> 4")
    if len(set(arrays["sample_id"].tolist())) != n:
        raise ValueError(f"{path}: duplicate sample IDs")
    metadata_ids = np.asarray([str(row["sample_id"]) for row in metadata])
    if not np.array_equal(metadata_ids, arrays["sample_id"]):
        raise ValueError(f"{path}: metadata order differs from samples.npz")
    state_ids = set(arrays["state_id"].tolist())
    manifest_state_ids = [str(row["state_id"]) for row in states]
    if len(manifest_state_ids) != len(set(manifest_state_ids)):
        raise ValueError(f"{path}: duplicate state-manifest IDs")
    if state_ids != set(manifest_state_ids):
        raise ValueError(f"{path}: state manifest/NPZ state IDs differ")
    if not np.array_equal(np.asarray([str(row["state_id"]) for row in metadata]), arrays["state_id"]):
        raise ValueError(f"{path}: metadata state order differs from samples.npz")
    if require_source_episode:
        required_state_fields = {"state_id", "source_episode", "leakage_group", "category"}
        incomplete = [row.get("state_id") for row in states if not required_state_fields.issubset(row)]
        if incomplete:
            raise ValueError(f"{path}: startup state rows missing required fields: {incomplete[:5]}")
        missing_episode = [row.get("state_id") for row in states if not row.get("source_episode")]
        if missing_episode:
            raise ValueError(f"{path}: startup states missing source_episode: {missing_episode[:5]}")
    validate_sample_alignment(arrays, states, metadata, str(path))
    return arrays, states, metadata, schema, protocol, manifest


def assert_compatible(v3_arrays: dict[str, np.ndarray], startup_arrays: dict[str, np.ndarray],
                      v3_schema: dict, startup_schema: dict,
                      v3_protocol: dict, startup_protocol: dict) -> None:
    if v3_schema != startup_schema:
        raise ValueError("startup feature_schema.json differs from V3")
    if v3_protocol.get("environment") != startup_protocol.get("environment"):
        raise ValueError("startup frozen environment protocol differs from V3")
    v3_hashes = v3_protocol.get("frozen_hashes", {})
    startup_hashes = startup_protocol.get("frozen_hashes", {})
    for key in ("environment", "projection", "retry", "checkpoint"):
        if not v3_hashes.get(key) or v3_hashes.get(key) != startup_hashes.get(key):
            raise ValueError(f"startup frozen {key} hash differs from V3")
    if set(v3_arrays) != set(startup_arrays):
        raise ValueError("startup samples.npz keys differ from V3")
    for key in v3_arrays:
        if v3_arrays[key].shape[1:] != startup_arrays[key].shape[1:]:
            raise ValueError(f"startup trailing shape differs for {key}")
        if v3_arrays[key].dtype.kind != startup_arrays[key].dtype.kind:
            raise ValueError(f"startup dtype kind differs for {key}")


def assert_v3_prefix_preserved(v3: dict[str, np.ndarray], merged: dict[str, np.ndarray]) -> None:
    n = len(v3["features"])
    for key, expected in v3.items():
        actual = merged[key][:n]
        if expected.dtype.kind in "biufc":
            # Compare in the original dtype so numeric sample bytes are exact.
            if actual.astype(expected.dtype, copy=False).tobytes() != expected.tobytes():
                raise AssertionError(f"V3 numeric prefix changed: {key}")
        elif not np.array_equal(actual, expected):
            raise AssertionError(f"V3 value prefix changed: {key}")


def prepare_merged(v3_path: Path, startup_path: Path, output: Path,
                   split_seed: int, ratios: dict[str, float]) -> tuple[dict, list, list, dict, dict]:
    before = {"v3": dataset_file_hashes(v3_path), "startup": dataset_file_hashes(startup_path)}
    v3, v3_states, v3_meta, schema, protocol, v3_manifest = audit_source_dataset(
        v3_path, require_source_episode=False)
    startup, startup_states, startup_meta, startup_schema, startup_protocol, startup_manifest = audit_source_dataset(
        startup_path, require_source_episode=False)
    assert_compatible(v3, startup, schema, startup_schema, protocol, startup_protocol)

    # The authoritative startup dataset is itself a V3-prefix merge.  Consume
    # it without appending V3 a second time, and independently prove that every
    # old sample survived unchanged.
    if len(startup["features"]) <= len(v3["features"]):
        raise ValueError("startup-complete dataset does not extend V3")
    assert_v3_prefix_preserved(v3, startup)
    if startup_states[:len(v3_states)] != v3_states:
        raise ValueError("startup-complete state_manifest V3 prefix is not exact")
    if startup_meta[:len(v3_meta)] != v3_meta:
        raise ValueError("startup-complete sample_metadata V3 prefix is not exact")
    if startup["state_id"][:len(v3["state_id"])].tolist() != v3["state_id"].tolist():
        raise ValueError("startup-complete dataset does not have the exact V3 state/sample prefix")
    v3_state_ids = set(v3["state_id"].tolist())
    added_states = [dict(row) for row in startup_states if str(row["state_id"]) not in v3_state_ids]
    if not added_states:
        raise ValueError("startup-complete dataset has no added startup states")
    required = {"state_id", "source_episode", "leakage_group", "category", "split"}
    incomplete = [row.get("state_id") for row in added_states if not required.issubset(row)]
    if incomplete:
        raise ValueError(f"startup state rows missing required fields: {incomplete[:5]}")
    if any(str(row["category"]) != "STARTUP" for row in added_states):
        raise ValueError("added startup state category must be STARTUP")
    added_ids = {str(row["state_id"]) for row in added_states}
    if set(startup["state_id"][len(v3["state_id"]):].tolist()) != added_ids:
        raise ValueError("startup sample suffix does not match added startup state manifest")

    merged = {key: value.copy() for key, value in startup.items()}
    final_states = [dict(row, dataset_origin=("V3" if str(row["state_id"]) in v3_state_ids else "STARTUP"))
                    for row in startup_states]
    final_meta = [dict(row, dataset_origin=("V3" if str(row["state_id"]) in v3_state_ids else "STARTUP"))
                  for row in startup_meta]

    # No trajectory/group may straddle the resulting split.  Startup episodes
    # already present in V3 inherit their V3 membership rather than leaking.
    group_splits = defaultdict(set)
    episode_splits = defaultdict(set)
    for row in final_states:
        if row.get("leakage_group"):
            group_splits[str(row["leakage_group"])].add(str(row["split"]))
        episode = str(row.get("source_episode") or row.get("source_trajectory") or "")
        if episode:
            episode_splits[episode].add(str(row["split"]))
    leaking_groups = {key: sorted(value) for key, value in group_splits.items() if len(value) > 1}
    leaking_episodes = {key: sorted(value) for key, value in episode_splits.items() if len(value) > 1}
    if leaking_groups or leaking_episodes:
        raise ValueError(f"merged source leakage: groups={leaking_groups}, episodes={leaking_episodes}")

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "merged_samples.npz", **merged)
    write_jsonl(output / "merged_state_manifest.jsonl", final_states)
    write_jsonl(output / "merged_sample_metadata.jsonl", final_meta)
    assignment = {str(row["source_episode"]): str(row["split"]) for row in added_states}
    episode_rows = [{"source_episode": episode, "split": assignment[episode]}
                    for episode in sorted(assignment)]
    write_csv(output / "startup_episode_split.csv", episode_rows)
    split_manifest = {
        "assignment_unit": "startup source_episode; assignments frozen by startup dataset; V3 memberships unchanged",
        "split_seed": split_seed, "requested_ratios": ratios,
        "startup_source_episode_counts": dict(Counter(assignment.values())),
        "startup_source_episode_realized_ratios": {
            split: list(assignment.values()).count(split) / len(assignment) for split in SPLITS},
        "startup_state_counts": dict(Counter(str(row["split"]) for row in added_states)),
        "startup_sample_counts": dict(Counter(
            merged["split"][np.isin(merged["state_id"], list(added_ids))].tolist())),
        "source_episode_assignment": assignment,
    }
    write_json(output / "frozen_split_manifest.json", split_manifest)

    after = {"v3": dataset_file_hashes(v3_path), "startup": dataset_file_hashes(startup_path)}
    if before != after:
        raise RuntimeError("an input dataset changed while preparing the merge")
    audit = {
        "passed": True,
        "input_hashes_before": before, "input_hashes_after": after,
        "v3_samples_sha256": sha256(v3_path / "samples.npz"),
        "startup_samples_sha256": sha256(startup_path / "samples.npz"),
        "v3_prefix_numeric_bytes_preserved": True,
        "v3_prefix_string_values_preserved": True,
        "v3_state_manifest_prefix_exact": True,
        "v3_sample_metadata_prefix_exact": True,
        "npz_metadata_state_alignment_passed": True,
        "v3_sample_count": len(v3["features"]),
        "startup_sample_count": len(startup["features"]) - len(v3["features"]),
        "merged_sample_count": len(merged["features"]),
        "v3_state_count": len(v3_states),
        "startup_state_count": len(added_states),
        "state_id_overlap": 0, "sample_id_overlap": 0,
        "merged_split_leaking_groups": 0, "merged_split_leaking_source_episodes": 0,
        "feature_schema_sha256": sha256(v3_path / "feature_schema.json"),
        "source_manifests": {"v3": v3_manifest, "startup": startup_manifest},
    }
    write_json(output / "merge_audit.json", audit)
    return merged, final_states, final_meta, schema, protocol


def load_checkpoint(path: Path):
    with np.load(path, allow_pickle=False) as data:
        indices = sorted({int(key.split("_")[1]) for key in data.files
                          if key.startswith("layer_") and key.endswith("_weight")})
        params = [{"w": np.asarray(data[f"layer_{i}_weight"]),
                   "b": np.asarray(data[f"layer_{i}_bias"])} for i in indices]
        return params, np.asarray(data["normalization_mean"]), np.asarray(data["normalization_scale"])


def numpy_predict_checkpoint(path: Path, features: np.ndarray) -> np.ndarray:
    params, mean, scale = load_checkpoint(path)
    value = ((features - mean) / scale).astype(np.float32)
    for index, layer in enumerate(params):
        value = value @ layer["w"] + layer["b"]
        if index < len(params) - 1:
            value = value / (1.0 + np.exp(-value))
    return np.asarray(value)


def metric_rows(prediction: np.ndarray, arrays: dict, metadata_by_sample: dict,
                indices_by_split: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    n_v3 = sum(row.get("dataset_origin") == "V3" for row in metadata_by_sample.values())
    origin = np.asarray(["WARM_V3"] * n_v3 + ["STARTUP"] * (len(arrays["features"]) - n_v3))
    for split, indices in indices_by_split.items():
        groups = {"ALL": np.ones(len(indices), dtype=bool)}
        for cohort in ("WARM_V3", "STARTUP"):
            groups[cohort] = origin[indices] == cohort
        for category in sorted(set(arrays["category"][indices].tolist())):
            groups[f"CATEGORY:{category}"] = arrays["category"][indices] == category
        for name, mask in groups.items():
            if not np.any(mask):
                continue
            chosen = indices[mask]
            rows.append({"split": split, "subset": name,
                         **core.subset_metrics(prediction[split][mask], arrays["targets"][chosen],
                                               arrays["state_id"][chosen])})
    return rows


def run_training(arrays: dict, states: list[dict], metadata: list[dict], schema: dict,
                 protocol: dict, output: Path, config: dict, v3_count: int) -> None:
    # ML imports occur only after all merge gates have passed.
    import jax

    indices = {split: np.flatnonzero(arrays["split"] == split) for split in SPLITS}
    for split in SPLITS:
        if not len(indices[split]):
            raise ValueError(f"empty merged split: {split}")
    mean, scale, binary = core.fit_normalization(arrays["features"][indices["train"]], schema)
    x = core.normalize(arrays["features"], mean, scale).astype(np.float32)
    y = arrays["targets"].astype(np.float32)
    write_json(output / "normalization.json", {
        "fit_split": "merged train only", "mean": mean.tolist(), "scale": scale.tolist(),
        "binary_feature_indices": np.flatnonzero(binary).tolist(),
    })

    base = {
        "hidden": [128, 128], "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"], "batch_size": config["batch_size"],
        "max_epochs": config["early_stopping"]["max_epochs"],
        "eval_interval": config["early_stopping"]["eval_interval_epochs"],
        "patience_evaluations": config["early_stopping"]["patience_evaluations"],
        "min_delta": config["early_stopping"]["min_delta"],
    }
    histories, runs = [], []
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    train = indices["train"]
    validation = indices["validation"]
    for seed in config["training_seeds"]:
        run_config = {**base, "name": f"startup_complete_seed{seed}", "seed": seed}
        params, history, summary = core.train_mlp(
            x[train], y[train], arrays["state_id"][train],
            x[validation], y[validation], arrays["state_id"][validation], run_config)
        histories.extend(history)
        validation_prediction = core.predict_mlp(params, x[validation])
        validation_metrics = core.subset_metrics(
            validation_prediction, y[validation], arrays["state_id"][validation])
        core.save_checkpoint(checkpoint_dir / f"seed{seed}.npz", params, run_config, mean, scale, binary)
        runs.append({"seed": seed, "params": params, "config": run_config,
                     "summary": summary, "validation_metrics": validation_metrics})
    selected = min(runs, key=lambda row: (row["validation_metrics"]["state_grouped_mse"], row["seed"]))
    core.save_checkpoint(output / "best_checkpoint.npz", selected["params"], selected["config"], mean, scale, binary)
    predictions = {split: core.predict_mlp(selected["params"], x[split_indices])
                   for split, split_indices in indices.items()}
    write_csv(output / "training_history.csv", histories)
    write_csv(output / "seed_selection.csv", [{
        "seed": row["seed"], "selected": row is selected,
        "best_epoch": row["summary"]["best_epoch"],
        "validation_state_grouped_mse": row["validation_metrics"]["state_grouped_mse"],
        "validation_state_grouped_mean_l2": row["validation_metrics"]["state_grouped_mean_l2"],
    } for row in runs])
    write_json(output / "selection_audit.json", {
        "selected_seed": selected["seed"], "selection_metric": "validation state-grouped MSE",
        "test_used_for_selection": False, "training_seeds": config["training_seeds"],
        "jax_devices": [str(device) for device in jax.devices()],
    })

    metadata_by_sample = {str(row["sample_id"]): row for row in metadata}
    subset_rows = metric_rows(predictions, arrays, metadata_by_sample, indices)
    write_csv(output / "offline_subset_metrics.csv", subset_rows)
    startup_rows = [row for row in subset_rows if row["subset"] == "STARTUP"]
    warm_rows = [row for row in subset_rows if row["subset"] == "WARM_V3"]
    write_csv(output / "startup_metrics.csv", startup_rows)
    write_csv(output / "warm_history_metrics.csv", warm_rows)
    write_csv(output / "category_metrics.csv", [row for row in subset_rows
                                                  if row["subset"].startswith("CATEGORY:")])

    # Frozen baselines and a validation-selected linear diagnostic.  None can
    # influence selection of the registered MLP seed.
    model_predictions: dict[str, dict[str, np.ndarray]] = {
        "ZERO": {split: np.zeros_like(y[split_indices]) for split, split_indices in indices.items()},
        "TRAIN_MEAN": {split: np.broadcast_to(y[train].mean(axis=0), y[split_indices].shape).copy()
                       for split, split_indices in indices.items()},
    }
    linear_candidates = []
    for weight_decay in (0.0, 1e-4):
        linear = core.ridge_fit(x[train], y[train], weight_decay)
        validation_prediction = core.ridge_predict(linear, x[validation])
        score = core.subset_metrics(validation_prediction, y[validation],
                                    arrays["state_id"][validation])["state_grouped_mse"]
        linear_candidates.append((score, weight_decay, linear))
    _, linear_weight_decay, linear = min(linear_candidates, key=lambda row: row[:2])
    model_predictions["LINEAR"] = {
        split: core.ridge_predict(linear, x[split_indices]) for split, split_indices in indices.items()}
    model_predictions["NEW_MLP"] = predictions
    comparison_rows = []
    for model_name, by_split in model_predictions.items():
        row = {"model": model_name, "selected": model_name == "NEW_MLP"}
        if model_name == "LINEAR":
            row["validation_selected_weight_decay"] = linear_weight_decay
        if model_name == "NEW_MLP":
            row["validation_selected_seed"] = selected["seed"]
        for split, split_indices in indices.items():
            metrics = core.subset_metrics(by_split[split], y[split_indices],
                                          arrays["state_id"][split_indices])
            for key in ("mse", "state_grouped_mse", "state_grouped_mean_l2"):
                row[f"{split}_{key}"] = metrics[key]
        comparison_rows.append(row)

    # Compare on the exact immutable V3 test prefix only.
    v3_test = np.flatnonzero((np.arange(len(arrays["features"])) < v3_count) & (arrays["split"] == "test"))
    old_checkpoint = PREVIOUS_TRAINING / "best_checkpoint.npz"
    old_prediction = numpy_predict_checkpoint(old_checkpoint, arrays["features"][v3_test])
    new_prediction = core.predict_mlp(selected["params"], x[v3_test])
    previous = core.subset_metrics(old_prediction, y[v3_test], arrays["state_id"][v3_test])
    current = core.subset_metrics(new_prediction, y[v3_test], arrays["state_id"][v3_test])
    write_json(output / "previous_v3_comparison.json", {
        "comparison_set": "exact unchanged V3 test samples", "sample_count": len(v3_test),
        "previous_checkpoint": str(old_checkpoint), "previous_checkpoint_sha256": sha256(old_checkpoint),
        "previous_v3": previous, "startup_complete": current,
        "state_grouped_mean_l2_change": current["state_grouped_mean_l2"] - previous["state_grouped_mean_l2"],
    })
    comparison_rows.append({
        "model": "OLD_V3_CHECKPOINT", "selected": False,
        "evaluation_scope": "exact unchanged V3 test only",
        "test_mse": previous["mse"], "test_state_grouped_mse": previous["state_grouped_mse"],
        "test_state_grouped_mean_l2": previous["state_grouped_mean_l2"],
    })
    write_csv(output / "model_comparison.csv", comparison_rows)

    finite = all(np.isfinite(value).all() for value in predictions.values())
    projection = core.projection_replay(predictions, arrays, indices, protocol)
    write_json(output / "projection_replay_metrics.json", projection)
    failures = sum(len(projection[split]["projection_failures"]) for split in ("validation", "test"))
    reconstruction_error = max(projection[split]["oracle_action_reconstruction_max_abs_error"]
                               for split in ("validation", "test"))
    gate = {
        "passed": bool(finite and failures == 0 and reconstruction_error <= 1e-10),
        "finite_predictions_all_splits": finite,
        "validation_test_solver_failures": failures,
        "oracle_action_reconstruction_max_abs_error": reconstruction_error,
        "selection_independent_of_projection_test_metrics": True,
        "note": "rewrite magnitude is descriptive and is not a checkpoint-selection criterion",
    }
    write_json(output / "projection_replay_gate.json", gate)
    split_groups = {split: {row.get("leakage_group") for row in states if row["split"] == split}
                    for split in SPLITS}
    overlap = {f"{a}_{b}": sorted((split_groups[a] & split_groups[b]) - {None, ""})
               for index, a in enumerate(SPLITS) for b in SPLITS[index + 1:]}
    sanity = {
        "passed": bool(gate["passed"] and not any(overlap.values())),
        "architecture_exact_214_128_128_4": True,
        "training_seeds_exact_17_23_41": config["training_seeds"] == [17, 23, 41],
        "normalization_fit_train_only": True,
        "selection_validation_only": True,
        "test_used_for_selection": False,
        "v3_prefix_preservation": read_json(output / "merge_audit.json")["v3_prefix_numeric_bytes_preserved"],
        "split_leakage_groups": overlap,
        "finite_selected_predictions": finite,
        "projection_replay_gate_passed": gate["passed"],
    }
    write_json(output / "sanity_checks.json", sanity)
    if not gate["passed"]:
        raise RuntimeError("projection replay integrity gate failed; refusing completion")
    if not sanity["passed"]:
        raise RuntimeError("training sanity gate failed; refusing completion")
    selected_test = next(row for row in subset_rows if row["split"] == "test" and row["subset"] == "ALL")
    startup_test = next((row for row in startup_rows if row["split"] == "test"), None)
    warm_test = next((row for row in warm_rows if row["split"] == "test"), None)
    report = f"""# Startup-complete deterministic G_phi training

- Selected seed: {selected['seed']} (validation state-grouped MSE only)
- Architecture: 214 -> 128 -> 128 -> 4, SiLU, linear output
- Optimizer: AdamW, lr={config['learning_rate']}, weight decay={config['weight_decay']}
- Test state-grouped mean L2: {selected_test['state_grouped_mean_l2']:.8g}
- Startup test state-grouped mean L2: {startup_test['state_grouped_mean_l2'] if startup_test else 'N/A'}
- Warm-V3 test state-grouped mean L2: {warm_test['state_grouped_mean_l2'] if warm_test else 'N/A'}
- Previous V3 checkpoint on exact V3 test: {previous['state_grouped_mean_l2']:.8g}
- New checkpoint on exact V3 test: {current['state_grouped_mean_l2']:.8g}
- Projection replay integrity gate: {'PASS' if gate['passed'] else 'FAIL'}
- Test was used for final reporting only, never model selection.

The projection rewrite magnitude is descriptive.  It is not a model-selection
criterion and does not alter the supervised target.
"""
    (output / "training_report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-data", type=Path, default=V3)
    parser.add_argument("--startup-data", type=Path, default=STARTUP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.output.resolve() in {args.v3_data.resolve(), args.startup_data.resolve()}:
        raise SystemExit("output must not be an input dataset directory")
    config = read_json(HERE / "config.json")
    try:
        assert_fresh_output(args.output)
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from exc
    work_output = args.output.with_name(f".{args.output.name}.inprogress.{os.getpid()}")
    if work_output.exists():
        raise SystemExit(f"refusing pre-existing staging path: {work_output}")
    work_output.mkdir(parents=True)
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    write_json(work_output / "run_status.json", {
        "status": "IN_PROGRESS", "started_utc": started_utc,
        "completed_manifest_present": False,
    })
    try:
        merged, states, metadata, schema, protocol = prepare_merged(
            args.v3_data, args.startup_data, work_output, config["split_seed"],
            config["startup_source_episode_split_ratios"])
        if not args.prepare_only:
            run_training(merged, states, metadata, schema, protocol, work_output, config,
                         v3_count=read_json(args.v3_data / "manifest.json")["supervised_samples"])
        final_status = "PREPARED_ONLY" if args.prepare_only else "COMPLETED"
        write_json(work_output / "runtime_statistics.json", {
            "started_utc": started_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
            "wall_clock_seconds": time.monotonic() - started,
            "mode": "prepare-only" if args.prepare_only else "train-and-evaluate",
            "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "input_datasets_written": False,
        })
        write_json(work_output / "manifest.json", {
            "status": final_status,
            "selected_checkpoint": None if args.prepare_only else "best_checkpoint.npz",
            "generated_files_sha256": {
                path.name: sha256(path) for path in sorted(work_output.iterdir())
                if path.is_file() and path.name not in {"manifest.json", "run_status.json"}
            },
        })
        write_json(work_output / "run_status.json", {
            "status": final_status, "started_utc": started_utc,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "completed_manifest_present": True,
        })
        # Same-filesystem directory rename publishes the complete run in one
        # operation.  Partial runs never appear at the requested output path.
        work_output.rename(args.output)
    except BaseException as exc:
        # Explicitly invalidate the isolated staging run.  It is never renamed
        # to the requested output path, so stale files cannot mix with a retry.
        manifest_path = work_output / "manifest.json"
        if manifest_path.exists():
            manifest_path.rename(work_output / "manifest.invalidated.json")
        write_json(work_output / "run_status.json", {
            "status": "FAILED", "started_utc": started_utc,
            "failed_utc": datetime.now(timezone.utc).isoformat(),
            "completed_manifest_present": False,
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
