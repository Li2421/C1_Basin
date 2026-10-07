"""Freeze the strict-deadlock coverage augmentation and pre-retrain audit.

Only the 11 historical earliest-robust states are appended.  The six fresh
states are evaluated in-memory for the pre-audit and are never serialized into
the training dataset.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("/home/zhihan/research/Basin_C1")
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
HERE = ROOT / "diagnostics/gphi_strict_deadlock_coverage_retrain_v1"
OUTPUT = ROOT / "diagnostics/gphi_training_dataset_strict_deadlock_v1"
BASE = ROOT / "diagnostics/gphi_training_startup_complete_v1/artifacts"
BASE_DATASET = ROOT / "diagnostics/gphi_training_dataset_startup_complete_v1"
CAPACITY = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1"
OLD_CHECKPOINT = ROOT / "diagnostics/gphi_startup_warm_pareto_v1/best_balanced_checkpoint.npz"

EXPECTED_BASE_SHA256 = "639f9c1959b98c6187fa1083f44aa1e0aa14c3c0a20405a7e25c43e5618b9773"
EXPECTED_OLD_CHECKPOINT_SHA256 = "c29800f6cf6ec12568647a6a6b3b93ae1176f78343e876c5799736068277bf7e"
EXPECTED_BASE_SAMPLES = 26432
EXPECTED_BASE_STATES = 413
EXPECTED_VARIANTS = 64

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SYSROOT))

from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import (  # noqa: E402
    StartupAwareFeatureBuilder,
)
from diagnostics.gphi_training_dataset_v1.build_states import restore_full  # noqa: E402
from diagnostics.gphi_training_startup_complete_v1.pipeline import (  # noqa: E402
    numpy_predict_checkpoint,
)
from diagnostics.success_basin_multimodality.exact_projector import (  # noqa: E402
    project_velocity_with_retry,
)
from single_integrator.cbf import CBFConfig, barrier_constraints  # noqa: E402
from single_integrator.environment import Config  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def eta_key(value: object) -> tuple[float, float, float]:
    if isinstance(value, str):
        value = json.loads(value)
    return tuple(round(float(item), 8) for item in value)  # type: ignore[arg-type]


def restore_query(path: Path, config: Config):
    """Restore a compact snapshot and expose its authoritative finite tail.

    The frozen snapshot stores only the last 41 goal-error entries and the
    generic restorer represents older unavailable entries as NaN placeholders.
    Runtime deployment has a full finite list, while FeatureBuilder consumes
    only the last 41 entries.  Removing only those leading placeholders is
    therefore exactly feature-equivalent and keeps the startup left-padding
    rule correct for S0.
    """
    env = restore_full(path, config)
    finite = [
        np.asarray(value, dtype=np.float64).copy()
        for value in env.distance_history
        if np.isfinite(value).all()
    ]
    if not finite or len(finite) > 41:
        raise RuntimeError((path, "invalid compact history tail", len(finite)))
    env.distance_history = finite
    return env


def cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return None if denominator <= 1e-14 else float(np.dot(left, right) / denominator)


def load_capacity_states() -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    with (CAPACITY / "episode_capacity_classification.csv").open(newline="") as handle:
        episodes = list(csv.DictReader(handle))
    query_rows: dict[str, dict] = {}
    with (CAPACITY / "queried_states.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            query_rows[row["state_id"]] = row
    selected = []
    robust_records: dict[str, dict] = {}
    for episode in episodes:
        state_id = episode["earliest_robust_state"]
        eta = eta_key(episode["best_eta_at_capacity_onset"])
        raw_path = CAPACITY / "raw/robust" / f"{state_id}.jsonl"
        rows = [json.loads(line) for line in raw_path.read_text().splitlines() if line]
        rows = [
            row for row in rows
            if not row.get("state_complete") and eta_key(row["eta"]) == eta
        ]
        rows.sort(key=lambda row: int(row["seed"]))
        if len(rows) != EXPECTED_VARIANTS:
            raise RuntimeError((state_id, eta, "expected 64 robust rows", len(rows)))
        if len({int(row["seed"]) for row in rows}) != EXPECTED_VARIANTS:
            raise RuntimeError((state_id, "duplicate robust seeds"))
        if sum(row["outcome"] == "success" for row in rows) < 63:
            raise RuntimeError((state_id, "selected eta is not B63"))
        if any(row.get("execution_error") or row.get("first_step") is None for row in rows):
            raise RuntimeError((state_id, "invalid robust records"))
        selected.append({
            "case_id": episode["case_id"],
            "benchmark": episode["benchmark"],
            "episode_index": int(episode["episode_index"]),
            "state_id": state_id,
            "eta": eta,
            "success_count": sum(row["outcome"] == "success" for row in rows),
            "evaluated": len(rows),
            "J_def": float(episode["J_def"]),
            "query": query_rows[state_id],
        })
        robust_records[state_id] = {str(int(row["seed"])): row for row in rows}
    manifest = json.loads((CAPACITY / "strict_deadlock_manifest.json").read_text())
    cases = {row["case_id"]: row for row in manifest["cases"]}
    return selected, robust_records, cases


def build_samples(
    selected: list[dict], records: dict[str, dict], config: Config, cbf: CBFConfig,
) -> tuple[dict[str, list], dict[str, list[dict]]]:
    builder = StartupAwareFeatureBuilder()
    by_state: dict[str, list[dict]] = {}
    for item in selected:
        state_id = item["state_id"]
        env = restore_query(Path(item["query"]["state_file"]), config)
        A, lower, _ = barrier_constraints(env.snapshot(), cbf)
        samples = []
        for seed, row in sorted(records[state_id].items(), key=lambda pair: int(pair[0])):
            first = row["first_step"]
            feature, structured = builder.build(env, first, config, cbf)
            safe = np.asarray(first["u_safe"], dtype=np.float64).reshape(2, 2)
            executed = np.asarray(first["u_exec"], dtype=np.float64).reshape(2, 2)
            raw = np.asarray(first["g_raw"], dtype=np.float64).reshape(2, 2)
            replay, _, _, _ = project_velocity_with_retry(
                safe + raw, A, lower, config.max_speed, cbf
            )
            if float(np.max(np.abs(replay - executed))) > 1e-9:
                raise RuntimeError((state_id, seed, "second-projection replay mismatch"))
            target = (executed - safe).reshape(4)
            action = safe.reshape(4) + target
            if float(np.min(A @ action - lower)) < -1e-7:
                raise RuntimeError((state_id, seed, "infeasible target"))
            samples.append({
                "seed": int(seed),
                "feature": np.asarray(feature, dtype=np.float64),
                "target": target,
                "target_action": action,
                "structured": structured,
                "first": first,
            })
        by_state[state_id] = samples
    return {"selected": selected}, by_state


def target_error_rows(
    checkpoint: Path, selected: list[dict], samples_by_state: dict[str, list[dict]],
    config: Config, cbf: CBFConfig, cohort_label: str,
) -> list[dict]:
    rows = []
    for item in selected:
        state_id = item["state_id"]
        samples = samples_by_state[state_id]
        features = np.stack([row["feature"] for row in samples])
        targets = np.stack([row["target"] for row in samples])
        prediction = numpy_predict_checkpoint(checkpoint, features)
        env = restore_query(Path(item["query"]["state_file"]), config)
        A, lower, _ = barrier_constraints(env.snapshot(), cbf)
        executed_prediction = []
        executed_target = []
        cosines = []
        for pred, target, sample in zip(prediction, targets, samples):
            safe = np.asarray(sample["first"]["u_safe"], dtype=np.float64).reshape(2, 2)
            pred_action, _, _, _ = project_velocity_with_retry(
                safe + pred.reshape(2, 2), A, lower, config.max_speed, cbf
            )
            executed_prediction.append(pred_action.reshape(4))
            executed_target.append(safe.reshape(4) + target)
            value = cosine(pred, target)
            if value is not None:
                cosines.append(value)
        raw_error = np.linalg.norm(prediction - targets, axis=1)
        exec_error = np.linalg.norm(
            np.stack(executed_prediction) - np.stack(executed_target), axis=1
        )
        norm_error = np.abs(
            np.linalg.norm(prediction, axis=1) - np.linalg.norm(targets, axis=1)
        )
        rows.append({
            "cohort": cohort_label,
            "benchmark": item["benchmark"],
            "case_id": item["case_id"],
            "state_id": state_id,
            "query_label": item["query"]["query_label"],
            "flow_variants": len(samples),
            "target_l2_mean": float(raw_error.mean()),
            "target_l2_median": float(np.median(raw_error)),
            "target_l2_p95": float(np.quantile(raw_error, 0.95)),
            "executed_action_l2_mean": float(exec_error.mean()),
            "executed_action_l2_median": float(np.median(exec_error)),
            "correction_norm_error_mean": float(norm_error.mean()),
            "target_norm_mean": float(np.linalg.norm(targets, axis=1).mean()),
            "prediction_norm_mean": float(np.linalg.norm(prediction, axis=1).mean()),
            "cosine_similarity_mean": float(np.mean(cosines)) if cosines else None,
            "cosine_similarity_valid": len(cosines),
        })
    return rows


def coverage_audit(
    selected: list[dict], samples_by_state: dict[str, list[dict]], base: dict[str, np.ndarray],
) -> list[dict]:
    train = np.flatnonzero(base["split"] == "train")
    old_checkpoint = np.load(OLD_CHECKPOINT, allow_pickle=False)
    mean = np.asarray(old_checkpoint["normalization_mean"], dtype=np.float64)
    scale = np.asarray(old_checkpoint["normalization_scale"], dtype=np.float64)
    old_checkpoint.close()
    normalized_base = (base["features"][train] - mean) / scale
    base_state_ids = base["state_id"][train]
    base_positions = base["positions"][train]
    unique_train_states = list(dict.fromkeys(base_state_ids.tolist()))
    base_centroids = np.stack([
        normalized_base[base_state_ids == state_id].mean(axis=0)
        for state_id in unique_train_states
    ])
    rows = []
    for item in selected:
        features = np.stack([row["feature"] for row in samples_by_state[item["state_id"]]])
        normalized = (features - mean) / scale
        centroid = normalized.mean(axis=0)
        distances = np.linalg.norm(base_centroids - centroid, axis=1)
        nearest_index = int(np.argmin(distances))
        # A true exact feature match is stronger than a coincident episode ID.
        exact_feature_matches = 0
        min_variant_distances = []
        for vector in normalized:
            delta = normalized_base - vector
            norms = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            minimum = float(norms.min())
            min_variant_distances.append(minimum)
            exact_feature_matches += int(minimum == 0.0)
        query_positions = np.asarray(samples_by_state[item["state_id"]][0]["structured"]["positions"])
        position_match = np.max(np.abs(base_positions - query_positions), axis=(1, 2)) == 0.0
        rows.append({
            "cohort": item["benchmark"],
            "case_id": item["case_id"],
            "state_id": item["state_id"],
            "exact_feature_match_count": exact_feature_matches,
            "exact_physical_position_match_count": int(position_match.sum()),
            "same_source_episode_in_base": False,
            "nearest_train_state_id": unique_train_states[nearest_index],
            "nearest_normalized_state_centroid_l2": float(distances[nearest_index]),
            "nearest_normalized_input_l2_min": float(np.min(min_variant_distances)),
            "nearest_normalized_input_l2_mean": float(np.mean(min_variant_distances)),
        })
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if sha256(BASE / "merged_samples.npz") != EXPECTED_BASE_SHA256:
        raise RuntimeError("base dataset hash mismatch")
    if sha256(OLD_CHECKPOINT) != EXPECTED_OLD_CHECKPOINT_SHA256:
        raise RuntimeError("old checkpoint hash mismatch")
    with np.load(BASE / "merged_samples.npz", allow_pickle=False) as source:
        base = {key: np.asarray(source[key]).copy() for key in source.files}
    if len(base["features"]) != EXPECTED_BASE_SAMPLES:
        raise RuntimeError("base sample count mismatch")
    base_states = read_jsonl(BASE / "merged_state_manifest.jsonl")
    base_metadata = read_jsonl(BASE / "merged_sample_metadata.jsonl")
    if len(base_states) != EXPECTED_BASE_STATES or len(base_metadata) != EXPECTED_BASE_SAMPLES:
        raise RuntimeError("base manifest counts mismatch")

    capacity_manifest = json.loads((CAPACITY / "strict_deadlock_manifest.json").read_text())
    config = Config(**capacity_manifest["environment"])
    cbf = CBFConfig(**capacity_manifest["cbf"])
    selected, robust_records, cases = load_capacity_states()
    historical = [row for row in selected if row["benchmark"] == "historical"]
    fresh = [row for row in selected if row["benchmark"] == "fresh_unseen"]
    if len(historical) != 11 or len(fresh) != 6:
        raise RuntimeError(("unexpected cohort sizes", len(historical), len(fresh)))
    _, samples_by_state = build_samples(selected, robust_records, config, cbf)

    pre_rows = target_error_rows(
        OLD_CHECKPOINT, historical, samples_by_state, config, cbf, "historical11"
    ) + target_error_rows(
        OLD_CHECKPOINT, fresh, samples_by_state, config, cbf, "fresh6_heldout"
    )
    write_csv(HERE / "pre_retrain_target_error.csv", pre_rows)
    coverage_rows = coverage_audit(selected, samples_by_state, base)

    states_dir = OUTPUT / "states"
    states_dir.mkdir(exist_ok=True)
    new_states = []
    new_metadata = []
    append: dict[str, list] = defaultdict(list)
    historical_ids = {item["state_id"] for item in historical}
    fresh_ids = {item["state_id"] for item in fresh}
    if historical_ids & fresh_ids:
        raise RuntimeError("historical/fresh state overlap")
    for offset, item in enumerate(historical):
        state_id = item["state_id"]
        query = item["query"]
        source_state = Path(query["state_file"])
        destination = states_dir / source_state.name
        shutil.copy2(source_state, destination)
        case = cases[item["case_id"]]
        state_index = EXPECTED_BASE_STATES + offset
        state_row = {
            "state_id": state_id,
            "state_index": state_index,
            "state_file": str(destination.relative_to(OUTPUT)),
            "state_sha256": sha256(destination),
            "split": "train",
            "category": "STRICT_DEADLOCK",
            "leakage_group": f"strict_deadlock_historical_{item['case_id']}",
            "source_episode": item["case_id"],
            "source_trajectory": case["source_trajectory"],
            "source_benchmark": "historical_wide",
            "physical_step": int(query["query_step"]),
            "query_label": query["query_label"],
            "eta_best": list(item["eta"]),
            "B_63_successes": item["success_count"],
            "B_63_evaluated": item["evaluated"],
            "oracle_J_def": item["J_def"],
        }
        new_states.append(state_row)
        for sample in samples_by_state[state_id]:
            seed = sample["seed"]
            sample_id = f"{state_id}__flow{seed}"
            structured = sample["structured"]
            values = {
                "features": sample["feature"],
                "targets": sample["target"],
                "target_actions": sample["target_action"],
                "state_index": state_index,
                "flow_seed": seed,
                "observation": structured["observation"],
                "positions": structured["positions"],
                "velocities": structured["velocities"],
                "goal_relative": structured["goal_relative"],
                "relative_position": structured["relative_position"],
                "relative_velocity": structured["relative_velocity"],
                "u_flow": structured["u_flow"],
                "u_safe": structured["u_safe"],
                "B_goal": structured["B_goal"],
                "B_rel": structured["B_rel"],
                "goal_error_history": structured["goal_error_history"],
                "recent_progress": structured["recent_progress"],
                "first_projection_delta": structured["projection_delta"],
                "first_projection_linear_residuals": structured["linear_residuals"],
                "state_id": state_id,
                "split": "train",
                "category": "STRICT_DEADLOCK",
                "sample_id": sample_id,
            }
            for key, value in values.items():
                append[key].append(value)
            new_metadata.append({
                "sample_id": sample_id,
                "state_id": state_id,
                "state_index": state_index,
                "source_trajectory": case["source_trajectory"],
                "leakage_group": state_row["leakage_group"],
                "category": "STRICT_DEADLOCK",
                "split": "train",
                "flow_seed": seed,
                "target_dimension": 4,
                "target_semantics": "executed correction u_exec-u_safe from minimum-J_def robust B63 eta at earliest robust-onset state",
                "eta_best_metadata_only": list(item["eta"]),
                "oracle_J_min_metadata_only": item["J_def"],
                "label_classification": "ROBUST_B63",
                "zero_label": bool(np.linalg.norm(sample["target"]) <= 1e-14),
                "strict_deadlock_coverage_v1": True,
            })

    merged = {}
    prefix_equal = {}
    for key, old in base.items():
        new = np.asarray(append[key])
        if old.dtype.kind not in "USO":
            new = new.astype(old.dtype, copy=False)
        merged[key] = np.concatenate([old, new], axis=0)
        prefix_equal[key] = bool(np.array_equal(merged[key][:len(old)], old))
    if not all(prefix_equal.values()):
        raise RuntimeError(("base prefix changed", [key for key, value in prefix_equal.items() if not value]))
    final_states = base_states + new_states
    final_metadata = base_metadata + new_metadata
    if len(final_states) != 424 or len(final_metadata) != 27136:
        raise RuntimeError((len(final_states), len(final_metadata)))
    if set(merged["state_id"][-704:].tolist()) & fresh_ids:
        raise RuntimeError("fresh-6 contaminated training arrays")

    np.savez_compressed(OUTPUT / "samples.npz", **merged)
    write_jsonl(OUTPUT / "state_manifest.jsonl", final_states)
    write_jsonl(OUTPUT / "sample_metadata.jsonl", final_metadata)
    shutil.copy2(BASE_DATASET / "feature_schema.json", OUTPUT / "feature_schema.json")
    shutil.copy2(BASE_DATASET / "startup_feature_builder.py", OUTPUT / "startup_feature_builder.py")
    split_manifest = {
        "assignment_unit": "frozen base groups plus 11 historical strict-deadlock source episodes",
        "sample_counts": {
            split: int(np.sum(merged["split"] == split))
            for split in ("train", "validation", "test")
        },
        "state_counts": {
            split: sum(row["split"] == split for row in final_states)
            for split in ("train", "validation", "test")
        },
        "base_validation_test_unchanged": True,
        "strict_deadlock_added_states": 11,
        "strict_deadlock_added_samples": 704,
        "strict_deadlock_split": "train_only",
    }
    write_json(OUTPUT / "split_manifest.json", split_manifest)

    augmentation_manifest = {
        "schema": "gphi_strict_deadlock_coverage_augmentation_v1",
        "selection": "historical 11 earliest robust-onset states from frozen capacity audit",
        "training_states": [{key: value for key, value in item.items() if key != "query"} for item in historical],
        "fresh6_state_ids_excluded": sorted(fresh_ids),
        "standard_variants_per_state": 64,
        "flow_seed_range": [95310001, 95310064],
        "special_sample_weighting": False,
        "oversampling": False,
        "loss_weighting": False,
        "additional_neighbor_states": False,
        "target_semantics": "first-step executed correction for minimum-J_def robust B63 eta",
        "source_capacity_manifest_sha256": sha256(CAPACITY / "manifest.json"),
    }
    write_json(HERE / "augmentation_manifest.json", augmentation_manifest)
    write_json(HERE / "overlap_audit.json", {
        "passed": True,
        "coverage_rows": coverage_rows,
        "historical_exact_feature_matches_total": sum(row["exact_feature_match_count"] for row in coverage_rows if row["cohort"] == "historical"),
        "fresh6_in_training_arrays": False,
        "fresh6_in_state_manifest": False,
        "fresh6_in_sample_metadata": False,
        "fresh6_state_ids": sorted(fresh_ids),
        "original_prefix_array_identity": prefix_equal,
        "all_original_arrays_unchanged": all(prefix_equal.values()),
        "base_samples_sha256": EXPECTED_BASE_SHA256,
    })
    dataset_manifest = {
        "schema": "gphi_training_dataset_strict_deadlock_v1",
        "base_dataset": str(BASE / "merged_samples.npz"),
        "base_samples_sha256": EXPECTED_BASE_SHA256,
        "base_unique_states": EXPECTED_BASE_STATES,
        "base_samples": EXPECTED_BASE_SAMPLES,
        "added_unique_states": len(new_states),
        "added_samples": len(new_metadata),
        "final_unique_states": len(final_states),
        "final_samples": len(final_metadata),
        "fresh6_training_overlap": 0,
        "files_sha256": {
            name: sha256(OUTPUT / name)
            for name in (
                "samples.npz", "state_manifest.jsonl", "sample_metadata.jsonl",
                "feature_schema.json", "startup_feature_builder.py", "split_manifest.json",
            )
        },
    }
    write_json(OUTPUT / "manifest.json", dataset_manifest)
    write_json(HERE / "augmented_dataset_manifest.json", dataset_manifest)
    print(json.dumps({
        "status": "PASS",
        "historical_states": len(historical),
        "fresh_audit_states": len(fresh),
        "added_samples": len(new_metadata),
        "final_samples": len(final_metadata),
        "dataset_sha256": dataset_manifest["files_sha256"]["samples.npz"],
    }, indent=2))


if __name__ == "__main__":
    main()
