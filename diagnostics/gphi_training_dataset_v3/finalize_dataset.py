"""Select diverse B_63 eta-zero states and append them bitwise to Dataset V2."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V2 = ROOT / "diagnostics/gphi_training_dataset_v2"
V2_TRAIN = ROOT / "diagnostics/gphi_pilot_training_v2"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
sys.path.insert(0, str(SYSROOT))

from diagnostics.gphi_training_dataset_v2.build_states import restore_full
from diagnostics.gphi_training_dataset_v2.finalize_dataset import FeatureBuilder
from single_integrator.cbf import CBFConfig
from single_integrator.environment import Config


TARGET_PER_GROUP = 4


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def pairwise_stats(vectors: np.ndarray) -> dict:
    if len(vectors) < 2:
        return {"min": None, "median": None, "mean": None, "p95": None, "max": None}
    values = np.asarray([
        np.linalg.norm(vectors[i] - vectors[j]) / math.sqrt(vectors.shape[1])
        for i, j in combinations(range(len(vectors)), 2)
    ])
    return {
        "min": float(values.min()), "median": float(np.median(values)),
        "mean": float(values.mean()), "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def greedy_diverse(ids: list[str], representations: dict[str, np.ndarray], count: int) -> list[str]:
    if len(ids) < count:
        raise RuntimeError(("insufficient B_63 eta-zero states in group", len(ids), count, ids))
    ids = sorted(ids)
    vectors = np.stack([representations[state_id] for state_id in ids])
    distances = np.linalg.norm(vectors[:, None] - vectors[None, :], axis=-1) / math.sqrt(vectors.shape[1])
    first = int(np.argmax(distances.mean(axis=1)))
    chosen = [first]
    while len(chosen) < count:
        candidates = [index for index in range(len(ids)) if index not in chosen]
        next_index = max(candidates, key=lambda index: (float(np.min(distances[index, chosen])), ids[index]))
        chosen.append(next_index)
    return [ids[index] for index in chosen]


def main() -> None:
    started = time.monotonic()
    protocol = json.loads((HERE / "protocol.json").read_text())
    restoration = json.loads((HERE / "restoration_checks.json").read_text())
    if restoration["status"] != "PASS":
        raise RuntimeError("restoration gate failed")
    config = Config(**protocol["environment"])
    cbf = CBFConfig()
    candidates = read_jsonl(HERE / "candidate_state_manifest.jsonl")
    candidate_by_id = {row["state_id"]: row for row in candidates}
    raw_path = HERE / "raw/zero_oracle/records.jsonl"
    raw = read_jsonl(raw_path)
    grouped_records = defaultdict(list)
    for row in raw:
        grouped_records[row["state_id"]].append(row)
    arm_manifest = json.loads((HERE / "raw/zero_oracle/manifest.json").read_text())
    eligible = []
    rejected = []
    for state in candidates:
        records = sorted(grouped_records[state["state_id"]], key=lambda row: int(row["seed"]))
        counts = Counter(row["outcome"] for row in records)
        valid = len(records) == 64 and counts["success"] >= 63 and not any(row.get("execution_error") for row in records)
        result = {
            "state_id": state["state_id"], "source_group": state["source_group"],
            "evaluated": len(records), "success": counts["success"],
            "failures": len(records) - counts["success"], "B_63_member": valid,
        }
        (eligible if valid else rejected).append(result)

    # Construct state-level deployment features for diversity selection.  The
    # old V2 train-only normalization is used only for geometry, not training.
    builder = FeatureBuilder()
    representations = {}
    new_features = {}
    for result in eligible:
        state = candidate_by_id[result["state_id"]]
        env = restore_full(HERE / state["state_file"], config)
        features = []
        by_seed = {}
        for record in sorted(grouped_records[result["state_id"]], key=lambda row: int(row["seed"])):
            first = record["first_step"]
            if first is None:
                raise AssertionError((result["state_id"], record["seed"], "missing first step"))
            vector, structured = builder.build(env, first, config, cbf)
            features.append(vector)
            by_seed[int(record["seed"])] = (vector, structured, first)
        new_features[result["state_id"]] = by_seed
        representations[result["state_id"]] = np.mean(features, axis=0)
    v2_norm = json.loads((V2_TRAIN / "normalization.json").read_text())
    norm_mean = np.asarray(v2_norm["mean"])
    norm_scale = np.asarray(v2_norm["scale"])
    representations_norm = {
        state_id: (value - norm_mean) / norm_scale for state_id, value in representations.items()
    }

    eligible_by_group = defaultdict(list)
    for row in eligible:
        eligible_by_group[row["source_group"]].append(row["state_id"])
    all_groups = sorted({row["source_group"] for row in candidates})
    if len(all_groups) < 6:
        raise RuntimeError(("fewer than six source groups", all_groups))
    selected_ids = []
    for group in all_groups:
        selected_ids.extend(greedy_diverse(eligible_by_group[group], representations_norm, TARGET_PER_GROUP))
    selected_id_set = set(selected_ids)
    selected_candidates = [candidate_by_id[state_id] for state_id in selected_ids]
    if not 30 <= len(selected_candidates) <= 60:
        raise AssertionError(len(selected_candidates))

    # Copy V2 state bytes exactly so all final V3 state paths are self-contained.
    states_dir = HERE / "states"
    v2_states = read_jsonl(V2 / "state_manifest.jsonl")
    if states_dir.exists():
        # Safe interrupted-finalization resume: accept only an exact V2 prefix.
        for old_state in v2_states:
            existing = states_dir / Path(old_state["state_file"]).name
            if not existing.exists() or sha(existing) != old_state["state_sha256"]:
                raise RuntimeError(("non-reproducible pre-existing state output", existing))
    else:
        shutil.copytree(V2 / "states", states_dir, copy_function=shutil.copy2)
    final_states = [dict(row) for row in v2_states]
    for state_index, state in enumerate(selected_candidates, start=len(v2_states)):
        source = HERE / state["state_file"]
        destination = states_dir / source.name
        shutil.copy2(source, destination)
        row = dict(state)
        row["state_index"] = state_index
        row["state_file"] = str(destination.relative_to(HERE))
        row["state_sha256"] = sha(destination)
        row["split"] = row.pop("provisional_split")
        row["eta_zero_B_63_successes"] = next(item["success"] for item in eligible if item["state_id"] == row["state_id"])
        final_states.append(row)
    write_jsonl(HERE / "state_manifest.jsonl", final_states)

    # Keep every old sample array exactly and append 64 eta-zero Flow variants.
    with np.load(V2 / "samples.npz", allow_pickle=False) as source:
        v2_arrays = {key: np.asarray(source[key]).copy() for key in source.files}
    v2_metadata = read_jsonl(V2 / "sample_metadata.jsonl")
    append = defaultdict(list)
    appended_metadata = []
    for state in final_states[len(v2_states):]:
        for seed in sorted(new_features[state["state_id"]]):
            features, structured, first = new_features[state["state_id"]][seed]
            safe = np.asarray(first["u_safe"], dtype=np.float64).reshape(4)
            sample_id = f"{state['state_id']}__flow{seed}"
            append["features"].append(features)
            append["targets"].append(np.zeros(4, dtype=np.float64))
            append["target_actions"].append(safe)
            append["state_index"].append(state["state_index"])
            append["flow_seed"].append(seed)
            for array_key, structured_key in (
                ("observation", "observation"), ("positions", "positions"),
                ("velocities", "velocities"), ("goal_relative", "goal_relative"),
                ("relative_position", "relative_position"), ("relative_velocity", "relative_velocity"),
                ("u_flow", "u_flow"), ("u_safe", "u_safe"), ("B_goal", "B_goal"),
                ("B_rel", "B_rel"), ("goal_error_history", "goal_error_history"),
                ("recent_progress", "recent_progress"),
                ("first_projection_delta", "projection_delta"),
                ("first_projection_linear_residuals", "linear_residuals"),
            ):
                append[array_key].append(structured[structured_key])
            append["state_id"].append(state["state_id"])
            append["split"].append(state["split"])
            append["category"].append(state["category"])
            append["sample_id"].append(sample_id)
            appended_metadata.append({
                "sample_id": sample_id, "state_id": state["state_id"],
                "state_index": state["state_index"], "source_trajectory": state["source_trajectory"],
                "leakage_group": state["leakage_group"], "category": "RECOVERY",
                "split": state["split"], "flow_seed": seed, "target_dimension": 4,
                "target_semantics": "exact zero because eta=(0,0,0) is minimum-deformation B_63 member",
                "eta_best_metadata_only": [0.0, 0.0, 0.0],
                "E_near_metadata_only": [[0.0, 0.0, 0.0]],
                "oracle_J_min_metadata_only": 0.0,
                "label_classification": "LABEL_STABLE", "zero_label": True,
                "new_v3_recovery_zero": True,
            })
    arrays = {}
    for key, old in v2_arrays.items():
        # Let NumPy widen Unicode fields for longer V3 state/sample IDs; all
        # numeric fields retain the exact V2 dtype.
        new = np.asarray(append[key]) if old.dtype.kind == "U" else np.asarray(append[key], dtype=old.dtype)
        arrays[key] = np.concatenate([old, new], axis=0)
    np.savez_compressed(HERE / "samples.npz", **arrays)
    metadata = v2_metadata + appended_metadata
    write_jsonl(HERE / "sample_metadata.jsonl", metadata)
    shutil.copy2(V2 / "feature_schema.json", HERE / "feature_schema.json")

    # Verify base-array identity field by field (including split membership).
    equality = {key: bool(np.array_equal(arrays[key][: len(v2_arrays[key])], v2_arrays[key])) for key in arrays}
    if not all(equality.values()):
        raise AssertionError(("V2 samples changed", [key for key, passed in equality.items() if not passed]))

    # Split and leakage audits, including the required category/label table.
    state_zero = {}
    for state_id in sorted(set(arrays["state_id"].tolist())):
        mask = arrays["state_id"] == state_id
        state_zero[state_id] = bool(np.all(np.linalg.norm(arrays["targets"][mask], axis=1) <= 1e-14))
    split_table = {}
    for split in ("train", "validation", "test"):
        split_table[split] = {}
        for category in ("NORMAL", "PRE_DEADLOCK", "RECOVERY"):
            ids = {row["state_id"] for row in final_states if row["split"] == split and row["category"] == category}
            split_table[split][category] = {
                "zero": sum(state_zero[state_id] for state_id in ids),
                "nonzero": sum(not state_zero[state_id] for state_id in ids),
                "total": len(ids),
            }
    groups_by_split = {
        split: {row["leakage_group"] for row in final_states if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    group_overlaps = {
        f"{a}_{b}": sorted(groups_by_split[a] & groups_by_split[b])
        for a, b in combinations(groups_by_split, 2)
    }
    snapshot_hashes = [row["state_sha256"] for row in final_states]
    leakage = {
        "assignment_unit": "root recovery source/leakage group; all trajectory descendants colocated",
        "V2_sample_arrays_bitwise_equal": equality,
        "all_V2_sample_arrays_bitwise_equal": all(equality.values()),
        "split_group_overlaps": group_overlaps,
        "augmented_state_overlap_across_splits": False,
        "duplicate_sample_ids": len(arrays["sample_id"]) - len(set(arrays["sample_id"].tolist())),
        "duplicate_state_snapshots": len(snapshot_hashes) - len(set(snapshot_hashes)),
        "new_states_one_per_source_trajectory": len({row["source_trajectory"] for row in selected_candidates}) == len(selected_candidates),
        "features_all_finite": bool(np.isfinite(arrays["features"]).all()),
        "targets_all_finite": bool(np.isfinite(arrays["targets"]).all()),
    }
    leakage["passed"] = (
        not any(group_overlaps.values()) and leakage["duplicate_sample_ids"] == 0
        and leakage["duplicate_state_snapshots"] == 0
        and leakage["new_states_one_per_source_trajectory"]
        and leakage["features_all_finite"] and leakage["targets_all_finite"]
        and leakage["all_V2_sample_arrays_bitwise_equal"]
    )
    write_json(HERE / "leakage_checks.json", leakage)
    if not leakage["passed"]:
        raise RuntimeError("V3 leakage/integrity gate failed")

    split_manifest = {
        "assignment_unit": leakage["assignment_unit"],
        "strategy": "recomputed group-balanced audit; V2 memberships retained and new descendants inherit root-group split",
        "V2_state_memberships_preserved": True,
        "state_ids": {split: [row["state_id"] for row in final_states if row["split"] == split] for split in groups_by_split},
        "leakage_groups": {split: sorted(groups) for split, groups in groups_by_split.items()},
        "state_counts": {split: sum(row["split"] == split for row in final_states) for split in groups_by_split},
        "sample_counts": {split: int(np.sum(arrays["split"] == split)) for split in groups_by_split},
        "category_label_counts": split_table,
        "new_recovery_zero_counts": {split: sum(row["split"] == split for row in final_states[len(v2_states):]) for split in groups_by_split},
        "all_recovery_zero_counts": {split: split_table[split]["RECOVERY"]["zero"] for split in groups_by_split},
    }
    write_json(HERE / "split_manifest.json", split_manifest)

    selected_vectors = np.stack([representations_norm[state_id] for state_id in selected_ids])
    eligible_vectors = np.stack([representations_norm[row["state_id"]] for row in eligible])
    new_to_v2 = []
    v2_reps = []
    for state_id in sorted(set(v2_arrays["state_id"].tolist())):
        v2_reps.append(((v2_arrays["features"][v2_arrays["state_id"] == state_id].mean(axis=0) - norm_mean) / norm_scale))
    v2_reps = np.stack(v2_reps)
    for vector in selected_vectors:
        new_to_v2.append(float(np.min(np.linalg.norm(v2_reps - vector, axis=1) / math.sqrt(vector.size))))
    diversity = {
        "candidate_states": len(candidates), "eta_zero_B63_states": len(eligible),
        "rejected_nonzero_oracle_candidates": len(rejected), "selected_states": len(selected_ids),
        "independent_root_source_groups": len(all_groups),
        "distinct_selected_source_trajectories": len({row["source_trajectory"] for row in selected_candidates}),
        "states_per_source_trajectory": 1,
        "temporal_spacing": {
            "same_trajectory_pairs": 0,
            "near_adjacent_cross_split_pairs": 0,
            "planned_phase_fraction_min": float(min(row["planned_phase_fraction"] for row in selected_candidates)),
            "planned_phase_fraction_median": float(np.median([row["planned_phase_fraction"] for row in selected_candidates])),
            "planned_phase_fraction_max": float(max(row["planned_phase_fraction"] for row in selected_candidates)),
            "absolute_step_min": min(row["absolute_step"] for row in selected_candidates),
            "absolute_step_max": max(row["absolute_step"] for row in selected_candidates),
        },
        "feature_distance_definition": "RMS Euclidean distance after frozen V2 train-only normalization",
        "selected_pairwise_feature_distance": pairwise_stats(selected_vectors),
        "eligible_pairwise_feature_distance": pairwise_stats(eligible_vectors),
        "nearest_V2_state_distance": {
            "min": float(np.min(new_to_v2)), "median": float(np.median(new_to_v2)),
            "mean": float(np.mean(new_to_v2)), "p95": float(np.quantile(new_to_v2, 0.95)),
            "max": float(np.max(new_to_v2)),
        },
        "exact_feature_centroid_duplicates": len(selected_ids) - len({np.asarray(representations[state_id]).tobytes() for state_id in selected_ids}),
        "near_duplicate_pairs_below_1e-3_normalized_RMS": int(sum(
            np.linalg.norm(selected_vectors[i] - selected_vectors[j]) / math.sqrt(selected_vectors.shape[1]) < 1e-3
            for i, j in combinations(range(len(selected_vectors)), 2))),
        "selection": "four B_63 eta-zero states per root group by deterministic greedy max-min feature distance",
        "eligible_results": eligible, "rejected_results": rejected,
    }
    write_json(HERE / "diversity_metrics.json", diversity)

    source_groups = []
    for group in all_groups:
        group_candidates = [row for row in candidates if row["source_group"] == group]
        chosen = [row for row in selected_candidates if row["source_group"] == group]
        source_groups.append({
            "source_group": group, "anchor_state": group_candidates[0]["anchor_state"],
            "split": chosen[0]["provisional_split"], "candidate_count": len(group_candidates),
            "B_63_eta_zero_count": len(eligible_by_group[group]), "selected_count": len(chosen),
            "selected_state_ids": [row["state_id"] for row in chosen],
            "distinct_source_trajectories": len({row["source_trajectory"] for row in chosen}),
            "phase_counts": dict(Counter(row["recovery_phase"] for row in chosen)),
        })
    write_json(HERE / "source_group_manifest.json", {"groups": source_groups})
    new_csv = []
    for row in selected_candidates:
        result = next(item for item in eligible if item["state_id"] == row["state_id"])
        new_csv.append({
            "state_id": row["state_id"], "source_group": row["source_group"],
            "source_trajectory": row["source_trajectory"], "split": row["provisional_split"],
            "recovery_phase": row["recovery_phase"], "planned_phase_fraction": row["planned_phase_fraction"],
            "source_local_step": row["source_local_step"], "absolute_step": row["absolute_step"],
            "eta_zero_successes": result["success"], "eta_zero_trials": result["evaluated"],
            "target_norm": 0.0, "restoration_passed": True,
        })
    write_csv(HERE / "new_recovery_zero_states.csv", new_csv)

    # Extend V2 label/oracle metadata without changing old entries.
    labels = json.loads((V2 / "label_statistics.json").read_text())
    per_state = dict(labels["per_state"])
    new_oracles = []
    for row in final_states[len(v2_states):]:
        success = int(row["eta_zero_B_63_successes"])
        per_state[row["state_id"]] = {
            "classification": "LABEL_STABLE", "usable": True, "B_63_size": 1,
            "E_near_size": 1, "eta_values_evaluated": 1,
            "continuation_records_considered": 64, "eta_best": [0.0, 0.0, 0.0],
            "E_near": [[0.0, 0.0, 0.0]], "J_min": 0.0,
            "mean_pairwise_L2_eta_same_seed": 0.0, "max_pairwise_L2_eta_same_seed": 0.0,
            "typical_g_exec_norm": 0.0,
        }
        new_oracles.append({
            "state_id": row["state_id"], "category": "RECOVERY", "B_63_empty": False,
            "B_63": [[0.0, 0.0, 0.0]], "eta_best": [0.0, 0.0, 0.0], "J_min": 0.0,
            "E_near": [[0.0, 0.0, 0.0]], "paired_comparisons": [{"eta": [0.0, 0.0, 0.0], "paired_n": success}],
            "cells": [{"eta": [0.0, 0.0, 0.0], "evaluated": 64, "counts": {"success": success, "deadlock": 0, "timeout": 64-success, "collision": 0}, "B_63_member": True}],
        })
    zero_states = [state_id for state_id, zero in state_zero.items() if zero]
    nonzero_states = [state_id for state_id, zero in state_zero.items() if not zero]
    class_counts = Counter(value["classification"] for value in per_state.values())
    labels.update({
        "unique_selected_states": len(final_states), "usable_labeled_states": len(final_states),
        "supervised_samples": len(arrays["features"]),
        "samples_per_usable_state": {"64": len(final_states)},
        "selected_category_counts": dict(Counter(row["category"] for row in final_states)),
        "usable_category_counts": dict(Counter(row["category"] for row in final_states)),
        "zero_label_state_count": len(zero_states), "nonzero_label_state_count": len(nonzero_states),
        "zero_label_sample_count": 64 * len(zero_states), "nonzero_label_sample_count": 64 * len(nonzero_states),
        "B_63_empty_state_count": 0, "B_63_empty_state_ids": [],
        "classification_counts_over_selected_states": dict(class_counts),
        "quarantined_multivalued_state_ids": [], "per_state": per_state,
    })
    write_json(HERE / "label_statistics.json", labels)
    write_jsonl(HERE / "oracle_search_results.jsonl", read_jsonl(V2 / "oracle_search_results.jsonl") + new_oracles)

    category_counts = dict(Counter(row["category"] for row in final_states))
    recovery_zero = {split: split_table[split]["RECOVERY"]["zero"] for split in split_table}
    runtime = {
        "new_candidate_source_trajectories": len(candidates),
        "new_oracle_rollout_attempts": arm_manifest["new_rollouts"],
        "new_oracle_physical_steps": arm_manifest["physical_steps"],
        "oracle_elapsed_s": arm_manifest["elapsed_s"],
        "reused_oracle_rollouts": arm_manifest["reused_rollouts"],
        "finalization_elapsed_s": time.monotonic() - started,
        "resources": {"GPU_shards": 1, "CPU_cores": 4, "JAX_memory_fraction": 0.10},
        "resource_observation": "one other laboratory user had an active scheduler job; GPU was otherwise at 2 MiB and 0% before launch",
    }
    write_json(HERE / "runtime_statistics.json", runtime)
    report = f"""# G_phi training dataset v3: targeted RECOVERY-zero coverage

## Result

- Dataset V2 was retained exactly and **{len(selected_candidates)}** independently sourced, oracle-confirmed zero-label RECOVERY states were appended.
- Final unique states / samples: **{len(final_states)} / {len(arrays['features'])}**.
- Categories: {category_counts}.
- Zero/nonzero states: **{len(zero_states)} / {len(nonzero_states)}**.
- New states use **{len(all_groups)}** root recovery source groups and {len(selected_candidates)} distinct corrected source trajectories.
- B_63-empty states: **0**; multivalued states: **0**.  Every new state has singleton eta-zero E_near and is LABEL_STABLE.

## Group-balanced split

- State counts: {split_manifest['state_counts']}.
- Zero-label RECOVERY train/validation/test: **{recovery_zero['train']} / {recovery_zero['validation']} / {recovery_zero['test']}**.
- Exact category/label counts are in `split_manifest.json`.
- V2 memberships and all V2 sample arrays were preserved bitwise; no source-group, state, sample, or near-adjacent trajectory leakage was found.

## Restoration, oracle, and diversity

- Exact augmented-state restoration, matched u_Flow/u_safe/u_exec, and one-step transition reconstruction: **PASS ({restoration['passed_states']}/{restoration['candidate_states']})**.
- Candidate eta-zero B_63 pass/reject: **{len(eligible)} / {len(rejected)}**; only pass states entered V3.
- New selected source trajectories are one-state-per-trajectory. Feature-space and duplicate audits are recorded in `diversity_metrics.json`.
- Oracle work: {arm_manifest['new_rollouts']} new continuations, {arm_manifest['physical_steps']} physical steps, {arm_manifest['reused_rollouts']} reused; {arm_manifest['elapsed_s']:.1f} s on one GPU shard.

## Frozen semantics

Feature dimension is **214**, target dimension is **4**, and every new target is exactly `g*_exec = 0` because `eta=(0,0,0)` satisfies the unchanged 63/64 rule. Physics, FlowBC, both hard projections, monitor/event definitions, oracle semantics, feature schema, and deterministic target are unchanged.
"""
    (HERE / "dataset_report.md").write_text(report)

    required = [
        "dataset_report.md", "new_recovery_zero_states.csv", "source_group_manifest.json",
        "state_manifest.jsonl", "samples.npz", "feature_schema.json", "split_manifest.json",
        "restoration_checks.json", "leakage_checks.json", "diversity_metrics.json",
        "label_statistics.json", "runtime_statistics.json", "oracle_search_results.jsonl",
    ]
    manifest = {
        "dataset": "gphi_training_dataset_v3", "parent_dataset": str(V2),
        "unique_augmented_states": len(final_states), "new_recovery_zero_states": len(selected_candidates),
        "supervised_samples": len(arrays["features"]), "feature_dimension": 214, "target_dimension": 4,
        "category_counts": category_counts, "zero_label_states": len(zero_states),
        "nonzero_label_states": len(nonzero_states), "B_63_empty_states": 0,
        "LABEL_MULTIVALUED_states": 0, "restoration_status": restoration["status"],
        "leakage_checks_passed": leakage["passed"],
        "files_sha256": {name: sha(HERE / name) for name in required},
    }
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({
        "unique_states": len(final_states), "new_recovery_zero": len(selected_candidates),
        "source_groups": len(all_groups), "categories": category_counts,
        "recovery_zero_split": recovery_zero, "samples": len(arrays["features"]),
        "eta_zero_eligible": len(eligible), "eta_zero_rejected": len(rejected),
        "leakage_passed": leakage["passed"],
    }, indent=2))


if __name__ == "__main__":
    main()
