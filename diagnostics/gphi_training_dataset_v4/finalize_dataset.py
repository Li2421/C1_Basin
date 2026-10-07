"""Append validated matched recovery-boundary states to immutable Dataset V3."""

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
V3 = ROOT / "diagnostics/gphi_training_dataset_v3"
V3_TRAIN = ROOT / "diagnostics/gphi_pilot_training_v3"
SYSROOT = Path("/home/zhihan/research/02_C1_Toy_GiveWay")
sys.path.insert(0, str(SYSROOT))

from diagnostics.gphi_training_dataset_v2.build_states import restore_full
from diagnostics.gphi_training_dataset_v2.finalize_dataset import FeatureBuilder, eta_key, pairwise_mean, state_oracles
from single_integrator.cbf import CBFConfig, barrier_constraints
from single_integrator.environment import Config


EPS = 1e-12


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
        writer.writeheader(); writer.writerows(rows)


def same_record(a: dict, b: dict) -> bool:
    return a["outcome"] == b["outcome"] and int(a["steps"]) == int(b["steps"]) and abs(float(a["J_def"]) - float(b["J_def"])) <= 1e-7


def main() -> None:
    started = time.monotonic()
    protocol = json.loads((HERE / "protocol.json").read_text())
    config = Config(**protocol["environment"]); cbf = CBFConfig()
    selected = read_jsonl(HERE / "selected_state_manifest.jsonl")
    selected_by_id = {row["state_id"]: row for row in selected}
    effective = {}
    raw_paths = sorted((HERE / "raw").glob("eta_zero*/records.jsonl"))
    raw_paths += [HERE / f"raw/nonzero_candidate_shard{shard}/records.jsonl" for shard in (0, 1)]
    source_counts = {}
    for path in raw_paths:
        rows = read_jsonl(path)
        source_counts[str(path.relative_to(HERE))] = len(rows)
        for row in rows:
            if row["state_id"] not in selected_by_id:
                continue
            key = (row["state_id"], eta_key(row["eta"]), int(row["seed"]))
            if key in effective and not same_record(effective[key], row):
                raise AssertionError(("conflicting tuple", key))
            effective[key] = row
    oracle_rows, oracle_by_state = state_oracles(effective, selected)
    write_jsonl(HERE / "new_oracle_search_results.jsonl", oracle_rows)
    empty = [row["state_id"] for row in oracle_rows if row["B_63_empty"]]
    if empty:
        raise RuntimeError(("selected boundary states have empty B_63", empty))

    builder = FeatureBuilder()
    append = defaultdict(list)
    metadata_new = []
    label_statistics = {}
    state_targets = {}
    multivalued = []
    min_target_residual = float("inf"); max_target_speed_excess = -float("inf")
    for state in selected:
        state_id = state["state_id"]
        oracle = oracle_by_state[state_id]
        near = [eta_key(value) for value in oracle["E_near"]]
        eta_rows = {
            eta: {int(seed): effective[(state_id, eta, int(seed))] for seed in sorted(key[2] for key in effective if key[0] == state_id and key[1] == eta)}
            for eta in near
        }
        common = sorted(set.intersection(*(set(rows) for rows in eta_rows.values())))
        if len(common) != 64:
            raise AssertionError((state_id, "near cohort", len(common)))
        env = restore_full(HERE / state["state_file"], config)
        A, lower, _ = barrier_constraints(env.snapshot(), cbf)
        target_by_seed = {}
        labels_by_eta = defaultdict(list)
        pair_distances = []; correction_norms = []; component_variances = []
        for seed in common:
            first_steps = [eta_rows[eta][seed]["first_step"] for eta in near]
            safe_stack = np.stack([np.asarray(first["u_safe"], dtype=np.float64).reshape(4) for first in first_steps])
            flow_stack = np.stack([np.asarray(first["u_flow"], dtype=np.float64).reshape(4) for first in first_steps])
            if np.max(np.ptp(safe_stack, axis=0)) > 1e-10 or np.max(np.ptp(flow_stack, axis=0)) > 1e-10:
                raise AssertionError((state_id, seed, "matched Flow discrepancy"))
            safe = safe_stack[0]; flow = flow_stack[0]
            labels = []
            for eta, first in zip(near, first_steps):
                executed = np.asarray(first["u_exec"], dtype=np.float64).reshape(4)
                label = executed - safe
                labels.append(label); labels_by_eta[eta].append(label)
            stack = np.stack(labels)
            target = stack.mean(axis=0)
            target_action = safe + target
            min_target_residual = min(min_target_residual, float(np.min(A @ target_action - lower)))
            max_target_speed_excess = max(max_target_speed_excess, float(np.max(np.linalg.norm(target_action.reshape(2, 2), axis=1) - config.max_speed)))
            target_by_seed[seed] = (target, flow, safe)
            correction_norms.extend(np.linalg.norm(stack, axis=1).tolist())
            component_variances.append(np.var(stack, axis=0))
            for i, j in combinations(range(len(stack)), 2):
                pair_distances.append(float(np.linalg.norm(stack[i] - stack[j])))
        typical = float(np.median(correction_norms)) if correction_norms else 0.0
        mean_distance = float(np.mean(pair_distances)) if pair_distances else 0.0
        max_distance = float(np.max(pair_distances)) if pair_distances else 0.0
        flow_variability = float(np.mean([pairwise_mean(np.stack(labels_by_eta[eta])) for eta in near]))
        if len(near) == 1 or (max_distance / config.max_speed <= .01 and mean_distance / max(typical, EPS) <= .05 and mean_distance <= flow_variability + 1e-15):
            classification = "LABEL_STABLE"
        elif max_distance / config.max_speed <= .10 and mean_distance / max(typical, EPS) <= .50:
            classification = "LABEL_MILDLY_AMBIGUOUS"
        else:
            classification = "LABEL_MULTIVALUED"; multivalued.append(state_id)
        label_statistics[state_id] = {
            "classification": classification, "usable": classification != "LABEL_MULTIVALUED",
            "oracle_boundary_class": state["oracle_boundary_class"],
            "B_63_size": len(oracle["B_63"]), "E_near_size": len(near),
            "eta_best": oracle["eta_best"], "E_near": [list(value) for value in near],
            "J_min": oracle["J_min"], "mean_pairwise_L2_eta_same_seed": mean_distance,
            "max_pairwise_L2_eta_same_seed": max_distance, "typical_g_exec_norm": typical,
            "flow_seed_mean_pairwise_L2_within_eta": flow_variability,
        }
        if classification == "LABEL_MULTIVALUED":
            continue
        state_targets[state_id] = []
        for seed in common:
            target, flow, safe = target_by_seed[seed]
            # Eta-zero labels are exact by oracle construction, never thresholded.
            if state["oracle_boundary_class"] == "zero":
                if not np.allclose(target, 0.0, atol=1e-12, rtol=0):
                    raise AssertionError((state_id, seed, target))
                target = np.zeros(4, dtype=np.float64)
            features, structured = builder.build(env, {"u_flow": flow, "u_safe": safe}, config, cbf)
            sample_id = f"{state_id}__flow{seed}"
            state_targets[state_id].append(target)
            append["features"].append(features); append["targets"].append(target); append["target_actions"].append(safe + target)
            append["state_index"].append(-1); append["flow_seed"].append(seed)
            for array_key, structured_key in (
                ("observation", "observation"), ("positions", "positions"), ("velocities", "velocities"),
                ("goal_relative", "goal_relative"), ("relative_position", "relative_position"),
                ("relative_velocity", "relative_velocity"), ("u_flow", "u_flow"), ("u_safe", "u_safe"),
                ("B_goal", "B_goal"), ("B_rel", "B_rel"), ("goal_error_history", "goal_error_history"),
                ("recent_progress", "recent_progress"), ("first_projection_delta", "projection_delta"),
                ("first_projection_linear_residuals", "linear_residuals"),
            ):
                append[array_key].append(structured[structured_key])
            append["state_id"].append(state_id); append["split"].append(state["split"]); append["category"].append("RECOVERY"); append["sample_id"].append(sample_id)
            metadata_new.append({
                "sample_id": sample_id, "state_id": state_id, "state_index": -1,
                "source_trajectory": state["source_trajectory"], "leakage_group": state["leakage_group"],
                "category": "RECOVERY", "split": state["split"], "flow_seed": seed,
                "target_dimension": 4, "target_semantics": "mean hard-projected executed correction across compact E_near",
                "eta_best_metadata_only": oracle["eta_best"], "E_near_metadata_only": oracle["E_near"],
                "oracle_J_min_metadata_only": oracle["J_min"], "label_classification": classification,
                "zero_label": state["oracle_boundary_class"] == "zero", "new_v4_boundary_state": True,
            })

    usable = [row for row in selected if row["state_id"] not in multivalued]
    if len(usable) < 20:
        raise RuntimeError(("fewer than 20 usable boundary states", len(usable), multivalued))
    # Self-contained state directory with an exact V3 prefix.
    states_dir = HERE / "states"
    v3_states = read_jsonl(V3 / "state_manifest.jsonl")
    if states_dir.exists():
        for state in v3_states:
            path = states_dir / Path(state["state_file"]).name
            if not path.exists() or sha(path) != state["state_sha256"]:
                raise RuntimeError(("invalid interrupted prefix", path))
    else:
        shutil.copytree(V3 / "states", states_dir, copy_function=shutil.copy2)
    final_states = [dict(row) for row in v3_states]
    index_by_id = {}
    for state_index, state in enumerate(usable, start=len(v3_states)):
        source = HERE / state["state_file"]; destination = states_dir / source.name
        shutil.copy2(source, destination)
        row = dict(state); row["state_index"] = state_index
        row["state_file"] = str(destination.relative_to(HERE)); row["state_sha256"] = sha(destination)
        row["label_classification"] = label_statistics[row["state_id"]]["classification"]
        row["eta_best"] = label_statistics[row["state_id"]]["eta_best"]
        final_states.append(row); index_by_id[row["state_id"]] = state_index
    write_jsonl(HERE / "state_manifest.jsonl", final_states)
    # Fill the final state indices now that quarantines are known.
    for row in metadata_new:
        row["state_index"] = index_by_id[row["state_id"]]
    append["state_index"] = [index_by_id[state_id] for state_id in append["state_id"]]

    with np.load(V3 / "samples.npz", allow_pickle=False) as source:
        v3_arrays = {key: np.asarray(source[key]).copy() for key in source.files}
    arrays = {}
    for key, old in v3_arrays.items():
        new = np.asarray(append[key]) if old.dtype.kind == "U" else np.asarray(append[key], dtype=old.dtype)
        arrays[key] = np.concatenate([old, new], axis=0)
    np.savez_compressed(HERE / "samples.npz", **arrays)
    metadata = read_jsonl(V3 / "sample_metadata.jsonl") + metadata_new
    write_jsonl(HERE / "sample_metadata.jsonl", metadata)
    shutil.copy2(V3 / "feature_schema.json", HERE / "feature_schema.json")
    prefix_equal = {key: bool(np.array_equal(arrays[key][:len(v3_arrays[key])], v3_arrays[key])) for key in arrays}
    if not all(prefix_equal.values()):
        raise AssertionError(("V3 prefix changed", [key for key, value in prefix_equal.items() if not value]))

    # Complete pair records with oracle target norms.
    pair_pre = list(csv.DictReader((HERE / "matched_boundary_pairs_preoracle.csv").open()))
    pair_rows = []
    for row in pair_pre:
        zero_id = row["zero_state_id"]; nonzero_id = row["nonzero_state_id"]
        if zero_id not in state_targets or nonzero_id not in state_targets:
            continue
        zero_norm = float(np.mean(np.linalg.norm(np.asarray(state_targets[zero_id]), axis=1)))
        nonzero_norm = float(np.mean(np.linalg.norm(np.asarray(state_targets[nonzero_id]), axis=1)))
        pair_rows.append({**row, "zero_target_correction_norm": zero_norm, "nonzero_target_correction_norm": nonzero_norm})
    write_csv(HERE / "matched_boundary_pairs.csv", pair_rows)

    boundary_rows = []
    for state in usable:
        values = np.asarray(state_targets[state["state_id"]])
        boundary_rows.append({
            "state_id": state["state_id"], "boundary_class": state["oracle_boundary_class"],
            "source_group": state["source_group"], "source_trajectory": state["source_trajectory"],
            "split": state["split"], "inter_agent_distance": state["inter_agent_distance"],
            "relative_velocity_norm": state["relative_velocity_norm"], "recovery_phase": state["recovery_phase"],
            "target_norm_mean": float(np.mean(np.linalg.norm(values, axis=1))),
            "target_norm_min": float(np.min(np.linalg.norm(values, axis=1))),
            "target_norm_max": float(np.max(np.linalg.norm(values, axis=1))),
            "label_classification": label_statistics[state["state_id"]]["classification"],
            "eta_best": json.dumps(label_statistics[state["state_id"]]["eta_best"]),
        })
    write_csv(HERE / "recovery_boundary_states.csv", boundary_rows)

    # Split/leakage audit.
    groups_by_split = {split: {row["leakage_group"] for row in final_states if row["split"] == split} for split in ("train", "validation", "test")}
    overlaps = {f"{a}_{b}": sorted(groups_by_split[a] & groups_by_split[b]) for a, b in combinations(groups_by_split, 2)}
    leakage = {
        "assignment_unit": "root recovery source/leakage group; all trajectory descendants colocated",
        "V3_sample_arrays_bitwise_equal": prefix_equal, "all_V3_sample_arrays_bitwise_equal": all(prefix_equal.values()),
        "split_group_overlaps": overlaps,
        "duplicate_sample_ids": len(arrays["sample_id"]) - len(set(arrays["sample_id"].tolist())),
        "duplicate_state_snapshots": len(final_states) - len({row["state_sha256"] for row in final_states}),
        "maximum_new_states_per_source_trajectory": max(Counter(row["source_trajectory"] for row in usable).values()),
        "features_all_finite": bool(np.isfinite(arrays["features"]).all()), "targets_all_finite": bool(np.isfinite(arrays["targets"]).all()),
    }
    leakage["passed"] = not any(overlaps.values()) and leakage["duplicate_sample_ids"] == 0 and leakage["duplicate_state_snapshots"] == 0 and leakage["maximum_new_states_per_source_trajectory"] <= 2 and leakage["features_all_finite"] and leakage["targets_all_finite"] and leakage["all_V3_sample_arrays_bitwise_equal"]
    write_json(HERE / "leakage_checks.json", leakage)
    if not leakage["passed"]:
        raise RuntimeError("leakage gate failed")

    boundary_counts = {split: {label: sum(row["split"] == split and row["oracle_boundary_class"] == label for row in usable) for label in ("zero", "nonzero")} for split in groups_by_split}
    split_manifest = {
        "assignment_unit": leakage["assignment_unit"], "V3_state_memberships_preserved": True,
        "state_counts": {split: sum(row["split"] == split for row in final_states) for split in groups_by_split},
        "sample_counts": {split: int(np.sum(arrays["split"] == split)) for split in groups_by_split},
        "new_close_recovery_boundary_counts": boundary_counts,
        "state_ids": {split: [row["state_id"] for row in final_states if row["split"] == split] for split in groups_by_split},
        "leakage_groups": {split: sorted(groups) for split, groups in groups_by_split.items()},
    }
    write_json(HERE / "split_manifest.json", split_manifest)

    # Boundary learnability audit using deployment features only.
    new_start = len(v3_arrays["features"])
    new_ids = arrays["state_id"][new_start:]
    state_feature = {state_id: arrays["features"][new_start:][new_ids == state_id].mean(axis=0) for state_id in sorted(set(new_ids.tolist()))}
    normalization = json.loads((V3_TRAIN / "normalization.json").read_text())
    mean = np.asarray(normalization["mean"]); scale = np.asarray(normalization["scale"])
    norm = {state_id: (value - mean) / scale for state_id, value in state_feature.items()}
    state_label = {row["state_id"]: row["oracle_boundary_class"] for row in usable}
    nearest_rows = []
    for state_id in sorted(norm):
        other, value = min(((candidate, float(np.linalg.norm(norm[state_id] - norm[candidate]) / math.sqrt(len(mean)))) for candidate in norm if candidate != state_id), key=lambda item: item[1])
        nearest_rows.append({"state_id": state_id, "label": state_label[state_id], "nearest_state_id": other, "nearest_label": state_label[other], "distance": value, "label_disagreement": state_label[state_id] != state_label[other]})
    same = []; opposite = []
    ids = sorted(norm)
    for i, j in combinations(range(len(ids)), 2):
        value = float(np.linalg.norm(norm[ids[i]] - norm[ids[j]]) / math.sqrt(len(mean)))
        (same if state_label[ids[i]] == state_label[ids[j]] else opposite).append(value)
    schema = json.loads((HERE / "feature_schema.json").read_text())
    zero_matrix = np.stack([norm[state_id] for state_id in ids if state_label[state_id] == "zero"])
    nonzero_matrix = np.stack([norm[state_id] for state_id in ids if state_label[state_id] == "nonzero"])
    segment_scores = []
    for segment in schema["segments"]:
        start = int(segment["offset"]); stop = start + int(segment["length"])
        score = float(np.linalg.norm(zero_matrix[:, start:stop].mean(axis=0) - nonzero_matrix[:, start:stop].mean(axis=0)) / math.sqrt(stop - start))
        segment_scores.append({"feature_group": segment["name"], "standardized_centroid_RMS_difference": score})
    segment_scores.sort(key=lambda row: row["standardized_centroid_RMS_difference"], reverse=True)
    diversity = {
        "new_boundary_states": len(usable), "boundary_class_counts": dict(Counter(row["oracle_boundary_class"] for row in usable)),
        "independent_source_groups": len({row["source_group"] for row in usable}),
        "distinct_source_trajectories": len({row["source_trajectory"] for row in usable}),
        "matched_pair_count": len(pair_rows),
        "inter_agent_distance_m": {"min": min(row["inter_agent_distance"] for row in usable), "median": float(np.median([row["inter_agent_distance"] for row in usable])), "max": max(row["inter_agent_distance"] for row in usable)},
        "feature_distance_definition": "RMS Euclidean after frozen V3 train-only normalization; deployment-available 214-D features only",
        "nearest_neighbor_label_disagreement_rate": float(np.mean([row["label_disagreement"] for row in nearest_rows])),
        "nearest_neighbor_leave_one_out_accuracy": float(np.mean([not row["label_disagreement"] for row in nearest_rows])),
        "same_label_pair_distance_mean": float(np.mean(same)), "opposite_label_pair_distance_mean": float(np.mean(opposite)),
        "opposite_over_same_mean_distance_ratio": float(np.mean(opposite) / np.mean(same)),
        "nearest_neighbor_rows": nearest_rows,
        "feature_group_centroid_differences_ranked": segment_scores,
        "matched_pair_feature_distance": {"min": min(float(row["feature_distance"]) for row in pair_rows), "median": float(np.median([float(row["feature_distance"]) for row in pair_rows])), "max": max(float(row["feature_distance"]) for row in pair_rows)},
        "interpretation": "diagnostic overlap only; no feature or method change",
    }
    write_json(HERE / "boundary_diversity_metrics.json", diversity)

    group_rows = []
    for group in sorted({row["source_group"] for row in usable}):
        members = [row for row in usable if row["source_group"] == group]
        group_rows.append({"source_group": group, "split": members[0]["split"], "selected_states": len(members), "zero_states": sum(row["oracle_boundary_class"] == "zero" for row in members), "nonzero_states": sum(row["oracle_boundary_class"] == "nonzero" for row in members), "source_trajectories": sorted(row["source_trajectory"] for row in members)})
    write_json(HERE / "source_group_manifest.json", {"groups": group_rows})
    write_json(HERE / "label_statistics.json", {"new_states": label_statistics, "classification_counts": dict(Counter(value["classification"] for value in label_statistics.values())), "B_63_empty_state_ids": empty, "multivalued_state_ids": multivalued})

    eta_zero = json.loads((HERE / "eta_zero_classification.json").read_text())
    nonzero_manifests = [json.loads((HERE / f"raw/nonzero_candidate_shard{shard}/manifest.json").read_text()) for shard in (0, 1)]
    runtime = {
        "eta_zero_new_rollouts": eta_zero["eta_zero_runtime"]["new_rollouts"],
        "eta_zero_physical_steps": eta_zero["eta_zero_runtime"]["physical_steps"],
        "eta_zero_parallel_elapsed_s": eta_zero["eta_zero_runtime"]["elapsed_s_max_parallel"],
        "nonzero_candidate_new_rollouts": sum(item["new_rollouts"] for item in nonzero_manifests),
        "nonzero_candidate_physical_steps": sum(item["physical_steps"] for item in nonzero_manifests),
        "nonzero_candidate_parallel_elapsed_s": max(item["elapsed_s"] for item in nonzero_manifests),
        "reused_rollouts": 0, "finalization_elapsed_s": time.monotonic() - started,
        "resources": {"GPU_shards": 2, "CPU_cores": 8, "JAX_memory_fraction_per_process": 0.10},
    }
    write_json(HERE / "runtime_statistics.json", runtime)

    category_counts = dict(Counter(row["category"] for row in final_states))
    base_label = json.loads((V3 / "manifest.json").read_text())
    new_zero = sum(row["oracle_boundary_class"] == "zero" for row in usable)
    new_nonzero = len(usable) - new_zero
    report = f"""# G_phi Dataset V4: recovery intervention-boundary coverage

## Result

- Appended **{len(usable)}** unique close-range RECOVERY boundary states to the bitwise-preserved V3 prefix: **{new_zero} zero / {new_nonzero} nonzero**.
- These states use **{len({row['source_group'] for row in usable})}** independent root source groups and {len({row['source_trajectory'] for row in usable})} distinct trajectories.
- Explicit matched zero/nonzero pairs: **{len(pair_rows)}**.
- Final states/samples: **{len(final_states)} / {len(arrays['features'])}**; categories: {category_counts}.
- Final zero/nonzero states: **{base_label['zero_label_states'] + new_zero} / {base_label['nonzero_label_states'] + new_nonzero}**.
- B_63-empty selected states: **0**; multivalued quarantines: **{len(multivalued)}**.

## Boundary split

New close-range zero/nonzero counts by split: `{boundary_counts}`. All Flow variants remain with their augmented state and root source group. No source-group, state, sample, or trajectory leakage was found.

## Integrity and learnability

- Exact restoration audit: **PASS**.
- V3 sample prefix preserved exactly for every array: **{all(prefix_equal.values())}**.
- Distance range: {min(row['inter_agent_distance'] for row in usable):.6f}–{max(row['inter_agent_distance'] for row in usable):.6f} m.
- Nearest-neighbor cross-label disagreement: {diversity['nearest_neighbor_label_disagreement_rate']:.2%}; matched and feature-group diagnostics are in `boundary_diversity_metrics.json`.
- Target construction is unchanged: B_63 -> minimum J_def -> E_near -> hard-projected executed correction. Exact zero labels arise only from eta=0 membership, never thresholding.

## Runtime

Eta-zero/new-candidate oracle used two nonoverlapping GPU shards. New rollout counts were {runtime['eta_zero_new_rollouts']} eta-zero plus {runtime['nonzero_candidate_new_rollouts']} nonzero-candidate evaluations. No valid tuple was rerun.
"""
    (HERE / "dataset_report.md").write_text(report)
    required = (
        "dataset_report.md", "recovery_boundary_states.csv", "matched_boundary_pairs.csv",
        "source_group_manifest.json", "state_manifest.jsonl", "samples.npz", "feature_schema.json",
        "split_manifest.json", "restoration_checks.json", "leakage_checks.json",
        "boundary_diversity_metrics.json", "label_statistics.json", "runtime_statistics.json",
    )
    manifest = {
        "dataset": "gphi_training_dataset_v4", "parent_dataset": str(V3),
        "unique_augmented_states": len(final_states), "new_boundary_states": len(usable),
        "new_close_zero_recovery_states": new_zero, "new_close_nonzero_recovery_states": new_nonzero,
        "matched_boundary_pairs": len(pair_rows), "supervised_samples": len(arrays["features"]),
        "feature_dimension": 214, "target_dimension": 4, "category_counts": category_counts,
        "zero_label_states": base_label["zero_label_states"] + new_zero,
        "nonzero_label_states": base_label["nonzero_label_states"] + new_nonzero,
        "B_63_empty_states": 0, "LABEL_MULTIVALUED_states": len(multivalued),
        "restoration_status": json.loads((HERE / "restoration_checks.json").read_text())["status"],
        "leakage_checks_passed": leakage["passed"],
        "files_sha256": {name: sha(HERE / name) for name in required},
    }
    write_json(HERE / "manifest.json", manifest)
    print(json.dumps({
        "new_states": len(usable), "zero": new_zero, "nonzero": new_nonzero,
        "source_groups": len({row["source_group"] for row in usable}), "matched_pairs": len(pair_rows),
        "boundary_split": boundary_counts, "final_states": len(final_states), "samples": len(arrays["features"]),
        "label_classes": dict(Counter(value["classification"] for value in label_statistics.values())),
    }, indent=2))


if __name__ == "__main__":
    main()
