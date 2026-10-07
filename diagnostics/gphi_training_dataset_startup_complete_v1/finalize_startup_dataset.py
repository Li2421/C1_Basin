"""Reconstruct V3-style startup labels and append them to V3 bitwise."""

from __future__ import annotations

import csv
import inspect
import json
import math
import shutil
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import t

from audit_startup_states import restore_full
from common import (
    HERE, ROOT, SYSROOT, V3, assert_frozen_sources, eta_key,
    load_effective_records, read_jsonl, sha256, write_json, write_jsonl,
)


VMAX_EPS = 1e-12


def pairwise_mean(vectors: np.ndarray) -> float:
    if len(vectors) < 2:
        return 0.0
    distances = np.linalg.norm(vectors[:, None] - vectors[None, :], axis=-1)
    upper = np.triu_indices(len(vectors), 1)
    return float(distances[upper].mean())


def paired_comparison(candidate: list[dict], best: list[dict]) -> dict:
    a = {int(row["seed"]): row for row in candidate if row["outcome"] == "success"}
    b = {int(row["seed"]): row for row in best if row["outcome"] == "success"}
    seeds = sorted(set(a) & set(b))
    differences = np.asarray([float(a[s]["J_def"]) - float(b[s]["J_def"]) for s in seeds])
    if not len(differences):
        return {"paired_n": 0, "candidate_minus_best_mean": None, "two_sided_paired_t_95_CI": None, "indistinguishable_from_zero": False}
    mean = float(differences.mean())
    if len(differences) == 1 or float(differences.std(ddof=1)) == 0.0:
        interval = [mean, mean]
    else:
        half = float(t.ppf(0.975, len(differences) - 1) * differences.std(ddof=1) / math.sqrt(len(differences)))
        interval = [mean - half, mean + half]
    return {"paired_n": len(differences), "candidate_minus_best_mean": mean, "two_sided_paired_t_95_CI": interval, "indistinguishable_from_zero": interval[0] <= 0 <= interval[1]}


def assert_dataset_alignment(
    arrays: dict[str, np.ndarray], metadata: list[dict], states: list[dict], label: str,
) -> None:
    """Prove that arrays, sample metadata, and state metadata describe one split."""
    required = {"state_id", "state_index", "sample_id", "split", "category"}
    if not required.issubset(arrays):
        raise RuntimeError((label, "missing alignment arrays", sorted(required - set(arrays))))
    sample_count = len(arrays["state_id"])
    if any(len(value) != sample_count for value in arrays.values()) or len(metadata) != sample_count:
        raise RuntimeError((label, "inconsistent sample/metadata lengths"))
    state_by_id: dict[str, dict] = {}
    state_by_index: dict[int, dict] = {}
    for state in states:
        state_id = str(state["state_id"])
        state_index = int(state["state_index"])
        if state_id in state_by_id or state_index in state_by_index:
            raise RuntimeError((label, "duplicate state ID/index", state_id, state_index))
        state_by_id[state_id] = state
        state_by_index[state_index] = state
    sample_ids = [str(value) for value in arrays["sample_id"].tolist()]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError((label, "duplicate sample IDs"))
    represented_states = set()
    for index, meta in enumerate(metadata):
        state_id = str(arrays["state_id"][index])
        represented_states.add(state_id)
        if state_id not in state_by_id:
            raise RuntimeError((label, "sample references unknown state", state_id))
        state = state_by_id[state_id]
        array_values = {
            "sample_id": str(arrays["sample_id"][index]),
            "state_id": state_id,
            "state_index": int(arrays["state_index"][index]),
            "split": str(arrays["split"][index]),
            "category": str(arrays["category"][index]),
        }
        for field, value in array_values.items():
            if field not in meta or (int(meta[field]) if field == "state_index" else str(meta[field])) != value:
                raise RuntimeError((label, "array/sample-metadata mismatch", index, field))
        for field in ("state_index", "split", "category"):
            meta_value = int(meta[field]) if field == "state_index" else str(meta[field])
            state_value = int(state[field]) if field == "state_index" else str(state[field])
            if meta_value != state_value:
                raise RuntimeError((label, "sample/state-metadata mismatch", index, field))
        for field in ("leakage_group", "source_trajectory"):
            if str(meta.get(field, "")) != str(state.get(field, "")):
                raise RuntimeError((label, "sample/state provenance mismatch", index, field))
        if state_by_index[array_values["state_index"]]["state_id"] != state_id:
            raise RuntimeError((label, "state_index maps to wrong state", index))
    if represented_states != set(state_by_id):
        raise RuntimeError((label, "state manifest/sample state set mismatch"))


def build_oracles(effective: dict, states: list[dict]) -> tuple[list[dict], dict]:
    grouped = defaultdict(list)
    for (state_id, eta, _), row in effective.items():
        grouped[(state_id, eta)].append(row)
    results = []
    for state in states:
        cells = []
        for (state_id, eta), rows in sorted(grouped.items()):
            if state_id != state["state_id"]:
                continue
            rows = sorted(rows, key=lambda row: int(row["seed"])); counts = Counter(row["outcome"] for row in rows)
            complete = len(rows) == 64 and not any(row.get("execution_error") for row in rows)
            successes = [row for row in rows if row["outcome"] == "success"]
            cells.append({
                "eta": list(eta), "evaluated": len(rows), "success": counts["success"],
                "deadlock": counts["deadlock"], "timeout": counts["timeout"], "collision": counts["collision"],
                "execution_error": counts["execution_error"], "complete_matched_64": complete,
                "B_63_member": complete and counts["success"] >= 63,
                "mean_J_def_success": float(np.mean([row["J_def"] for row in successes])) if successes else None,
            })
        feasible = [cell for cell in cells if cell["B_63_member"]]
        if not feasible:
            results.append({"state_id": state["state_id"], "category": "STARTUP", "B_63_empty": True, "B_63": [], "eta_best": None, "E_near": [], "cells": cells}); continue
        best = min(feasible, key=lambda cell: (cell["mean_J_def_success"], cell["eta"]))
        best_eta = eta_key(best["eta"]); best_rows = grouped[(state["state_id"], best_eta)]
        near = []; comparisons = []
        for cell in feasible:
            eta = eta_key(cell["eta"])
            comparison = ({"paired_n": sum(row["outcome"] == "success" for row in best_rows), "candidate_minus_best_mean": 0.0, "two_sided_paired_t_95_CI": [0.0, 0.0], "indistinguishable_from_zero": True} if eta == best_eta else paired_comparison(grouped[(state["state_id"], eta)], best_rows))
            comparisons.append({"eta": list(eta), **comparison})
            if comparison["indistinguishable_from_zero"]:
                near.append(list(eta))
        results.append({"state_id": state["state_id"], "category": "STARTUP", "B_63_empty": False, "B_63": [cell["eta"] for cell in feasible], "eta_best": best["eta"], "J_min": best["mean_J_def_success"], "E_near": near, "paired_comparisons": comparisons, "cells": cells})
    return results, {row["state_id"]: row for row in results}


def load_builder():
    import startup_feature_builder as module
    for name in ("StartupAwareFeatureBuilder", "StartupFeatureBuilder"):
        if hasattr(module, name):
            return getattr(module, name)()
    raise ImportError("startup_feature_builder.py must expose StartupAwareFeatureBuilder or StartupFeatureBuilder")


def main() -> None:
    started = time.monotonic(); frozen_hashes = assert_frozen_sources()
    integrity = json.loads((HERE / "startup_state_integrity_checks.json").read_text())
    if integrity["status"] != "PASS":
        raise RuntimeError("startup state integrity did not pass")
    protocol = json.loads((HERE / "protocol.json").read_text())
    frozen_v3 = {
        "manifest.json": protocol["base_v3_manifest_sha256"],
        "samples.npz": protocol["base_v3_samples_sha256"],
        "state_manifest.jsonl": protocol["base_v3_state_manifest_sha256"],
    }
    for name, expected_hash in frozen_v3.items():
        path = V3 / name
        if not path.is_file() or sha256(path) != expected_hash:
            raise RuntimeError(f"V3 {name} changed since startup protocol freeze")
    # Resolve the authoritative controller diagnostics independently of the
    # caller's working directory/PYTHONPATH, while keeping SYSROOT first for
    # the frozen single_integrator implementation.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if str(SYSROOT) in sys.path:
        sys.path.remove(str(SYSROOT))
    sys.path.insert(0, str(SYSROOT))
    from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi
    from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
    from single_integrator.cbf import CBFConfig, barrier_constraints
    from single_integrator.environment import Config

    if Path(inspect.getsourcefile(barrier_constraints)).resolve() != (SYSROOT / "single_integrator/cbf.py").resolve():
        raise RuntimeError("alternate CBF/projection implementation imported")
    config = Config(**protocol["environment"]); cbf = CBFConfig()
    startup_states = read_jsonl(HERE / "startup_state_manifest.jsonl")
    effective, duplicates = load_effective_records()
    oracle_rows, oracle_by_state = build_oracles(effective, startup_states)
    write_jsonl(HERE / "startup_oracle_search_results.jsonl", oracle_rows)
    empty = [row["state_id"] for row in oracle_rows if row["B_63_empty"]]
    startup_by_state = {row["state_id"]: row for row in startup_states}
    write_json(HERE / "b63_empty_quarantine.json", {
        "status": "QUARANTINED_EXCLUDED_FROM_SUPERVISED_DATASET" if empty else "NONE",
        "policy": "A B_63-empty state has no defensible oracle label and is retained only in frozen-candidate/oracle audit artifacts.",
        "frozen_startup_candidate_count": len(startup_states),
        "B_63_empty_state_count": len(empty),
        "B_63_empty_state_ids": empty,
        "quarantined_states": [
            {
                "state_id": state_id,
                "state_index_startup": startup_by_state[state_id]["state_index_startup"],
                "source_episode": startup_by_state[state_id]["source_episode"],
                "physical_step": startup_by_state[state_id]["step"],
                "split": startup_by_state[state_id]["split"],
                "reason": "B_63_EMPTY_NO_ORACLE_LABEL",
                "oracle_result": oracle_by_state[state_id],
            }
            for state_id in empty
        ],
        "traceability": {
            "all_frozen_candidates": "startup_state_manifest.jsonl",
            "all_oracle_results": "startup_oracle_search_results.jsonl",
        },
    })

    builder = load_builder(); arrays_new = defaultdict(list); metadata_new = []
    label_statistics = {}; validated_states = []; quarantined = []
    max_first_replay = 0.0; max_second_replay = 0.0; max_formula = 0.0
    min_target_residual = float("inf"); max_speed_excess = -float("inf")
    for state in startup_states:
        state_id = state["state_id"]; oracle = oracle_by_state[state_id]
        if oracle["B_63_empty"]:
            label_statistics[state_id] = {
                "classification": "B63_EMPTY",
                "usable": False,
                "exclusion_reason": "No B_63-feasible eta; no supervised label assigned.",
                "B_63_size": 0,
                "E_near_size": 0,
                "eta_values_evaluated": len(oracle["cells"]),
                "continuation_records_considered": sum(cell["evaluated"] for cell in oracle["cells"]),
            }
            continue
        near = [eta_key(eta) for eta in oracle["E_near"]]
        eta_rows = {eta: {int(seed): effective[(state_id, eta, int(seed))] for seed in sorted(key[2] for key in effective if key[0] == state_id and key[1] == eta)} for eta in near}
        common = sorted(set.intersection(*(set(rows) for rows in eta_rows.values())))
        if len(common) != 64:
            raise RuntimeError((state_id, "E_near lacks matched 64", len(common)))
        env = restore_full(HERE / state["state_file"], config)
        observation = np.asarray(env.observation(), dtype=np.float64)
        A, lower, _ = barrier_constraints(env.snapshot(), cbf)
        target_by_seed = {}; labels_by_eta = defaultdict(list); pair_distances = []; correction_norms = []; component_variances = []
        for seed in common:
            first_steps = [eta_rows[eta][seed]["first_step"] for eta in near]
            if any(item is None for item in first_steps):
                raise RuntimeError((state_id, seed, "missing first-step record"))
            safe_stack = np.stack([np.asarray(item["u_safe"], dtype=np.float64).reshape(4) for item in first_steps])
            flow_stack = np.stack([np.asarray(item["u_flow"], dtype=np.float64).reshape(4) for item in first_steps])
            if np.max(np.ptp(safe_stack, axis=0)) > 1e-10 or np.max(np.ptp(flow_stack, axis=0)) > 1e-10:
                raise RuntimeError((state_id, seed, "same-seed Flow/first projection mismatch across eta"))
            safe = safe_stack[0].reshape(2, 2); flow = flow_stack[0].reshape(2, 2)
            safe_replay, _, _, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
            max_first_replay = max(max_first_replay, float(np.max(np.abs(safe_replay - safe))))
            executed_labels = []
            for eta, first in zip(near, first_steps):
                raw = np.asarray(first["g_raw"], dtype=np.float64).reshape(2, 2)
                executed = np.asarray(first["u_exec"], dtype=np.float64).reshape(2, 2)
                formula = DiagnosticCorrector(DiagnosticPhi(*eta))(observation, safe, config.max_speed)
                max_formula = max(max_formula, float(np.max(np.abs(formula - raw))))
                replay, _, _, _ = project_velocity_with_retry(safe + raw, A, lower, config.max_speed, cbf)
                max_second_replay = max(max_second_replay, float(np.max(np.abs(replay - executed))))
                label = (executed - safe).reshape(4); executed_labels.append(label); labels_by_eta[eta].append(label)
            stack = np.stack(executed_labels); target = stack.mean(axis=0); action = safe.reshape(4) + target
            target_by_seed[seed] = {"target": target, "u_flow": flow.reshape(4), "u_safe": safe.reshape(4)}
            correction_norms.extend(np.linalg.norm(stack, axis=1).tolist()); component_variances.append(np.var(stack, axis=0))
            for i, j in combinations(range(len(near)), 2): pair_distances.append(float(np.linalg.norm(stack[i] - stack[j])))
            min_target_residual = min(min_target_residual, float(np.min(A @ action - lower)))
            max_speed_excess = max(max_speed_excess, float(np.max(np.linalg.norm(action.reshape(2, 2), axis=-1) - config.max_speed)))
        typical = float(np.median(correction_norms)); mean_distance = float(np.mean(pair_distances)) if pair_distances else 0.0; max_distance = max(pair_distances, default=0.0)
        flow_variability = float(np.mean([pairwise_mean(np.stack(labels_by_eta[eta])) for eta in near]))
        if len(near) == 1 or (max_distance / config.max_speed <= .01 and mean_distance / max(typical, VMAX_EPS) <= .05 and mean_distance <= flow_variability + 1e-15): classification = "LABEL_STABLE"
        elif max_distance / config.max_speed <= .10 and mean_distance / max(typical, VMAX_EPS) <= .50: classification = "LABEL_MILDLY_AMBIGUOUS"
        else: classification = "LABEL_MULTIVALUED"
        label_statistics[state_id] = {"classification": classification, "usable": classification != "LABEL_MULTIVALUED", "B_63_size": len(oracle["B_63"]), "E_near_size": len(near), "eta_best": oracle["eta_best"], "E_near": oracle["E_near"], "J_min": oracle["J_min"], "mean_pairwise_L2_eta_same_seed": mean_distance, "max_pairwise_L2_eta_same_seed": max_distance, "flow_seed_mean_pairwise_L2_within_eta": flow_variability, "typical_g_exec_norm": typical, "componentwise_variance_due_to_eta_mean_over_seeds": np.mean(component_variances, axis=0).tolist()}
        if classification == "LABEL_MULTIVALUED": quarantined.append(state_id); continue
        validated_states.append(state)
        for seed in common:
            source = target_by_seed[seed]
            features, structured = builder.build(env, {"u_flow": source["u_flow"], "u_safe": source["u_safe"]}, config, cbf)
            if features.shape != (214,) or not np.isfinite(features).all():
                raise RuntimeError((state_id, seed, "invalid startup feature", features.shape))
            target = source["target"]; sample_id = f"{state_id}__flow{seed}"
            metadata_new.append({"sample_id": sample_id, "state_id": state_id, "source_trajectory": state["source_trajectory"], "leakage_group": state["leakage_group"], "category": "STARTUP", "split": state["split"], "flow_seed": seed, "target_dimension": 4, "target_semantics": "mean executed correction u_exec-u_safe across compact E_near at identical startup state/Flow seed", "eta_best_metadata_only": oracle["eta_best"], "E_near_metadata_only": oracle["E_near"], "oracle_J_min_metadata_only": oracle["J_min"], "label_classification": classification, "zero_label": bool(np.linalg.norm(target) <= 1e-14), "physical_step": state["step"], "startup_left_padding_count": state["left_padding_count"]})
            for key, value in (("features", features), ("targets", target), ("target_actions", source["u_safe"] + target), ("flow_seed", seed), ("observation", structured["observation"]), ("positions", structured["positions"]), ("velocities", structured["velocities"]), ("goal_relative", structured["goal_relative"]), ("relative_position", structured["relative_position"]), ("relative_velocity", structured["relative_velocity"]), ("u_flow", structured["u_flow"]), ("u_safe", structured["u_safe"]), ("B_goal", structured["B_goal"]), ("B_rel", structured["B_rel"]), ("goal_error_history", structured["goal_error_history"]), ("recent_progress", structured["recent_progress"]), ("first_projection_delta", structured["projection_delta"]), ("first_projection_linear_residuals", structured["linear_residuals"]), ("state_id", state_id), ("split", state["split"]), ("category", "STARTUP"), ("sample_id", sample_id)):
                arrays_new[key].append(value)

    if not validated_states:
        raise RuntimeError("no validated startup states")
    with np.load(V3 / "samples.npz", allow_pickle=False) as source:
        v3_arrays = {key: np.asarray(source[key]).copy() for key in source.files}
    v3_states = read_jsonl(V3 / "state_manifest.jsonl")
    v3_metadata = read_jsonl(V3 / "sample_metadata.jsonl")
    assert_dataset_alignment(v3_arrays, v3_metadata, v3_states, "V3")
    v3_count = len(v3_states)
    startup_index = {row["state_id"]: v3_count + i for i, row in enumerate(validated_states)}
    arrays_new["state_index"] = [startup_index[row["state_id"]] for row in metadata_new]
    for row in metadata_new: row["state_index"] = startup_index[row["state_id"]]
    merged = {}
    for key, old in v3_arrays.items():
        new = np.asarray(arrays_new[key]) if old.dtype.kind == "U" else np.asarray(arrays_new[key], dtype=old.dtype)
        merged[key] = np.concatenate([old, new], axis=0)
        if not np.array_equal(merged[key][:len(old)], old):
            raise AssertionError((key, "V3 prefix changed"))
    final_metadata = v3_metadata + metadata_new
    metadata_prefix_exact = final_metadata[:len(v3_metadata)] == v3_metadata
    if not metadata_prefix_exact:
        raise AssertionError("V3 sample-metadata prefix changed")

    # Self-contained snapshots; copy bytes and preserve all V3 content hashes.
    states_dir = HERE / "states"; final_states = []
    for row in v3_states:
        source = V3 / row["state_file"]; destination = states_dir / source.name
        if not source.is_file() or sha256(source) != row["state_sha256"]:
            raise RuntimeError((source, "V3 snapshot hash mismatch"))
        if destination.exists() and sha256(destination) != row["state_sha256"]: raise RuntimeError((destination, "conflicting snapshot"))
        if not destination.exists(): shutil.copy2(source, destination)
        copied = dict(row); copied["state_file"] = str(destination.relative_to(HERE)); final_states.append(copied)
    for row in validated_states:
        copied = dict(row); copied["state_index"] = startup_index[row["state_id"]]; final_states.append(copied)
    state_prefix_exact = final_states[:len(v3_states)] == v3_states
    if not state_prefix_exact:
        raise AssertionError("V3 state-manifest prefix changed")
    assert_dataset_alignment(merged, final_metadata, final_states, "merged startup-complete")
    np.savez_compressed(HERE / "samples.npz", **merged)
    write_jsonl(HERE / "sample_metadata.jsonl", final_metadata)
    write_jsonl(HERE / "state_manifest.jsonl", final_states)

    # Schema must remain exactly the authoritative 214-D V3 schema.
    v3_schema = json.loads((V3 / "feature_schema.json").read_text())
    if builder.schema != v3_schema["segments"]:
        raise RuntimeError("startup builder schema differs from V3")
    shutil.copy2(V3 / "feature_schema.json", HERE / "feature_schema.json")
    write_json(HERE / "label_statistics.json", {
        "state_statistics": label_statistics,
        "counts": dict(Counter(value["classification"] for value in label_statistics.values())),
        "frozen_startup_candidate_count": len(startup_states),
        "usable_startup_state_count": len(validated_states),
        "B_63_empty": len(empty),
        "B_63_empty_state_count": len(empty),
        "B_63_empty_state_ids": empty,
        "quarantined_B_63_empty": empty,
        "quarantined_multivalued": quarantined,
    })

    split_groups = {split: {row["leakage_group"] for row in final_states if row["split"] == split} for split in ("train", "validation", "test")}
    overlaps = {f"{a}_{b}": sorted(split_groups[a] & split_groups[b]) for a, b in combinations(split_groups, 2)}
    prefix_equal = {key: bool(np.array_equal(merged[key][:len(v3_arrays[key])], v3_arrays[key])) for key in v3_arrays}
    leakage = {
        "passed": (
            not any(overlaps.values()) and all(prefix_equal.values())
            and state_prefix_exact and metadata_prefix_exact
            and len(merged["sample_id"]) == len(set(merged["sample_id"].tolist()))
        ),
        "assignment_unit": "source episode/leakage group",
        "split_group_overlaps": overlaps,
        "V3_sample_array_prefix_bitwise_equal": prefix_equal,
        "all_V3_sample_arrays_unchanged": all(prefix_equal.values()),
        "V3_state_manifest_prefix_exact": state_prefix_exact,
        "V3_sample_metadata_prefix_exact": metadata_prefix_exact,
        "sample_array_metadata_state_alignment": True,
        "duplicate_sample_ids": len(merged["sample_id"]) - len(set(merged["sample_id"].tolist())),
    }
    write_json(HERE / "leakage_checks.json", leakage)
    if not leakage["passed"]: raise RuntimeError("merged dataset leakage/integrity failure")
    split_manifest = {"assignment_unit": "source episode/leakage group", "state_counts": {split: sum(row["split"] == split for row in final_states) for split in split_groups}, "sample_counts": {split: int(np.sum(merged["split"] == split)) for split in split_groups}, "startup_state_counts": {split: sum(row["split"] == split for row in validated_states) for split in split_groups}, "startup_sample_counts": {split: sum(row["split"] == split for row in metadata_new) for split in split_groups}, "leakage_groups": {split: sorted(groups) for split, groups in split_groups.items()}}
    write_json(HERE / "split_manifest.json", split_manifest)
    projection = {"max_first_projection_replay_error": max_first_replay, "max_raw_formula_error": max_formula, "max_second_projection_replay_error": max_second_replay, "minimum_target_linear_residual": min_target_residual, "maximum_target_speed_excess": max_speed_excess, "solver_failures": 0, "passed": max_first_replay <= 1e-9 and max_formula <= 1e-9 and max_second_replay <= 1e-9 and min_target_residual >= -1e-7 and max_speed_excess <= 1e-7}
    write_json(HERE / "projection_replay_checks.json", projection)
    if not projection["passed"]: raise RuntimeError("projection replay gate failed")
    zero_states = sum(all(np.linalg.norm(np.asarray(arrays_new["targets"])[np.asarray(arrays_new["state_id"]) == row["state_id"]], axis=1) <= 1e-14) for row in validated_states)
    empty_display = ", ".join(f"`{state_id}`" for state_id in empty) if empty else "none"
    report = f"""# Startup-complete G_phi dataset

- Base V3 states/samples: {len(v3_states)} / {len(v3_arrays['features'])}
- Frozen startup candidates: {len(startup_states)} from {len(startup_states)} independent episodes
- Validated startup states/samples: {len(validated_states)} / {len(metadata_new)}
- Startup zero/nonzero states: {zero_states} / {len(validated_states)-zero_states}
- Quarantined multivalued startup states: {len(quarantined)}
- Final states/samples: {len(final_states)} / {len(merged['features'])}
- **B63-empty startup candidates excluded without labels: {len(empty)}**
- **B63-empty state IDs: {empty_display}**
- Every frozen candidate remains traceable in `startup_state_manifest.jsonl`, `startup_oracle_search_results.jsonl`, and `b63_empty_quarantine.json`.
- V3 sample arrays unchanged as prefix: yes
- Projection source: `{SYSROOT / 'single_integrator/cbf.py'}` (`{frozen_hashes[str(SYSROOT / 'single_integrator/cbf.py')]}`)
"""
    (HERE / "dataset_report.md").write_text(report)
    runtime = {"finalization_elapsed_s": time.monotonic() - started, "startup_states_frozen": len(startup_states), "startup_states_validated": len(validated_states), "startup_samples": len(metadata_new), "B_63_empty_states_excluded": len(empty), "B_63_empty_state_ids": empty, "cache_duplicate_rows": len(duplicates)}
    write_json(HERE / "runtime_statistics.json", runtime)
    files = [
        "dataset_report.md", "protocol.json", "frozen_source_audit.json",
        "startup_feature_rule.md", "startup_feature_continuity_checks.json",
        "restoration_checks.json", "startup_state_integrity_checks.json",
        "startup_selection_audit.json", "startup_states.csv",
        "startup_state_manifest.jsonl", "samples.npz", "sample_metadata.jsonl",
        "state_manifest.jsonl", "feature_schema.json", "split_manifest.json",
        "leakage_checks.json", "label_statistics.json",
        "b63_empty_quarantine.json",
        "projection_replay_checks.json", "startup_oracle_search_results.jsonl",
        "runtime_statistics.json",
    ]
    missing_manifest_files = [name for name in files if not (HERE / name).is_file()]
    if missing_manifest_files:
        raise RuntimeError(("required manifest artifacts missing", missing_manifest_files))
    write_json(HERE / "manifest.json", {
        "dataset": "gphi_training_dataset_startup_complete_v1",
        "parent_dataset": str(V3),
        "parent_dataset_manifest_sha256": protocol["base_v3_manifest_sha256"],
        "parent_samples_sha256": protocol["base_v3_samples_sha256"],
        "protocol_sha256": sha256(HERE / "protocol.json"),
        "frozen_hashes": protocol["frozen_hashes"],
        "unique_augmented_states": len(final_states),
        "supervised_samples": len(merged["features"]),
        "startup_candidates_frozen": len(startup_states),
        "startup_states": len(validated_states),
        "startup_samples": len(metadata_new),
        "feature_dimension": 214, "target_dimension": 4,
        "B_63_empty_states": len(empty),
        "B_63_empty_state_ids": empty,
        "B_63_empty_policy": "quarantined and excluded without supervised labels",
        "V3_samples_unchanged": True,
        "files_sha256": {name: sha256(HERE / name) for name in files},
    })
    print(json.dumps({"status": "PASS_WITH_QUARANTINE" if empty or quarantined else "PASS", "startup_candidates_frozen": len(startup_states), "startup_states": len(validated_states), "startup_samples": len(metadata_new), "final_states": len(final_states), "final_samples": len(merged["features"]), "zero_states": zero_states, "B_63_empty_states": len(empty), "B_63_empty_state_ids": empty, "quarantined_multivalued": len(quarantined)}, indent=2))


if __name__ == "__main__":
    main()
