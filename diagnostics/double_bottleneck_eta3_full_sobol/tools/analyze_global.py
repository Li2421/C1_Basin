#!/usr/bin/env python3
"""Analyze the common 256-point P0 map and freeze downstream selections."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol"
OLD = ROOT / "diagnostics/double_bottleneck_eta3_basin/SUMMARY.json"
CAPACITY = ROOT / "diagnostics/double_bottleneck_eta_representation_capacity/H0_STOP_AUDIT.json"
HARD_SAFETY = ROOT / "diagnostics/double_bottleneck_hard_safety_baseline"


def stats(values):
    values = np.asarray(list(values), dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "p10": float(np.percentile(values, 10)) if len(values) else None,
        "p90": float(np.percentile(values, 90)) if len(values) else None,
        "minimum": float(np.min(values)) if len(values) else None,
        "maximum": float(np.max(values)) if len(values) else None,
    }


def grouped_existence(rows, key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key))].append(row)
    return {
        name: {
            "episodes": len(group),
            "basin_exists": sum(row["basin_exists"] for row in group),
            "fraction": sum(row["basin_exists"] for row in group) / len(group),
            "rho_B_mean": float(np.mean([row["rho_B"] for row in group])),
        }
        for name, group in sorted(grouped.items())
    }


def load_rows():
    rows = []
    paths = [STUDY / "raw/global_cached.jsonl"] + sorted((STUDY / "raw").glob("global_pending_shard*.jsonl"))
    for path in paths:
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    unique = {}
    for row in rows:
        if row["job_id"] in unique:
            raise RuntimeError(f"duplicate {row['job_id']}")
        unique[row["job_id"]] = row
    return list(unique.values())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def weighted_rollout_stats(rows):
    steps = np.asarray([row["episode_steps"] for row in rows], dtype=np.float64)
    total = float(steps.sum())
    def weighted(path):
        values = []
        for row in rows:
            value = row["correction"]
            for key in path:
                value = value[key]
            values.append(float(value))
        return float(np.dot(values, steps) / total) if total else None
    raw = weighted(("raw_norm", "mean"))
    second = weighted(("second_projection_norm", "mean"))
    return {
        "rollouts": len(rows),
        "steps": int(total),
        "raw_norm_step_weighted_mean": raw,
        "executable_norm_step_weighted_mean": weighted(("executable_norm", "mean")),
        "second_projection_norm_step_weighted_mean": second,
        "removal_to_raw_ratio_of_means": second / raw if raw and raw > 0 else None,
        "retention_ratio_step_weighted_mean": weighted(("retention_ratio", "mean")),
        "raw_executable_cosine_step_weighted_mean": weighted(("raw_executable_cosine", "mean")),
        "mostly_rewritten_step_fraction": weighted(("mostly_rewritten_fraction",)),
        "second_projection_active_step_fraction": weighted(("second_projection_active_fraction",)),
    }


def main() -> int:
    rows = load_rows()
    if len(rows) != 85 * 256:
        raise RuntimeError(f"global result incomplete: {len(rows)} != 21760")
    catalog = json.loads((STUDY / "episode_catalog.json").read_text())["episodes"]
    points = json.loads((STUDY / "eta_points.json").read_text())["points"]
    episode_ids = [row["episode_id"] for row in catalog]
    point_ids = [row["parameter_id"] for row in points]
    theta = np.asarray([row["theta"] for row in points], dtype=np.float64)
    eindex = {value: index for index, value in enumerate(episode_ids)}
    pindex = {value: index for index, value in enumerate(point_ids)}
    matrix = np.zeros((85, 256), dtype=bool)
    lookup = {}
    for row in rows:
        key = (row["episode_id"], row["parameter_id"])
        if key in lookup:
            raise RuntimeError(f"duplicate pair {key}")
        lookup[key] = row
        matrix[eindex[key[0]], pindex[key[1]]] = bool(row["success"])
    targets = [row for row in catalog if row["population"] == "safe_timeout_target"]
    controls = [row for row in catalog if row["population"] == "baseline_success_control"]
    target_indices = np.asarray([eindex[row["episode_id"]] for row in targets], dtype=int)
    control_indices = np.asarray([eindex[row["episode_id"]] for row in controls], dtype=int)
    target_matrix = matrix[target_indices]
    control_matrix = matrix[control_indices]
    coverage = target_matrix.sum(axis=0)
    preservation = control_matrix.sum(axis=0)
    baseline_modes = {}
    for set_name in ("existing_untouched_test", "fresh_untouched_test"):
        artifact = json.loads((HARD_SAFETY / f"hard_safety_{set_name}_outcomes.json").read_text())
        for rollout in artifact["rollouts"]:
            mode = rollout.get("coordination_mode") or {}
            baseline_modes[f"{set_name}|{int(rollout['rollout_id']):03d}"] = {
                "baseline_first_direction": mode.get("first_direction"),
                "baseline_order_signature": mode.get("signature"),
            }
    eta_rows = []
    for index, point in enumerate(points):
        dominated = any(
            coverage[j] >= coverage[index] and preservation[j] >= preservation[index]
            and (coverage[j] > coverage[index] or preservation[j] > preservation[index])
            for j in range(256)
        )
        eta_rows.append({
            **point,
            "timeout_coverage": int(coverage[index]),
            "controls_preserved": int(preservation[index]),
            "controls_degraded": int(24 - preservation[index]),
            "pareto_nondominated": not dominated,
        })

    low = np.asarray((0.5, -0.5, 0.0))
    high = np.asarray((1.25, 0.5, 0.75))
    unit = (theta - low) / (high - low)
    episode_rows, representatives = [], []
    for episode, row_index in zip(targets, target_indices):
        success_indices = np.flatnonzero(matrix[row_index])
        count = len(success_indices)
        item = {
            **episode,
            **baseline_modes.get(episode["episode_id"], {}),
            "successful_eta_count": count,
            "rho_B": count / 256.0,
            "basin_exists": bool(count),
            "successful_parameter_ids": [point_ids[index] for index in success_indices],
        }
        if count:
            origin = min(success_indices, key=lambda index: (float(np.linalg.norm(theta[index])), point_ids[index]))
            support_counts = {}
            for index in success_indices:
                distance = np.linalg.norm(unit[success_indices] - unit[index], axis=1)
                support_counts[int(index)] = int(np.sum(distance <= 0.20))
            remaining = set(map(int, success_indices))
            coarse_components = []
            while remaining:
                start = min(remaining)
                remaining.remove(start)
                queue, component = deque((start,)), [start]
                while queue:
                    current = queue.popleft()
                    neighbors = [candidate for candidate in remaining if np.linalg.norm(unit[current] - unit[candidate]) <= 0.20]
                    for candidate in neighbors:
                        remaining.remove(candidate)
                        queue.append(candidate)
                        component.append(candidate)
                coarse_components.append(sorted(component))
            coarse_components.sort(key=lambda component: (-len(component), component[0]))
            item["coarse_geometry_normalized_radius"] = 0.20
            item["coarse_component_count"] = len(coarse_components)
            item["coarse_component_sizes"] = [len(component) for component in coarse_components]
            item["largest_coarse_component_fraction"] = len(coarse_components[0]) / count
            support = min(success_indices, key=lambda index: (-support_counts[int(index)], -int(coverage[index]), point_ids[index]))
            shared = min(success_indices, key=lambda index: (-int(coverage[index]), -int(preservation[index]), float(np.linalg.norm(theta[index])), point_ids[index]))
            roles = (("minimum_eta_norm", origin), ("maximum_coarse_support", support), ("maximum_global_coverage", shared))
            seen = set()
            selected = []
            for role, index in roles:
                if int(index) in seen:
                    continue
                seen.add(int(index))
                selected.append({
                    "role": role,
                    "parameter_id": point_ids[index],
                    "theta": theta[index].tolist(),
                    "timeout_coverage": int(coverage[index]),
                    "controls_preserved": int(preservation[index]),
                    "coarse_support_within_normalized_radius_0.20": support_counts[int(index)],
                })
            item["selected_representatives"] = selected
            representatives.append({**episode, "representatives": selected})
        else:
            item["selected_representatives"] = []
        episode_rows.append(item)

    positive = [row for row in episode_rows if row["basin_exists"]]
    none = [row for row in episode_rows if not row["basin_exists"]]
    ordered = sorted(positive, key=lambda row: (row["rho_B"], row["episode_id"]))
    geometry = {
        "high": max(positive, key=lambda row: (row["rho_B"], row["episode_id"]))["episode_id"] if positive else None,
        "median": ordered[(len(ordered) - 1) // 2]["episode_id"] if ordered else None,
        "low": ordered[0]["episode_id"] if ordered else None,
        "none": sorted(row["episode_id"] for row in none)[0] if none else None,
    }

    positive_indices = [index for index, row in enumerate(episode_rows) if row["basin_exists"]]
    overlap = np.zeros((len(positive_indices), len(positive_indices)), dtype=np.float64)
    pair_rows = []
    catalog_by_id = {row["episode_id"]: row for row in targets}
    for ai, i in enumerate(positive_indices):
        bi = target_matrix[i]
        for aj, j in enumerate(positive_indices):
            bj = target_matrix[j]
            union_count = int(np.sum(bi | bj))
            value = float(np.sum(bi & bj) / union_count) if union_count else 0.0
            overlap[ai, aj] = value
            if aj > ai:
                id_i, id_j = episode_rows[i]["episode_id"], episode_rows[j]["episode_id"]
                pair_rows.append({
                    "episode_i": id_i,
                    "episode_j": id_j,
                    "jaccard": value,
                    "same_baseline_direction": catalog_by_id[id_i].get("baseline_first_direction") == catalog_by_id[id_j].get("baseline_first_direction"),
                    "same_baseline_order_signature": baseline_modes.get(id_i, {}).get("baseline_order_signature") == baseline_modes.get(id_j, {}).get("baseline_order_signature"),
                })
    threshold = 0.25
    visited, components = set(), []
    for start in range(len(positive_indices)):
        if start in visited:
            continue
        queue, component = deque((start,)), []
        visited.add(start)
        while queue:
            current = queue.popleft()
            component.append(episode_rows[positive_indices[current]]["episode_id"])
            for neighbor in range(len(positive_indices)):
                if neighbor not in visited and neighbor != current and overlap[current, neighbor] >= threshold:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    offdiag = [row["jaccard"] for row in pair_rows]
    same = [row["jaccard"] for row in pair_rows if row["same_baseline_direction"]]
    cross = [row["jaccard"] for row in pair_rows if not row["same_baseline_direction"]]
    same_signature = [row["jaccard"] for row in pair_rows if row["same_baseline_order_signature"]]
    cross_signature = [row["jaccard"] for row in pair_rows if not row["same_baseline_order_signature"]]
    overlap_output = {
        "schema": "double_bottleneck_eta3_overlap_v1",
        "positive_episode_ids": [episode_rows[index]["episode_id"] for index in positive_indices],
        "jaccard_matrix": overlap.tolist(),
        "pairwise": pair_rows,
        "offdiagonal_statistics": stats(offdiag),
        "same_direction_statistics": stats(same),
        "cross_direction_statistics": stats(cross),
        "same_order_signature_statistics": stats(same_signature),
        "cross_order_signature_statistics": stats(cross_signature),
        "cluster_edge_threshold": threshold,
        "connected_components": components,
        "component_sizes": sorted((len(component) for component in components), reverse=True),
    }

    success_rows = [row for row in rows if row["success"]]
    failed_rows = [row for row in rows if not row["success"]]
    projection = {
        "schema": "double_bottleneck_eta3_projection_coupling_v1",
        "successful": weighted_rollout_stats(success_rows),
        "failed": weighted_rollout_stats(failed_rows),
        "successful_timeout_targets": weighted_rollout_stats([row for row in rows if row["population"] == "safe_timeout_target" and row["success"]]),
        "failed_timeout_targets": weighted_rollout_stats([row for row in rows if row["population"] == "safe_timeout_target" and not row["success"]]),
        "preserved_success_controls": weighted_rollout_stats([row for row in rows if row["population"] == "baseline_success_control" and row["success"]]),
        "degraded_success_controls": weighted_rollout_stats([row for row in rows if row["population"] == "baseline_success_control" and not row["success"]]),
        "eta_zero": {"raw_eta_correction": 0.0, "executable_eta_correction": 0.0, "second_eta_projection_removal": 0.0, "note": "eta=0 is the frozen hard-safety baseline; there is no eta correction to project"},
    }
    summary = {
        "schema": "double_bottleneck_eta3_full_sobol_global_summary_v1",
        "rollouts": len(rows),
        "collisions": sum(row["wall_collision"] or row["agent_collision"] for row in rows),
        "outcomes": dict(sorted(Counter(row["outcome"] for row in rows).items())),
        "episode_steps_successful": stats(row["episode_steps"] for row in rows if row["success"]),
        "episode_steps_failed": stats(row["episode_steps"] for row in rows if not row["success"]),
        "minimum_wall_clearance_all": float(min(row["minimum_wall_clearance"] for row in rows)),
        "minimum_agent_clearance_all": float(min(row["minimum_agent_clearance"] for row in rows)),
        "rollout_wall_seconds": stats(row["wall_seconds"] for row in rows),
        "total_rollout_cpu_hours": float(sum(row["wall_seconds"] for row in rows) / 3600.0),
        "timeout_basin_existence": sum(row["basin_exists"] for row in episode_rows),
        "timeouts": 61,
        "rho_B_all": stats(row["rho_B"] for row in episode_rows),
        "rho_B_positive": stats(row["rho_B"] for row in episode_rows if row["basin_exists"]),
        "successful_eta_count_positive": stats(row["successful_eta_count"] for row in episode_rows if row["basin_exists"]),
        "coarse_component_count_positive": stats(row["coarse_component_count"] for row in episode_rows if row["basin_exists"]),
        "largest_coarse_component_fraction_positive": stats(row["largest_coarse_component_fraction"] for row in episode_rows if row["basin_exists"]),
        "existence_by_set": grouped_existence(episode_rows, "set"),
        "existence_by_regime": grouped_existence(episode_rows, "regime"),
        "existence_by_baseline_direction": grouped_existence(episode_rows, "baseline_first_direction"),
        "existence_by_baseline_order_signature": grouped_existence(episode_rows, "baseline_order_signature"),
        "control_preservation_fraction": stats(float(control_matrix[index].mean()) for index in range(24)),
        "best_timeout_coverage": int(coverage.max()),
        "best_control_preservation": int(preservation.max()),
        "pareto_parameter_ids": [row["parameter_id"] for row in eta_rows if row["pareto_nondominated"]],
        "geometry_representatives": geometry,
        "comparison": {
            "old_65_point_existence": 15,
            "A004_coverage": int(coverage[pindex["EMBED"]]),
            "six_new_point_union": json.loads((CAPACITY).read_text())["new_dense_P0_points"]["target_success"],
            "seven_point_union": json.loads((CAPACITY).read_text())["combined_observed_P0"]["target_success"],
            "full_256_existence": sum(row["basin_exists"] for row in episode_rows),
        },
    }

    np.savez_compressed(STUDY / "basin_matrices.npz", timeout_membership=target_matrix, control_preservation=control_matrix, eta=theta, timeout_episode_ids=np.asarray([row["episode_id"] for row in targets]), control_episode_ids=np.asarray([row["episode_id"] for row in controls]), parameter_ids=np.asarray(point_ids))
    (STUDY / "timeout_basin_membership.json").write_text(json.dumps({"schema": "double_bottleneck_eta3_timeout_membership_v1", "parameter_ids": point_ids, "episodes": episode_rows}, indent=2, sort_keys=True) + "\n")
    (STUDY / "control_preservation_matrix.json").write_text(json.dumps({"schema": "double_bottleneck_eta3_control_preservation_v1", "parameter_ids": point_ids, "episodes": [{**episode, "preserved_parameter_ids": [point_ids[index] for index in np.flatnonzero(control_matrix[row_index])], "preservation_fraction": float(control_matrix[row_index].mean())} for row_index, episode in enumerate(controls)]}, indent=2, sort_keys=True) + "\n")
    (STUDY / "per_eta_coverage.json").write_text(json.dumps({"schema": "double_bottleneck_eta3_per_eta_coverage_v1", "eta": eta_rows}, indent=2, sort_keys=True) + "\n")
    (STUDY / "selected_success_representatives.json").write_text(json.dumps({"schema": "double_bottleneck_eta3_selected_representatives_v1", "episodes": representatives}, indent=2, sort_keys=True) + "\n")
    (STUDY / "geometry_representatives.json").write_text(json.dumps(geometry, indent=2, sort_keys=True) + "\n")
    (STUDY / "cross_state_overlap.json").write_text(json.dumps(overlap_output, indent=2, sort_keys=True) + "\n")
    (STUDY / "projection_coupling_statistics.json").write_text(json.dumps(projection, indent=2, sort_keys=True) + "\n")
    (STUDY / "GLOBAL_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    raw_paths = [STUDY / "raw/global_cached.jsonl"] + sorted((STUDY / "raw").glob("global_pending_shard*.jsonl"))
    (STUDY / "global_outcomes_manifest.json").write_text(json.dumps({
        "schema": "double_bottleneck_eta3_global_outcomes_manifest_v1",
        "total_rows": len(rows),
        "files": [{
            "path": str(path.relative_to(ROOT)),
            "rows": sum(1 for line in path.read_text().splitlines() if line.strip()),
            "sha256": sha(path),
        } for path in raw_paths],
    }, indent=2, sort_keys=True) + "\n")
    with (STUDY / "per_episode_basin.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("episode_id", "set", "regime", "baseline_first_direction", "successful_eta_count", "rho_B", "basin_exists"))
        writer.writeheader()
        for row in episode_rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    with (STUDY / "per_eta_coverage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("parameter_id", "eta1", "eta2", "eta3", "timeout_coverage", "controls_preserved", "controls_degraded", "pareto_nondominated"))
        writer.writeheader()
        for row in eta_rows:
            writer.writerow({"parameter_id": row["parameter_id"], "eta1": row["theta"][0], "eta2": row["theta"][1], "eta3": row["theta"][2], **{key: row[key] for key in ("timeout_coverage", "controls_preserved", "controls_degraded", "pareto_nondominated")}})
    print(json.dumps({"existence": summary["timeout_basin_existence"], "collisions": summary["collisions"], "best_coverage": summary["best_timeout_coverage"], "positive_rho_median": summary["rho_B_positive"]["median"], "representative_pairs": sum(len(row["representatives"]) for row in representatives)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
