#!/usr/bin/env python3
"""Validate and summarize the frozen 3D-eta basin experiment."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta3_basin"
LOW = np.asarray((0.5, -0.5, 0.0), dtype=np.float64)
HIGH = np.asarray((1.25, 0.5, 0.75), dtype=np.float64)
RADIUS = 0.35


def dump(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_rows() -> list[dict]:
    rows = []
    for path in sorted((STUDY / "raw").glob("stage_*_shard*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    ids = [row["job_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate result job IDs")
    expected = 0
    for path in sorted((STUDY / "jobs").glob("stage_*.json")):
        expected += len(json.loads(path.read_text())["jobs"])
    if len(rows) != expected:
        raise RuntimeError(f"results incomplete: {len(rows)} != {expected}")
    return rows


def episode_id(row: dict) -> str:
    return f"{row['set']}|{row['rollout_id']:03d}"


def normalized(eta) -> np.ndarray:
    return (np.asarray(eta, dtype=np.float64) - LOW) / (HIGH - LOW)


def components(rows: list[dict]) -> list[list[str]]:
    if not rows:
        return []
    points = np.stack([normalized(row["eta"]) for row in rows])
    adjacency = np.linalg.norm(points[:, None] - points[None, :], axis=-1) <= RADIUS
    unseen = set(range(len(rows)))
    result = []
    while unseen:
        seed = min(unseen)
        stack = [seed]
        unseen.remove(seed)
        component = []
        while stack:
            current = stack.pop()
            component.append(rows[current]["eta_id"])
            neighbors = [index for index in list(unseen) if adjacency[current, index]]
            for index in neighbors:
                unseen.remove(index)
                stack.append(index)
        result.append(sorted(component))
    return sorted(result, key=lambda values: (-len(values), values))


def numeric_summary(values) -> dict:
    values = np.asarray(list(values), dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p90": None, "minimum": None, "maximum": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def main() -> int:
    catalog = json.loads((STUDY / "episode_catalog.json").read_text())
    samples = json.loads((STUDY / "eta_samples.json").read_text())
    refinement_centers = {}
    for stage in ("stage_b", "stage_c"):
        path = STUDY / "jobs" / f"{stage}.json"
        if path.exists():
            for decision in json.loads(path.read_text()).get("decisions", []):
                if decision.get("center") is not None:
                    refinement_centers[decision["episode_id"]] = decision["center"]
    rows = load_rows()
    by_episode = defaultdict(list)
    for row in rows:
        by_episode[episode_id(row)].append(row)
    expected_episodes = {row["episode_id"]: row for row in catalog["targets"] + catalog["controls"]}
    if set(by_episode) != set(expected_episodes):
        raise RuntimeError("episode set mismatch")

    zero_errors = []
    for eid, episode in expected_episodes.items():
        zero = [row for row in by_episode[eid] if row["eta_id"] == "ZERO"]
        if len(zero) != 1 or zero[0]["outcome"] != episode["baseline_outcome"] or zero[0]["episode_steps"] != episode["baseline_steps"]:
            zero_errors.append({"episode_id": eid, "expected": episode, "observed": zero})
    collision_rows = [row["job_id"] for row in rows if row["wall_collision"] or row["agent_collision"]]
    deadlock_rows = [row["job_id"] for row in rows if row["runtime_strict_deadlock"]]
    if zero_errors:
        raise RuntimeError(f"eta=0 failed frozen reproduction in {len(zero_errors)} episodes")
    if collision_rows:
        raise RuntimeError(f"second projection allowed {len(collision_rows)} collision rollouts")

    target_ids = {row["episode_id"] for row in catalog["targets"]}
    control_ids = {row["episode_id"] for row in catalog["controls"]}
    per_episode = []
    membership = []
    for episode in catalog["targets"]:
        eid = episode["episode_id"]
        erows = by_episode[eid]
        stage_a = sorted([row for row in erows if row["eta_id"].startswith("A")], key=lambda row: row["eta_id"])
        stage_a_success = [row for row in stage_a if row["success"]]
        dense = sorted([row for row in erows if row["sample_type"] == "stage_b_dense_sobol"], key=lambda row: row["eta_id"])
        local = sorted([row for row in erows if row["sample_type"] == "local_registered_neighbor"], key=lambda row: row["job_id"])
        all_domain = [row for row in erows if row["eta_id"] != "ZERO"]
        all_success = [row for row in all_domain if row["success"]]
        comps = components(stage_a_success)
        fragmented = sum(len(component) >= 2 for component in comps) >= 2
        rho = len(stage_a_success) / 33.0
        local_fraction = len([row for row in local if row["success"]]) / len(local) if local else None
        largest_component_fraction = max((len(component) for component in comps), default=0) / max(len(stage_a_success), 1)
        center_id = refinement_centers.get(eid)
        if not all_success:
            basin_type = "Type IV — no observed basin"
            representative = None
        elif fragmented:
            basin_type = "Type III — fragmented / multimodal basin"
            representative = None
        elif rho >= 0.20 and local_fraction is not None and local_fraction >= 0.50 and largest_component_fraction >= 0.75:
            basin_type = "Type I — broad basin"
            representative = None
        else:
            basin_type = "Type II — narrow basin"
            representative = None
        if all_success and representative is None:
            representative = next((row for row in all_success if row["eta_id"] == center_id), None)
            if representative is None:
                representative = min(all_success, key=lambda row: (float(np.linalg.norm(normalized(row["eta"]) - normalized((1.0, 0.0, 0.25)))), row["job_id"]))
        item = {
            **episode,
            "stage_a_successes": len(stage_a_success),
            "stage_a_points": len(stage_a),
            "stage_a_basin_fraction": rho,
            "stage_a_components": comps,
            "largest_component_fraction": largest_component_fraction,
            "dense_global_points": len(dense),
            "dense_global_successes": sum(row["success"] for row in dense),
            "local_points": len(local),
            "local_successes": sum(row["success"] for row in local),
            "local_success_fraction": local_fraction,
            "total_domain_points_tested": len(all_domain),
            "total_successes_observed": len(all_success),
            "basin_exists": bool(all_success),
            "basin_type": basin_type,
            "representative": None if representative is None else {
                "job_id": representative["job_id"],
                "eta_id": representative["eta_id"],
                "eta": representative["eta"],
                "stage": representative["stage"],
                "completion_time_seconds": representative["completion_time_seconds"],
                "g_mean": representative["correction"]["g_raw_joint_norm"]["mean"],
                "second_projection_mean": representative["correction"]["second_projection_correction_norm"]["mean"],
                "substantially_altered_fraction": representative["correction"]["substantially_altered_fraction"],
                "mostly_rewritten_fraction": representative["correction"]["mostly_rewritten_fraction"],
            },
        }
        per_episode.append(item)
        for row in stage_a:
            membership.append({"episode_id": eid, "eta_id": row["eta_id"], "success": row["success"]})

    eta_lookup = {row["eta_id"]: row["eta"] for row in samples["stage_a_global"]}
    cross_state = []
    for eta_id in sorted(eta_lookup):
        target_rows = [row for eid in target_ids for row in by_episode[eid] if row["eta_id"] == eta_id]
        control_rows = [row for eid in control_ids for row in by_episode[eid] if row["eta_id"] == eta_id]
        cross_state.append({
            "eta_id": eta_id,
            "eta": eta_lookup[eta_id],
            "timeout_targets_rescued": sum(row["success"] for row in target_rows),
            "target_count": len(target_rows),
            "baseline_success_controls_preserved": sum(row["success"] for row in control_rows),
            "control_count": len(control_rows),
            "control_outcomes": dict(sorted(Counter(row["outcome"] for row in control_rows).items())),
            "target_mean_g_norm": float(np.mean([row["correction"]["g_raw_joint_norm"]["mean"] for row in target_rows])),
            "target_mean_second_projection": float(np.mean([row["correction"]["second_projection_correction_norm"]["mean"] for row in target_rows])),
        })

    vectors = {
        item["episode_id"]: np.asarray([
            next(row["success"] for row in by_episode[item["episode_id"]] if row["eta_id"] == eta_id)
            for eta_id in sorted(eta_lookup)
        ], dtype=bool)
        for item in catalog["targets"]
    }
    jaccards = []
    ids = sorted(vectors)
    for left_index, left in enumerate(ids):
        for right in ids[left_index + 1:]:
            union = np.logical_or(vectors[left], vectors[right]).sum()
            if union:
                jaccards.append(float(np.logical_and(vectors[left], vectors[right]).sum() / union))

    stage_a_domain = [row for row in rows if row["eta_id"].startswith("A")]
    correction_groups = {}
    for label, selected in {
        "target_success": [row for row in stage_a_domain if episode_id(row) in target_ids and row["success"]],
        "target_failure": [row for row in stage_a_domain if episode_id(row) in target_ids and not row["success"]],
        "control_preserved": [row for row in stage_a_domain if episode_id(row) in control_ids and row["success"]],
        "control_degraded": [row for row in stage_a_domain if episode_id(row) in control_ids and not row["success"]],
    }.items():
        correction_groups[label] = {
            "rollouts": len(selected),
            "g_mean": numeric_summary(row["correction"]["g_raw_joint_norm"]["mean"] for row in selected),
            "second_projection_mean": numeric_summary(row["correction"]["second_projection_correction_norm"]["mean"] for row in selected),
            "projection_active_fraction": numeric_summary(row["correction"]["second_projection_active_fraction"] for row in selected),
            "substantially_altered_fraction": numeric_summary(row["correction"]["substantially_altered_fraction"] for row in selected),
            "mostly_rewritten_fraction": numeric_summary(row["correction"]["mostly_rewritten_fraction"] for row in selected),
        }

    grouped = {}
    for field in ("set", "regime", "baseline_first_direction"):
        values = defaultdict(list)
        for item in per_episode:
            values[item[field]].append(item)
        grouped[field] = {
            value: {
                "episodes": len(items),
                "basin_exists": sum(item["basin_exists"] for item in items),
                "mean_stage_a_basin_fraction": float(np.mean([item["stage_a_basin_fraction"] for item in items])),
                "types": dict(sorted(Counter(item["basin_type"] for item in items).items())),
            }
            for value, items in sorted(values.items())
        }

    best_eta = max(cross_state, key=lambda row: (row["timeout_targets_rescued"], row["baseline_success_controls_preserved"], row["eta_id"]))
    retry_counts = Counter()
    for row in rows:
        for side in ("first_projection_status_counts", "second_projection_status_counts"):
            for status, count in row["correction"][side].items():
                if "retry" in status:
                    retry_counts[status] += count
    summary = {
        "schema": "double_bottleneck_eta3_basin_summary_v1",
        "validation": {
            "result_rows": len(rows),
            "eta_zero_exact_reproductions": len(expected_episodes),
            "eta_zero_errors": len(zero_errors),
            "collision_rollouts": len(collision_rows),
            "runtime_strict_deadlock_rollouts": len(deadlock_rows),
            "numerical_retry_status_counts": dict(sorted(retry_counts.items())),
        },
        "targets": {
            "episodes": len(per_episode),
            "basin_exists": sum(item["basin_exists"] for item in per_episode),
            "basin_existence_fraction": float(np.mean([item["basin_exists"] for item in per_episode])),
            "basin_type_counts": dict(sorted(Counter(item["basin_type"] for item in per_episode).items())),
            "stage_a_basin_fraction": numeric_summary(item["stage_a_basin_fraction"] for item in per_episode),
            "local_success_fraction": numeric_summary(item["local_success_fraction"] for item in per_episode if item["local_success_fraction"] is not None),
        },
        "cross_state": {
            "best_shared_eta": best_eta,
            "target_coverage_counts": numeric_summary(row["timeout_targets_rescued"] for row in cross_state),
            "control_preservation_counts": numeric_summary(row["baseline_success_controls_preserved"] for row in cross_state),
            "pairwise_nonempty_stage_a_jaccard": numeric_summary(jaccards),
        },
        "grouped": grouped,
        "correction_groups": correction_groups,
    }

    dump(STUDY / "global_eta_domain.json", {"domain": samples["domain"], "zero_sentinel": samples["zero_sentinel"]})
    dump(STUDY / "all_rollout_outcomes.json", {"schema": "double_bottleneck_eta3_all_rollouts_v1", "rollouts": rows})
    dump(STUDY / "per_episode_basin.json", {"schema": "double_bottleneck_eta3_per_episode_v1", "episodes": per_episode})
    dump(STUDY / "baseline_success_controls.json", {
        "schema": "double_bottleneck_eta3_controls_v1",
        "episodes": catalog["controls"],
        "stage_a_outcomes": [row for row in rows if episode_id(row) in control_ids and row["eta_id"].startswith("A")],
    })
    dump(STUDY / "per_eta_cross_state_coverage.json", {"schema": "double_bottleneck_eta3_cross_state_v1", "etas": cross_state})
    dump(STUDY / "correction_projection_statistics.json", {"schema": "double_bottleneck_eta3_correction_v1", "groups": correction_groups})
    dump(STUDY / "SUMMARY.json", summary)
    with (STUDY / "basin_membership.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("episode_id", "eta_id", "success"))
        writer.writeheader()
        writer.writerows(membership)
    with (STUDY / "per_episode_basin.csv").open("w", newline="") as handle:
        fields = ("episode_id", "set", "regime", "baseline_first_direction", "basin_exists", "basin_type", "stage_a_successes", "stage_a_basin_fraction", "local_points", "local_success_fraction", "total_domain_points_tested", "total_successes_observed")
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_episode)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
