#!/usr/bin/env python3
"""Aggregate pilot rollout capacity and apply pre-registered promotion rules."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta_representation_capacity"
NAMES = ("P0-3D", "P1-Agent6", "P2-Pair8", "P3-Temporal6")


def stats(values):
    values = np.asarray(list(values), dtype=float)
    return {"count": int(len(values)), "mean": float(np.mean(values)) if len(values) else None, "median": float(np.median(values)) if len(values) else None, "minimum": float(np.min(values)) if len(values) else None, "maximum": float(np.max(values)) if len(values) else None}


def load(name):
    slug = name.replace("-", "_")
    rows = []
    for path in sorted((STUDY / "raw").glob(f"{slug}_*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    unique = {}
    for row in rows:
        if row["job_id"] in unique:
            raise RuntimeError(f"duplicate {row['job_id']}")
        unique[row["job_id"]] = row
    return list(unique.values())


def main() -> int:
    catalog = json.loads((STUDY / "pilot_catalog.json").read_text())["episodes"]
    design = json.loads((STUDY / "parameter_designs.json").read_text())
    offline = json.loads((STUDY / "offline_expert_fit_and_projection_audit.json").read_text())
    p0_dense = json.loads((STUDY / "p0_dense_search_audit.json").read_text())
    p0_dense_success = {row["episode_id"] for row in p0_dense["episodes"] if row["success_exists"]}
    target_ids = {row["episode_id"] for row in catalog if row["population"] == "safe_timeout_target"}
    control_ids = {row["episode_id"] for row in catalog if row["population"] == "baseline_success_control"}
    stratum = {row["episode_id"]: row["pilot_stratum"] for row in catalog}
    result = {}
    for name in NAMES:
        rows = load(name)
        if not rows:
            raise RuntimeError(f"missing results for {name}")
        expected_stage_a = len(catalog) * (257 if name == "P0-3D" else len(design[name]["stage_a"]))
        stage_a = [row for row in rows if row["stage"] == "pilot_stage_a"]
        if len(stage_a) != expected_stage_a:
            raise RuntimeError(f"{name} Stage A incomplete {len(stage_a)} != {expected_stage_a}")
        domain = [row for row in rows if row["parameter_id"] != "ZERO"]
        by_episode = defaultdict(list)
        for row in domain:
            by_episode[row["episode_id"]].append(row)
        episode_rows = []
        for episode in catalog:
            erows = by_episode[episode["episode_id"]]
            successes = [row for row in erows if row["success"]]
            structural_successes = [row for row in successes if row["sample_type"] != "exact_p0_subspace_inheritance"]
            local = [row for row in erows if row["sample_type"] == "registered_local_neighbor"]
            global_a = [row for row in erows if row["stage"] == "pilot_stage_a"]
            episode_rows.append({
                "episode_id": episode["episode_id"], "stratum": episode["pilot_stratum"], "population": episode["population"],
                "basin_exists": bool(successes), "successes": len(successes), "points": len(erows),
                "structural_basin_exists": bool(structural_successes),
                "stage_a_successes": sum(row["success"] for row in global_a),
                "stage_a_fraction": sum(row["success"] for row in global_a) / len(global_a),
                "local_points": len(local), "local_successes": sum(row["success"] for row in local),
                "local_success_fraction": sum(row["success"] for row in local) / len(local) if local else None,
            })
        target_episode_rows = [row for row in episode_rows if row["population"] == "safe_timeout_target"]
        local_fractions = [row["local_success_fraction"] for row in target_episode_rows if row["local_success_fraction"] is not None]
        successful_rollouts = [row for row in domain if row["episode_id"] in target_ids and row["success"]]
        shared = []
        stage_a_ids = sorted({row["parameter_id"] for row in stage_a if row["parameter_id"] != "ZERO"})
        for parameter_id in stage_a_ids:
            selected = [row for row in stage_a if row["parameter_id"] == parameter_id]
            shared.append({
                "parameter_id": parameter_id,
                "theta": selected[0]["theta"],
                "targets_rescued": sum(row["success"] and row["episode_id"] in target_ids for row in selected),
                "controls_preserved": sum(row["success"] and row["episode_id"] in control_ids for row in selected),
            })
        best_shared = max(shared, key=lambda row: (row["targets_rescued"], row["controls_preserved"], row["parameter_id"]))
        fit_improvement = offline["representations"][name]["relative_improvement_from_P0_best"]["median"]
        result[name] = {
            "rollouts": len(rows), "domain_rollouts": len(domain),
            "collision_rollouts": sum(row["wall_collision"] or row["agent_collision"] for row in rows),
            "target_basin_exists": sum(row["basin_exists"] for row in target_episode_rows),
            "target_structural_basin_exists": sum(row["structural_basin_exists"] for row in target_episode_rows),
            "targets": len(target_episode_rows),
            "existence_by_stratum": {
                value: {"episodes": sum(row["stratum"] == value for row in episode_rows), "basin_exists": sum(row["stratum"] == value and row["basin_exists"] for row in episode_rows)}
                for value in ("p0_positive", "p0_negative", "baseline_success_control")
            },
            "control_episodes_with_any_nonzero_success": sum(row["basin_exists"] for row in episode_rows if row["population"] == "baseline_success_control"),
            "local_success_fraction": stats(local_fractions),
            "global_stage_a_fraction_targets": stats(row["stage_a_fraction"] for row in target_episode_rows),
            "best_shared_parameter": best_shared,
            "successful_target_rollouts": len(successful_rollouts),
            "successful_correction": {
                "raw_norm": stats(row["correction"]["raw_norm"]["mean"] for row in successful_rollouts),
                "executable_norm": stats(row["correction"]["executable_norm"]["mean"] for row in successful_rollouts),
                "retention_ratio": stats(row["correction"]["retention_ratio"]["mean"] for row in successful_rollouts),
                "mostly_rewritten_fraction": stats(row["correction"]["mostly_rewritten_fraction"] for row in successful_rollouts),
            },
            "outcomes": dict(sorted(Counter(row["outcome"] for row in domain).items())),
            "offline_median_relative_fit_improvement_over_P0": fit_improvement,
            "episodes": episode_rows,
            "shared_parameter_coverage": shared,
        }
    p0_count = result["P0-3D"]["target_basin_exists"]
    promotions = {}
    for name in NAMES[1:]:
        value = result[name]
        gain = value["target_basin_exists"] - p0_count
        local_median = value["local_success_fraction"]["median"] or 0.0
        new_negative = value["existence_by_stratum"]["p0_negative"]["basin_exists"]
        structural_new_negative = sum(
            row["stratum"] == "p0_negative" and row["structural_basin_exists"] and row["episode_id"] not in p0_dense_success
            for row in value["episodes"]
        )
        fit = value["offline_median_relative_fit_improvement_over_P0"]
        criterion_a = gain >= 3 and local_median >= 0.25
        criterion_b = fit >= 0.25 and structural_new_negative >= 2
        promotions[name] = {"gain_over_dense_P0": gain, "novel_structural_p0_negative": structural_new_negative, "criterion_a": criterion_a, "criterion_b": criterion_b, "promote": bool(criterion_a or criterion_b)}
    p1 = result["P1-Agent6"]
    p4_trigger = (
        p1["target_basin_exists"] - p0_count >= 4
        and p1["target_basin_exists"] - result["P2-Pair8"]["target_basin_exists"] >= 2
        and p1["target_basin_exists"] - result["P3-Temporal6"]["target_basin_exists"] >= 2
        and p1["target_basin_exists"] < 10
        and offline["representations"]["P1-Agent6"]["best_residual"]["median"] >= 0.03
    )
    output = {"schema": "double_bottleneck_eta_capacity_pilot_summary_v1", "representations": result, "promotions": promotions, "p4_trigger": p4_trigger}
    (STUDY / "pilot_capacity_summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"existence": {name: result[name]["target_basin_exists"] for name in NAMES}, "promotions": promotions, "p4_trigger": p4_trigger}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
