#!/usr/bin/env python3
"""Aggregate registered full, robustness, and cross-state capacity results."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta_representation_capacity"
PRIOR = ROOT / "diagnostics/double_bottleneck_eta3_basin/episode_catalog.json"


def stats(values):
    values = np.asarray(list(values), dtype=float)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "p90": float(np.percentile(values, 90)) if len(values) else None,
        "minimum": float(np.min(values)) if len(values) else None,
        "maximum": float(np.max(values)) if len(values) else None,
    }


def theta_key(theta) -> str:
    canonical = json.dumps([round(float(value), 14) for value in theta], separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def load(rep):
    slug = rep.replace("-", "_")
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
    pilot = json.loads((STUDY / "pilot_capacity_summary.json").read_text())
    designs = json.loads((STUDY / "parameter_designs.json").read_text())
    catalog = json.loads(PRIOR.read_text())
    targets = {row["episode_id"] for row in catalog["targets"]}
    controls = {row["episode_id"] for row in catalog["controls"]}
    output = {
        "schema": "double_bottleneck_eta_representation_capacity_final_v1",
        "p0_dense": json.loads((STUDY / "p0_dense_search_audit.json").read_text()),
        "offline": json.loads((STUDY / "offline_expert_fit_and_projection_audit.json").read_text()),
        "pilot": pilot,
        "full": {},
    }
    promoted = [name for name, rule in pilot["promotions"].items() if rule["promote"]]
    for rep in promoted:
        rows = load(rep)
        stage_a = [row for row in rows if row["stage"] == "full_stage_a"]
        inheritance = [row for row in rows if row["stage"] == "full_p0_inheritance"]
        refinement = [row for row in rows if row["stage"] == "full_refinement"]
        post_local = [row for row in rows if row["stage"] == "full_post_confirmation_local"]
        registered = stage_a + inheritance + refinement + post_local
        expected_stage_a = 85 * len(designs[rep]["stage_a"])
        if len(stage_a) != expected_stage_a:
            raise RuntimeError(f"{rep} full Stage A incomplete {len(stage_a)} != {expected_stage_a}")
        inheritance_manifest = json.loads((STUDY / "jobs" / f"{rep.replace('-', '_')}_full_p0_inheritance.json").read_text())
        if len(inheritance) != len(inheritance_manifest["jobs"]):
            raise RuntimeError(f"{rep} full P0 inheritance incomplete")
        refinement_manifest = json.loads((STUDY / "jobs" / f"{rep.replace('-', '_')}_full_refinement.json").read_text())
        if len(refinement) != len(refinement_manifest["jobs"]):
            raise RuntimeError(f"{rep} full refinement incomplete")
        post_manifest = json.loads((STUDY / "jobs" / f"{rep.replace('-', '_')}_full_post_confirmation_local.json").read_text())
        if len(post_local) != len(post_manifest["jobs"]):
            raise RuntimeError(f"{rep} post-confirmation local check incomplete")
        by_episode = defaultdict(list)
        for row in registered:
            by_episode[row["episode_id"]].append(row)
        episode_summary = []
        for episode_id in sorted(targets | controls):
            erows = by_episode[episode_id]
            successes = [row for row in erows if row["success"]]
            p0_successes = [row for row in successes if row["sample_type"] == "exact_p0_subspace_inheritance" or row["parameter_id"] == "EMBED"]
            structural_successes = [row for row in successes if row not in p0_successes]
            global_rows = [row for row in erows if row["stage"] == "full_stage_a"]
            local_rows = [row for row in erows if row["sample_type"] == "registered_local_neighbor"]
            episode_summary.append({
                "episode_id": episode_id,
                "population": "safe_timeout_target" if episode_id in targets else "baseline_success_control",
                "basin_exists": bool(successes),
                "p0_subspace_basin_exists": bool(p0_successes),
                "structural_basin_exists": bool(structural_successes),
                "novel_structural_basin": bool(structural_successes) and not bool(p0_successes),
                "registered_successes": len(successes),
                "registered_points": len(erows),
                "global_success_fraction": sum(row["success"] for row in global_rows) / len(global_rows),
                "local_success_fraction": sum(row["success"] for row in local_rows) / len(local_rows) if local_rows else None,
            })
        target_summary = [row for row in episode_summary if row["population"] == "safe_timeout_target"]
        control_summary = [row for row in episode_summary if row["population"] == "baseline_success_control"]
        successful_target = [row for row in registered if row["episode_id"] in targets and row["success"]]
        shared = []
        for parameter_id in sorted({row["parameter_id"] for row in stage_a}):
            selected = [row for row in stage_a if row["parameter_id"] == parameter_id]
            shared.append({
                "parameter_id": parameter_id,
                "theta": selected[0]["theta"],
                "targets_rescued": sum(row["success"] and row["episode_id"] in targets for row in selected),
                "controls_preserved": sum(row["success"] and row["episode_id"] in controls for row in selected),
            })
        cross_rows = [row for row in rows if row["stage"] == "cross_state_reuse"]
        all_reuse_rows = registered + cross_rows
        reuse = defaultdict(dict)
        theta_values = {}
        for row in all_reuse_rows:
            key = theta_key(row["theta"])
            theta_values[key] = row["theta"]
            reuse[key][row["episode_id"]] = row
        successful_keys = {theta_key(row["theta"]) for row in registered if row["success"]}
        reuse_summary = []
        for key in sorted(successful_keys):
            values = reuse[key]
            reuse_summary.append({
                "parameter_key": key,
                "theta": theta_values[key],
                "episodes_evaluated": len(values),
                "targets_rescued": sum(row["success"] and episode_id in targets for episode_id, row in values.items()),
                "controls_preserved": sum(row["success"] and episode_id in controls for episode_id, row in values.items()),
            })
        robust = [row for row in rows if row["stage"] == "stochastic_robustness"]
        robust_by_episode = defaultdict(list)
        for row in robust:
            robust_by_episode[row["episode_id"]].append(row)
        output["full"][rep] = {
            "rollouts": len(rows),
            "registered_rollouts": len(registered),
            "collision_rollouts": sum(row["wall_collision"] or row["agent_collision"] for row in rows),
            "target_basin_exists": sum(row["basin_exists"] for row in target_summary),
            "target_p0_subspace_basin_exists": sum(row["p0_subspace_basin_exists"] for row in target_summary),
            "target_structural_basin_exists": sum(row["structural_basin_exists"] for row in target_summary),
            "target_novel_structural_basin": sum(row["novel_structural_basin"] for row in target_summary),
            "targets": len(target_summary),
            "control_basin_exists": sum(row["basin_exists"] for row in control_summary),
            "controls": len(control_summary),
            "global_fraction_targets": stats(row["global_success_fraction"] for row in target_summary),
            "local_fraction_targets": stats(row["local_success_fraction"] for row in target_summary if row["local_success_fraction"] is not None),
            "outcomes": dict(sorted(Counter(row["outcome"] for row in registered).items())),
            "successful_correction": {
                "raw_norm": stats(row["correction"]["raw_norm"]["mean"] for row in successful_target),
                "executable_norm": stats(row["correction"]["executable_norm"]["mean"] for row in successful_target),
                "retention_ratio": stats(row["correction"]["retention_ratio"]["mean"] for row in successful_target),
                "mostly_rewritten_fraction": stats(row["correction"]["mostly_rewritten_fraction"] for row in successful_target),
            },
            "best_shared_parameter": max(shared, key=lambda row: (row["targets_rescued"], row["controls_preserved"], row["parameter_id"])),
            "stochastic_robustness": {
                episode_id: {"successes": sum(row["success"] for row in erows), "new_seeds": len(erows), "success_probability": sum(row["success"] for row in erows) / len(erows)}
                for episode_id, erows in sorted(robust_by_episode.items())
            },
            "cross_state_reuse": reuse_summary,
            "episodes": episode_summary,
        }
    (STUDY / "capacity_summary.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "promoted": promoted,
        "full_existence": {rep: f"{value['target_basin_exists']}/{value['targets']}" for rep, value in output["full"].items()},
        "collisions": {rep: value["collision_rollouts"] for rep, value in output["full"].items()},
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
