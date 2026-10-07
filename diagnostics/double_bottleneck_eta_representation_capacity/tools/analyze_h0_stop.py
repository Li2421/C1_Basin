#!/usr/bin/env python3
"""Freeze the late H0 stop triggered by full-population P0 cross-state reuse."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta_representation_capacity"
PRIOR = ROOT / "diagnostics/double_bottleneck_eta3_basin/all_rollout_outcomes.json"


def main() -> int:
    prior = json.loads(PRIOR.read_text())["rollouts"]
    old_rows = [row for row in prior if row["population"] == "safe_timeout_target" and row["eta_id"] == "A004"]
    old_success = {f"{row['set']}|{int(row['rollout_id']):03d}" for row in old_rows if row["success"]}
    old_control_rows = [row for row in prior if row["population"] == "baseline_success_control" and row["eta_id"] == "A004"]
    old_control_success = {f"{row['set']}|{int(row['rollout_id']):03d}" for row in old_control_rows if row["success"]}
    inherited = []
    for path in sorted((STUDY / "raw").glob("P1_Agent6_full_p0_inheritance_shard*.jsonl")):
        inherited.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if len(inherited) != 510:
        raise RuntimeError(f"full P0 inheritance incomplete: {len(inherited)} != 510")
    target_rows = [row for row in inherited if row["population"] == "safe_timeout_target"]
    control_rows = [row for row in inherited if row["population"] == "baseline_success_control"]
    new_success = {row["episode_id"] for row in target_rows if row["success"]}
    new_control_success = {row["episode_id"] for row in control_rows if row["success"]}
    per_eta = defaultdict(lambda: {"target_success_episodes": set(), "control_success_episodes": set(), "target_success_rollouts": 0, "control_success_rollouts": 0})
    for row in inherited:
        if not row["success"]:
            continue
        value = per_eta[row["parameter_id"]]
        key = "target" if row["population"] == "safe_timeout_target" else "control"
        value[f"{key}_success_episodes"].add(row["episode_id"])
        value[f"{key}_success_rollouts"] += 1
    per_eta_json = {}
    for key, value in sorted(per_eta.items()):
        per_eta_json[key] = {
            **{name: count for name, count in value.items() if name.endswith("rollouts")},
            "target_success_episodes": sorted(value["target_success_episodes"]),
            "control_success_episodes": sorted(value["control_success_episodes"]),
            "target_coverage": len(value["target_success_episodes"]),
            "control_preservation": len(value["control_success_episodes"]),
        }
    union = old_success | new_success
    partial = []
    paths = list((STUDY / "raw").glob("P1_Agent6_full_stage_a_shard*.jsonl"))
    paths += list((STUDY / "incomplete_after_h0_stop/raw").glob("P1_Agent6_full_stage_a_shard*.jsonl"))
    for path in sorted(paths):
        partial.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    output = {
        "schema": "double_bottleneck_eta_capacity_h0_stop_v1",
        "decision": "STOP_SEARCH_INSUFFICIENCY",
        "reason": "Six P0 points discovered by the dense pilot audit and then applied unchanged to all frozen episodes more than doubled target coverage; this materially changes the prior 15/61 conclusion before structural attribution.",
        "old_A004": {"target_success": len(old_success), "target_episodes": sorted(old_success), "control_success": len(old_control_success), "control_episodes": sorted(old_control_success)},
        "new_dense_P0_points": {
            "points": 6,
            "rollouts": len(inherited),
            "target_success": len(new_success),
            "target_episodes": sorted(new_success),
            "control_success": len(new_control_success),
            "collision_rollouts": sum(row["wall_collision"] or row["agent_collision"] for row in inherited),
            "per_eta": per_eta_json,
        },
        "combined_observed_P0": {
            "target_success": len(union),
            "targets": 61,
            "fraction": len(union) / 61.0,
            "overlap": len(old_success & new_success),
            "new_beyond_A004": len(new_success - old_success),
            "A004_only": len(old_success - new_success),
            "target_episodes": sorted(union),
            "control_success": len(old_control_success | new_control_success),
            "controls": 24,
            "control_fraction": len(old_control_success | new_control_success) / 24.0,
            "control_overlap": len(old_control_success & new_control_success),
            "control_episodes": sorted(old_control_success | new_control_success),
        },
        "structural_full_sweep": {
            "status": "stopped_and_excluded",
            "partial_rollouts": len(partial),
            "registered_rollouts": 10965,
            "reason": "H0 stop was reached before full Agent6 completion; partial structural results are not used scientifically.",
        },
    }
    (STUDY / "H0_STOP_AUDIT.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"decision": output["decision"], "old_A004": len(old_success), "new_six": len(new_success), "union": len(union), "partial_structural_excluded": len(partial)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
