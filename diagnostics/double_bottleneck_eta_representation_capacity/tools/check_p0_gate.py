#!/usr/bin/env python3
"""Freeze the H0 dense-search gate before running expanded representations."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta_representation_capacity"


def main() -> int:
    rows = []
    for path in sorted((STUDY / "raw").glob("P0_3D_stage_a*_shard*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    episodes = json.loads((STUDY / "pilot_catalog.json").read_text())["episodes"]
    expected = len(episodes) * 257
    if len(rows) != expected:
        raise RuntimeError(f"P0 dense incomplete: {len(rows)} != {expected}")
    by_episode = {episode["episode_id"]: [row for row in rows if row["episode_id"] == episode["episode_id"]] for episode in episodes}
    zero_errors = []
    details = []
    for episode in episodes:
        erows = by_episode[episode["episode_id"]]
        zero = [row for row in erows if row["parameter_id"] == "ZERO"]
        if len(zero) != 1 or zero[0]["outcome"] != episode["baseline_outcome"] or zero[0]["episode_steps"] != episode["baseline_steps"]:
            zero_errors.append(episode["episode_id"])
        domain = [row for row in erows if row["parameter_id"] != "ZERO"]
        success = [row for row in domain if row["success"]]
        details.append({
            "episode_id": episode["episode_id"], "stratum": episode["pilot_stratum"],
            "success_exists": bool(success), "successful_samples": len(success),
            "basin_fraction": len(success) / 256.0,
            "successful_parameter_ids": [row["parameter_id"] for row in success],
        })
    new_negative = sum(item["success_exists"] for item in details if item["stratum"] == "p0_negative")
    output = {
        "schema": "double_bottleneck_eta_capacity_p0_dense_gate_v1",
        "rollouts": len(rows), "eta_zero_errors": zero_errors,
        "collision_rollouts": sum(row["wall_collision"] or row["agent_collision"] for row in rows),
        "existence_by_stratum": {
            stratum: {"episodes": sum(item["stratum"] == stratum for item in details), "basin_exists": sum(item["stratum"] == stratum and item["success_exists"] for item in details)}
            for stratum in ("p0_positive", "p0_negative", "baseline_success_control")
        },
        "new_p0_negative_successes": new_negative,
        "stop_threshold": 3,
        "decision": "STOP_SEARCH_INSUFFICIENCY" if new_negative >= 3 else "CONTINUE_STRUCTURAL_AUDIT",
        "outcomes": dict(sorted(Counter(row["outcome"] for row in rows if row["parameter_id"] != "ZERO").items())),
        "episodes": details,
    }
    (STUDY / "p0_dense_search_audit.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
