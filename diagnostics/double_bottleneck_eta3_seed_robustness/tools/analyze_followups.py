#!/usr/bin/env python3
"""Aggregate controls/reuse screen and freeze top-five reuse confirmation jobs."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol"
OUT = ROOT / "diagnostics/double_bottleneck_eta3_seed_robustness"
WIDTH = np.asarray((0.75, 1.0, 0.75), dtype=float)


def read_rows(pattern: str) -> list[dict]:
    rows = []
    for path in sorted((OUT / "raw").glob(pattern)):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def main() -> int:
    robust = json.loads((OUT / "eta_robust.json").read_text())["episodes"]
    eta_indices = sorted({int(row["eta_robust"]["eta_index"]) for row in robust})
    points = json.loads((SOURCE / "eta_points.json").read_text())["points"]
    catalog = json.loads((SOURCE / "episode_catalog.json").read_text())["episodes"]
    controls = [row for row in catalog if row["population"] == "baseline_success_control"]
    targets = [row for row in catalog if row["population"] == "safe_timeout_target"]

    control_rows = read_rows("control_shard*.jsonl")
    reuse_rows = read_rows("reuse_screen_shard*.jsonl")
    expected_control = len(eta_indices) * 24 * 4
    expected_reuse = len(eta_indices) * 61
    if len(control_rows) != expected_control or len({row["job_id"] for row in control_rows}) != expected_control:
        raise RuntimeError("control evaluation incomplete/duplicated")
    if len(reuse_rows) != expected_reuse or len({row["job_id"] for row in reuse_rows}) != expected_reuse:
        raise RuntimeError("reuse screen incomplete/duplicated")

    control_groups = defaultdict(list)
    for row in control_rows:
        control_groups[(int(row["eta_index"]), row["episode_id"])].append(row)
    control_eta = []
    for eta_index in eta_indices:
        per_control = []
        for episode in controls:
            group = sorted(control_groups[(eta_index, episode["episode_id"])], key=lambda row: row["seed"])
            if len(group) != 4 or {row["seed"] for row in group} != set(range(4001, 4005)):
                raise RuntimeError("control seed coverage mismatch")
            successes = sum(row["success"] for row in group)
            per_control.append({
                "episode_id": episode["episode_id"],
                "successes": successes,
                "trials": 4,
                "preservation_probability": successes / 4.0,
                "outcomes": dict(sorted(Counter(row["outcome"] for row in group).items())),
            })
        total = sum(row["successes"] for row in per_control)
        control_eta.append({
            "eta_index": eta_index,
            "parameter_id": points[eta_index]["parameter_id"],
            "theta": points[eta_index]["theta"],
            "success_trials": total,
            "total_trials": 96,
            "mean_preservation_probability": total / 96.0,
            "controls_preserved_all_4_seeds": sum(row["successes"] == 4 for row in per_control),
            "controls_degraded_all_4_seeds": sum(row["successes"] == 0 for row in per_control),
            "per_control": per_control,
        })

    reuse_groups = defaultdict(list)
    for row in reuse_rows:
        reuse_groups[int(row["eta_index"])].append(row)
    reuse_eta = []
    for eta_index in eta_indices:
        group = sorted(reuse_groups[eta_index], key=lambda row: row["episode_id"])
        if len(group) != 61 or {row["seed"] for row in group} != {5001}:
            raise RuntimeError("reuse screen seed/population mismatch")
        rescued = [row["episode_id"] for row in group if row["success"]]
        reuse_eta.append({
            "eta_index": eta_index,
            "parameter_id": points[eta_index]["parameter_id"],
            "theta": points[eta_index]["theta"],
            "screen_seed": 5001,
            "coverage": len(rescued),
            "coverage_fraction": len(rescued) / 61.0,
            "rescued_episode_ids": rescued,
            "outcomes": dict(sorted(Counter(row["outcome"] for row in group).items())),
        })
    ranked = sorted(reuse_eta, key=lambda row: (-row["coverage"], float(np.linalg.norm(np.asarray(row["theta"]) / WIDTH)), row["eta_index"]))
    top5 = ranked[:5]
    target_meta = {row["episode_id"]: row for row in targets}
    confirmation_jobs = []
    for item in top5:
        for episode_id in item["rescued_episode_ids"]:
            episode = target_meta[episode_id]
            for seed in range(6001, 6009):
                confirmation_jobs.append({
                    **episode,
                    "stage": "top5_reuse_seed_confirmation",
                    "representation": "P0-3D",
                    "eta_index": item["eta_index"],
                    "parameter_id": item["parameter_id"],
                    "sample_type": "top5_robust_eta_reuse_confirmation",
                    "theta": item["theta"],
                    "seed": seed,
                    "job_id": f"top5_reuse|{item['eta_index']:03d}|{episode_id}|{seed}",
                })

    (OUT / "control_preservation.json").write_text(json.dumps({"eta": control_eta}, indent=2, sort_keys=True) + "\n")
    (OUT / "cross_state_reuse_screen.json").write_text(json.dumps({"eta": reuse_eta, "top5": top5}, indent=2, sort_keys=True) + "\n")
    (OUT / "jobs/top5_reuse_confirmation.json").write_text(json.dumps({"jobs": confirmation_jobs}, indent=2, sort_keys=True) + "\n")
    summary = {
        "unique_eta_robust": len(eta_indices),
        "control_jobs_completed": len(control_rows),
        "reuse_screen_jobs_completed": len(reuse_rows),
        "top5": [{"eta_index": row["eta_index"], "coverage": row["coverage"]} for row in top5],
        "top5_confirmation_jobs": len(confirmation_jobs),
        "collisions": sum(row["wall_collision"] or row["agent_collision"] for row in control_rows + reuse_rows),
    }
    (OUT / "FOLLOWUP_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = json.loads((OUT / "run_manifest.json").read_text())
    manifest.update({
        "completed_control_jobs": len(control_rows),
        "completed_reuse_screen_jobs": len(reuse_rows),
        "top5_confirmation_jobs": len(confirmation_jobs),
        "state": "top5_confirmation_registered",
    })
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
