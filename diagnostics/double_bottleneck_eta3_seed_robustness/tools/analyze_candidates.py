#!/usr/bin/env python3
"""Aggregate candidate seed trials, freeze robust centers, and emit follow-up jobs."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol"
OUT = ROOT / "diagnostics/double_bottleneck_eta3_seed_robustness"
LOW = np.asarray((0.5, -0.5, 0.0), dtype=float)
HIGH = np.asarray((1.25, 0.5, 0.75), dtype=float)
WIDTH = HIGH - LOW


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(pattern: str) -> list[dict]:
    rows = []
    for path in sorted((OUT / "raw").glob(pattern)):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def main() -> int:
    rows = read_rows("candidate_seed16_shard*.jsonl")
    if len(rows) != 928 or len({row["job_id"] for row in rows}) != 928:
        raise RuntimeError(f"candidate runs incomplete/duplicated: {len(rows)}")
    expected_seeds = set(range(2001, 2017))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["episode_id"], int(row["eta_index"]))].append(row)
    if len(grouped) != 58 or any({row["seed"] for row in group} != expected_seeds for group in grouped.values()):
        raise RuntimeError("candidate pair/seed coverage mismatch")

    selection = json.loads((OUT / "candidate_selection.json").read_text())["episodes"]
    candidate_meta = {
        (episode["episode_id"], int(candidate["eta_index"])): candidate
        for episode in selection for candidate in episode["candidates"]
    }
    catalog = json.loads((SOURCE / "episode_catalog.json").read_text())["episodes"]
    episode_meta = {episode["episode_id"]: episode for episode in catalog}
    points = json.loads((SOURCE / "eta_points.json").read_text())["points"]

    candidate_results = []
    by_episode = defaultdict(list)
    for (episode_id, eta_index), group in sorted(grouped.items()):
        group = sorted(group, key=lambda row: row["seed"])
        meta = candidate_meta[(episode_id, eta_index)]
        item = {
            "episode_id": episode_id,
            "eta_index": eta_index,
            "parameter_id": points[eta_index]["parameter_id"],
            "theta": points[eta_index]["theta"],
            "successful_neighbors_among_8": meta["successful_neighbors_among_8"],
            "normalized_distance_to_eta_zero": meta["normalized_distance_to_eta_zero"],
            "successes": sum(row["success"] for row in group),
            "trials": 16,
            "Q_seed": sum(row["success"] for row in group) / 16.0,
            "outcomes": dict(sorted(Counter(row["outcome"] for row in group).items())),
            "seed_outcomes": [{"seed": row["seed"], "success": row["success"], "outcome": row["outcome"], "episode_steps": row["episode_steps"]} for row in group],
        }
        candidate_results.append(item)
        by_episode[episode_id].append(item)

    robust = []
    for episode_id, candidates in sorted(by_episode.items()):
        best = min(candidates, key=lambda row: (-row["Q_seed"], -row["successful_neighbors_among_8"], row["normalized_distance_to_eta_zero"], row["eta_index"]))
        q = best["Q_seed"]
        category = "strongly_robust" if q >= 0.75 else "moderately_robust" if q >= 0.50 else "weakly_robust" if q >= 0.25 else "seed_fragile"
        robust.append({
            **episode_meta[episode_id],
            "eta_robust": {key: best[key] for key in ("eta_index", "parameter_id", "theta", "successful_neighbors_among_8", "normalized_distance_to_eta_zero")},
            "Q_max": q,
            "successes": best["successes"],
            "trials": best["trials"],
            "category": category,
            "candidate_count": len(candidates),
        })
    if len(robust) != 38:
        raise RuntimeError("robust-center cardinality mismatch")

    category_counts = Counter(row["category"] for row in robust)
    eligible = [row for row in robust if row["Q_max"] >= 0.50]
    unique_eta = sorted({int(row["eta_robust"]["eta_index"]) for row in robust})
    offsets = (2.0 * qmc.Sobol(3, scramble=True, seed=20260926).random_base2(4) - 1.0) * 0.05
    local_jobs = []
    for episode in eligible:
        center = np.asarray(episode["eta_robust"]["theta"], dtype=float)
        center_unit = (center - LOW) / WIDTH
        for local_index, offset in enumerate(offsets):
            local_theta = LOW + np.clip(center_unit + offset, 0.0, 1.0) * WIDTH
            for seed in range(3001, 3009):
                local_jobs.append({
                    **{key: episode[key] for key in ("episode_id", "population", "set", "family_id", "regime", "rollout_id", "baseline_outcome")},
                    "stage": "local_seed_confirmation",
                    "representation": "P0-3D",
                    "center_eta_index": episode["eta_robust"]["eta_index"],
                    "eta_index": episode["eta_robust"]["eta_index"],
                    "parameter_id": f"L{local_index:02d}",
                    "sample_type": "fixed_sobol_local_radius_0.05",
                    "theta": local_theta.tolist(),
                    "seed": seed,
                    "job_id": f"local_seed|{episode['episode_id']}|{local_index:02d}|{seed}",
                })

    controls = [episode for episode in catalog if episode["population"] == "baseline_success_control"]
    targets = [episode for episode in catalog if episode["population"] == "safe_timeout_target"]
    if len(controls) != 24 or len(targets) != 61:
        raise RuntimeError("frozen episode population mismatch")
    control_jobs = []
    reuse_jobs = []
    for eta_index in unique_eta:
        point = points[eta_index]
        for episode in controls:
            for seed in range(4001, 4005):
                control_jobs.append({
                    **episode,
                    "stage": "robust_eta_controls",
                    "representation": "P0-3D",
                    "eta_index": eta_index,
                    "parameter_id": point["parameter_id"],
                    "sample_type": "eta_robust_control_preservation",
                    "theta": point["theta"],
                    "seed": seed,
                    "job_id": f"robust_control|{eta_index:03d}|{episode['episode_id']}|{seed}",
                })
        for episode in targets:
            reuse_jobs.append({
                **episode,
                "stage": "robust_eta_cross_state_screen",
                "representation": "P0-3D",
                "eta_index": eta_index,
                "parameter_id": point["parameter_id"],
                "sample_type": "eta_robust_reuse_screen",
                "theta": point["theta"],
                "seed": 5001,
                "job_id": f"reuse_screen|{eta_index:03d}|{episode['episode_id']}|5001",
            })

    (OUT / "Q_seed.json").write_text(json.dumps({"candidate_pairs": candidate_results}, indent=2, sort_keys=True) + "\n")
    (OUT / "eta_robust.json").write_text(json.dumps({"episodes": robust}, indent=2, sort_keys=True) + "\n")
    (OUT / "jobs/local_seed_confirmation.json").write_text(json.dumps({"jobs": local_jobs}, indent=2, sort_keys=True) + "\n")
    (OUT / "jobs/control_preservation.json").write_text(json.dumps({"jobs": control_jobs}, indent=2, sort_keys=True) + "\n")
    (OUT / "jobs/cross_state_screen.json").write_text(json.dumps({"jobs": reuse_jobs}, indent=2, sort_keys=True) + "\n")
    summary = {
        "candidate_jobs_completed": len(rows),
        "candidate_eta_pairs": len(candidate_results),
        "candidate_collision_rollouts": sum(row["wall_collision"] or row["agent_collision"] for row in rows),
        "category_counts": dict(sorted(category_counts.items())),
        "episodes_Q_max_ge_0_50": len(eligible),
        "median_Q_max": float(np.median([row["Q_max"] for row in robust])),
        "unique_eta_robust": len(unique_eta),
        "local_jobs": len(local_jobs),
        "control_jobs": len(control_jobs),
        "reuse_screen_jobs": len(reuse_jobs),
    }
    (OUT / "CANDIDATE_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest = json.loads((OUT / "run_manifest.json").read_text())
    manifest.update({
        "completed_candidate_jobs": len(rows),
        "candidate_raw_files": [{"path": str(path.relative_to(ROOT)), "sha256": sha(path), "rows": sum(1 for line in path.read_text().splitlines() if line.strip())} for path in sorted((OUT / "raw").glob("candidate_seed16_shard*.jsonl"))],
        "robust_episodes": len(eligible),
        "unique_eta_robust": len(unique_eta),
        "local_jobs": len(local_jobs),
        "control_jobs": len(control_jobs),
        "reuse_screen_jobs": len(reuse_jobs),
        "state": "followups_registered",
    })
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
