#!/usr/bin/env python3
"""Analyze pre-registered local eta and stochastic MACFlow-seed robustness."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol"


def stats(values):
    values = np.asarray(list(values), dtype=np.float64)
    return {
        "count": int(len(values)),
        "mean": float(values.mean()) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "p10": float(np.percentile(values, 10)) if len(values) else None,
        "p90": float(np.percentile(values, 90)) if len(values) else None,
        "minimum": float(values.min()) if len(values) else None,
        "maximum": float(values.max()) if len(values) else None,
    }


def load(stage: str):
    rows = []
    for path in sorted((STUDY / "raw").glob(f"{stage}_shard*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    unique = {row["job_id"]: row for row in rows}
    if len(unique) != len(rows):
        raise RuntimeError(f"duplicate {stage} result")
    return unique


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_local(value: float) -> str:
    return "broad" if value >= 0.75 else "narrow" if value >= 0.10 else "isolated"


def classify_seed(value: float) -> str:
    return "seed_robust" if value >= 0.75 else "seed_sensitive" if value >= 0.25 else "lucky"


def main() -> int:
    local_manifest = json.loads((STUDY / "jobs/local_robustness.json").read_text())
    stochastic_manifest = json.loads((STUDY / "jobs/stochastic_robustness.json").read_text())
    local = load("local_robustness")
    stochastic = load("stochastic_robustness")
    if len(local) != len(local_manifest["jobs"]):
        raise RuntimeError(f"local incomplete: {len(local)} != {len(local_manifest['jobs'])}")
    if len(stochastic) != len(stochastic_manifest["jobs"]):
        raise RuntimeError(f"stochastic incomplete: {len(stochastic)} != {len(stochastic_manifest['jobs'])}")

    center_by_id = {row["center_id"]: row for row in local_manifest["centers"]}
    local_groups, seed_groups = defaultdict(list), defaultdict(list)
    for job in local_manifest["jobs"]:
        center_id = job["sample_type"].removeprefix("center=")
        local_groups[center_id].append(local[job["job_id"]])
    for job in stochastic_manifest["jobs"]:
        center_id = job["sample_type"].removeprefix("center=")
        seed_groups[center_id].append(stochastic[job["job_id"]])

    local_centers = []
    for center_id, rows in local_groups.items():
        fraction = sum(row["success"] for row in rows) / len(rows)
        center = center_by_id[center_id]
        local_centers.append({
            **center,
            "samples": len(rows),
            "successes": sum(row["success"] for row in rows),
            "success_fraction": fraction,
            "classification": classify_local(fraction),
            "collisions": sum(row["wall_collision"] or row["agent_collision"] for row in rows),
            "terminal_reasons": dict(sorted(Counter(row["outcome"] for row in rows).items())),
        })
    local_by_episode = defaultdict(list)
    for row in local_centers:
        local_by_episode[row["episode_id"]].append(row)
    local_episodes = []
    for episode_id, rows in sorted(local_by_episode.items()):
        best = max(rows, key=lambda row: (row["success_fraction"], row["successes"], row["source_parameter_id"]))
        local_episodes.append({
            "episode_id": episode_id,
            "representative_count": len(rows),
            "best_local_success_fraction": best["success_fraction"],
            "best_center_parameter_id": best["source_parameter_id"],
            "best_center_role": best["role"],
            "episode_locally_robust": best["success_fraction"] >= 0.50,
            "representative_classifications": dict(sorted(Counter(row["classification"] for row in rows).items())),
        })
    local_output = {
        "schema": "double_bottleneck_eta3_local_robustness_v1",
        "protocol": {"normalized_radius": 0.03125, "broad_threshold": 0.75, "isolated_upper_exclusive": 0.10, "episode_robust_threshold": 0.50},
        "centers": sorted(local_centers, key=lambda row: row["center_id"]),
        "episodes": local_episodes,
        "summary": {
            "positive_episodes": len(local_episodes),
            "locally_robust_episodes": sum(row["episode_locally_robust"] for row in local_episodes),
            "center_classifications": dict(sorted(Counter(row["classification"] for row in local_centers).items())),
            "center_local_fraction": stats(row["success_fraction"] for row in local_centers),
            "best_episode_local_fraction": stats(row["best_local_success_fraction"] for row in local_episodes),
            "collisions": sum(row["collisions"] for row in local_centers),
        },
    }

    seed_centers = []
    for center_id, rows in seed_groups.items():
        fraction = sum(row["success"] for row in rows) / len(rows)
        center = center_by_id[center_id]
        lengths = [row["episode_steps"] for row in rows]
        seed_centers.append({
            **center,
            "seeds": len(rows),
            "successes": sum(row["success"] for row in rows),
            "success_fraction": fraction,
            "classification": classify_seed(fraction),
            "collisions": sum(row["wall_collision"] or row["agent_collision"] for row in rows),
            "terminal_reasons": dict(sorted(Counter(row["outcome"] for row in rows).items())),
            "episode_length_mean": float(np.mean(lengths)),
            "episode_length_std": float(np.std(lengths)),
        })
    seed_by_episode = defaultdict(list)
    for row in seed_centers:
        seed_by_episode[row["episode_id"]].append(row)
    seed_episodes = []
    for episode_id, rows in sorted(seed_by_episode.items()):
        best = max(rows, key=lambda row: (row["success_fraction"], row["successes"], row["source_parameter_id"]))
        seed_episodes.append({
            "episode_id": episode_id,
            "representative_count": len(rows),
            "best_seed_success_fraction": best["success_fraction"],
            "best_center_parameter_id": best["source_parameter_id"],
            "best_center_role": best["role"],
            "best_classification": best["classification"],
            "representative_classifications": dict(sorted(Counter(row["classification"] for row in rows).items())),
        })
    seed_output = {
        "schema": "double_bottleneck_eta3_stochastic_robustness_v1",
        "protocol": {"seeds": stochastic_manifest["seeds"], "seed_robust_threshold": 0.75, "lucky_upper_exclusive": 0.25},
        "centers": sorted(seed_centers, key=lambda row: row["center_id"]),
        "episodes": seed_episodes,
        "summary": {
            "positive_episodes": len(seed_episodes),
            "episodes_with_seed_robust_representative": sum(row["best_classification"] == "seed_robust" for row in seed_episodes),
            "center_classifications": dict(sorted(Counter(row["classification"] for row in seed_centers).items())),
            "center_seed_fraction": stats(row["success_fraction"] for row in seed_centers),
            "best_episode_seed_fraction": stats(row["best_seed_success_fraction"] for row in seed_episodes),
            "collisions": sum(row["collisions"] for row in seed_centers),
        },
    }
    (STUDY / "local_robustness_results.json").write_text(json.dumps(local_output, indent=2, sort_keys=True) + "\n")
    (STUDY / "stochastic_seed_robustness_results.json").write_text(json.dumps(seed_output, indent=2, sort_keys=True) + "\n")
    (STUDY / "ROBUSTNESS_SUMMARY.json").write_text(json.dumps({
        "schema": "double_bottleneck_eta3_robustness_summary_v1",
        "local": local_output["summary"],
        "stochastic": seed_output["summary"],
    }, indent=2, sort_keys=True) + "\n")
    raw_paths = sorted((STUDY / "raw").glob("local_robustness_shard*.jsonl")) + sorted((STUDY / "raw").glob("stochastic_robustness_shard*.jsonl"))
    (STUDY / "robustness_outcomes_manifest.json").write_text(json.dumps({
        "schema": "double_bottleneck_eta3_robustness_outcomes_manifest_v1",
        "total_rows": len(local) + len(stochastic),
        "files": [{
            "path": str(path.relative_to(ROOT)),
            "rows": sum(1 for line in path.read_text().splitlines() if line.strip()),
            "sha256": sha(path),
        } for path in raw_paths],
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"local": local_output["summary"], "stochastic": seed_output["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
