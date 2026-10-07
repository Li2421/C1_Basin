#!/usr/bin/env python3
"""Compare common-256 pilot maps and freeze equal-budget seed candidates."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "diagnostics/double_bottleneck_eta_basis_redesign"
ETA3 = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol"
SEED_STUDY = ROOT / "diagnostics/double_bottleneck_eta3_seed_robustness"
LOW = np.asarray((0.5, -0.5, 0.0), dtype=float)
HIGH = np.asarray((1.25, 0.5, 0.75), dtype=float)
WIDTH = HIGH - LOW


def rows_from(pattern: str) -> list[dict]:
    rows = []
    for path in sorted((OUT / "raw").glob(pattern)):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def candidate_rows(successful: list[int], membership: np.ndarray, theta: np.ndarray, points: list[dict]) -> list[dict]:
    ranked = []
    unit = (theta - LOW) / WIDTH
    successful_set = set(successful)
    for eta_index in successful:
        distances = np.linalg.norm(unit - unit[eta_index], axis=1)
        distances[eta_index] = np.inf
        nearest = np.argsort(distances, kind="stable")[:8]
        neighbors = sum(int(index) in successful_set for index in nearest)
        ranked.append({
            "eta_index": eta_index,
            "parameter_id": points[eta_index]["parameter_id"],
            "theta": theta[eta_index].tolist(),
            "successful_neighbors_among_8": neighbors,
            "normalized_distance_to_eta_zero": float(np.linalg.norm(theta[eta_index] / WIDTH)),
        })
    ranked.sort(key=lambda row: (-row["successful_neighbors_among_8"], row["normalized_distance_to_eta_zero"], row["eta_index"]))
    return ranked if len(ranked) <= 8 else ranked[:8]


def main() -> int:
    pilot = json.loads((OUT / "pilot_catalog.json").read_text())["episodes"]
    targets = [row for row in pilot if row["population"] == "safe_timeout_target"]
    controls = [row for row in pilot if row["population"] == "baseline_success_control"]
    points = json.loads((ETA3 / "eta_points.json").read_text())["points"]
    theta = np.asarray([point["theta"] for point in points], dtype=float)
    scale = json.loads((OUT / "P1_SCALE.json").read_text())["scale"]
    p0 = rows_from("P0_pilot_global_cached.jsonl")
    p1 = rows_from("P1_pilot_global_shard*.jsonl")
    if len(p0) != 6144 or len(p1) != 6144:
        raise RuntimeError("pilot global results incomplete")
    if len({row["job_id"] for row in p1}) != 6144:
        raise RuntimeError("duplicate P1 pilot jobs")
    by_rep = {"P0-3D": p0, "P1-OrthoFlow3": p1}
    episode_lookup = {row["episode_id"]: row for row in pilot}
    selection = {name: [] for name in by_rep}
    representation_summary = {}
    jobs = []
    for name, rows in by_rep.items():
        lookup = {(row["episode_id"], int(row["eta_index"])): row for row in rows}
        episode_results = []
        for episode in targets:
            membership = np.asarray([lookup[(episode["episode_id"], eta_index)]["success"] for eta_index in range(256)], dtype=bool)
            successful = np.flatnonzero(membership).astype(int).tolist()
            candidates = candidate_rows(successful, membership, theta, points)
            selection[name].append({"episode_id": episode["episode_id"], "successful_eta_count": len(successful), "successful_eta_indices": successful, "candidate_count": len(candidates), "candidates": candidates})
            episode_results.append({"episode_id": episode["episode_id"], "basin_exists": bool(successful), "successful_eta_count": len(successful), "basin_fraction": len(successful) / 256.0})
            if name == "P1-OrthoFlow3":
                for candidate in candidates:
                    for seed in range(2001, 2017):
                        jobs.append({
                            **episode,
                            "stage": "pilot_candidate_seed16",
                            "representation": name,
                            "eta_index": candidate["eta_index"],
                            "parameter_id": candidate["parameter_id"],
                            "sample_type": "successful_common256_candidate",
                            "theta": candidate["theta"],
                            "ortho_scale": scale,
                            "seed": seed,
                            "job_id": f"pilot_candidate_seed16|{name}|{episode['episode_id']}|{candidate['eta_index']:03d}|{seed}",
                        })
        control_results = []
        for episode in controls:
            successes = sum(lookup[(episode["episode_id"], eta_index)]["success"] for eta_index in range(256))
            control_results.append({"episode_id": episode["episode_id"], "successful_eta_count": int(successes), "preservation_fraction": successes / 256.0})
        representation_summary[name] = {
            "timeout_episodes": episode_results,
            "basin_existence": sum(row["basin_exists"] for row in episode_results),
            "successful_eta_counts": [row["successful_eta_count"] for row in episode_results],
            "median_positive_basin_fraction": float(np.median([row["basin_fraction"] for row in episode_results if row["basin_exists"]])) if any(row["basin_exists"] for row in episode_results) else 0.0,
            "controls": control_results,
            "mean_control_preservation_over_eta": float(np.mean([row["preservation_fraction"] for row in control_results])),
            "collision_rollouts": sum(row["wall_collision"] or row["agent_collision"] for row in rows),
        }

    # Reuse exact P0 16-seed rows for the selected pilot candidates.
    desired_p0 = {(episode["episode_id"], candidate["eta_index"]) for episode in selection["P0-3D"] for candidate in episode["candidates"]}
    cached_seed_rows = []
    for path in sorted((SEED_STUDY / "raw").glob("candidate_seed16_shard*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if (row["episode_id"], int(row["eta_index"])) in desired_p0:
                cached_seed_rows.append(row)
    expected_p0_seed = len(desired_p0) * 16
    if len(cached_seed_rows) != expected_p0_seed:
        raise RuntimeError(f"P0 cached seed rows incomplete: {len(cached_seed_rows)} != {expected_p0_seed}")
    cached_seed_rows.sort(key=lambda row: (row["episode_id"], int(row["eta_index"]), int(row["seed"])))
    (OUT / "raw/P0_pilot_candidate_seed16_cached.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in cached_seed_rows))
    (OUT / "pilot_global_summary.json").write_text(json.dumps({"schema": "eta_basis_redesign_pilot_global_v1", "representations": representation_summary}, indent=2, sort_keys=True) + "\n")
    (OUT / "candidate_selection.json").write_text(json.dumps({"schema": "eta_basis_redesign_candidate_selection_v1", "representations": selection}, indent=2, sort_keys=True) + "\n")
    (OUT / "jobs/P1_pilot_candidate_seed16.json").write_text(json.dumps({"jobs": jobs}, indent=2, sort_keys=True) + "\n")
    manifest = json.loads((OUT / "run_manifest.json").read_text())
    manifest.update({
        "state": "pilot_seed_jobs_registered",
        "p1_pilot_global_completed": len(p1),
        "p0_pilot_seed_cached": len(cached_seed_rows),
        "p1_pilot_seed_jobs": len(jobs),
        "pilot_global_existence": {name: value["basin_existence"] for name, value in representation_summary.items()},
    })
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"existence": {name: value["basin_existence"] for name, value in representation_summary.items()}, "p0_seed_cached": len(cached_seed_rows), "p1_seed_jobs": len(jobs), "p1_collisions": representation_summary["P1-OrthoFlow3"]["collision_rollouts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
