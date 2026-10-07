#!/usr/bin/env python3
"""Aggregate full P1 common-256 map and freeze full seed-robustness jobs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "diagnostics/double_bottleneck_eta_basis_redesign"
ETA3 = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol"
LOW = np.asarray((0.5, -0.5, 0.0), dtype=float)
HIGH = np.asarray((1.25, 0.5, 0.75), dtype=float)
WIDTH = HIGH - LOW


def read_rows(pattern: str) -> list[dict]:
    rows = []
    for path in sorted((OUT / "raw").glob(pattern)):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return rows


def candidates(successful: list[int], theta: np.ndarray, points: list[dict]) -> list[dict]:
    unit = (theta - LOW) / WIDTH
    success_set = set(successful)
    output = []
    for eta_index in successful:
        distances = np.linalg.norm(unit - unit[eta_index], axis=1)
        distances[eta_index] = np.inf
        nearest = np.argsort(distances, kind="stable")[:8]
        output.append({
            "eta_index": eta_index,
            "parameter_id": points[eta_index]["parameter_id"],
            "theta": theta[eta_index].tolist(),
            "successful_neighbors_among_8": sum(int(index) in success_set for index in nearest),
            "normalized_distance_to_eta_zero": float(np.linalg.norm(theta[eta_index] / WIDTH)),
        })
    output.sort(key=lambda row: (-row["successful_neighbors_among_8"], row["normalized_distance_to_eta_zero"], row["eta_index"]))
    return output if len(output) <= 8 else output[:8]


def stats(values):
    values = np.asarray(list(values), dtype=float)
    return {"count": int(len(values)), "mean": float(values.mean()), "median": float(np.median(values)), "p25": float(np.percentile(values, 25)), "p75": float(np.percentile(values, 75)), "minimum": float(values.min()), "maximum": float(values.max())}


def main() -> int:
    catalog = json.loads((ETA3 / "episode_catalog.json").read_text())["episodes"]
    targets = [row for row in catalog if row["population"] == "safe_timeout_target"]
    pilot_ids = {row["episode_id"] for row in json.loads((OUT / "pilot_catalog.json").read_text())["episodes"] if row["population"] == "safe_timeout_target"}
    points = json.loads((ETA3 / "eta_points.json").read_text())["points"]
    theta = np.asarray([point["theta"] for point in points], dtype=float)
    scale = json.loads((OUT / "P1_SCALE.json").read_text())["scale"]
    pilot_rows = [row for row in read_rows("P1_pilot_global_shard*.jsonl") if row["population"] == "safe_timeout_target"]
    pending_rows = read_rows("P1_full_global_pending_shard*.jsonl")
    rows = pilot_rows + pending_rows
    if len(pilot_rows) != 12 * 256 or len(pending_rows) != 49 * 256 or len({(row["episode_id"], int(row["eta_index"])) for row in rows}) != 61 * 256:
        raise RuntimeError("full P1 global matrix incomplete")
    lookup = {(row["episode_id"], int(row["eta_index"])): row for row in rows}
    episode_output = []
    pending_jobs = []
    for episode in targets:
        membership = [lookup[(episode["episode_id"], eta_index)]["success"] for eta_index in range(256)]
        successful = np.flatnonzero(membership).astype(int).tolist()
        selected = candidates(successful, theta, points)
        episode_output.append({**episode, "basin_exists": bool(successful), "successful_eta_count": len(successful), "basin_fraction": len(successful) / 256.0, "successful_eta_indices": successful, "candidates": selected})
        if episode["episode_id"] not in pilot_ids:
            for candidate in selected:
                for seed in range(2001, 2017):
                    pending_jobs.append({
                        **episode,
                        "stage": "full_candidate_seed16",
                        "representation": "P1-OrthoFlow3",
                        "eta_index": candidate["eta_index"],
                        "parameter_id": candidate["parameter_id"],
                        "sample_type": "successful_common256_candidate",
                        "theta": candidate["theta"],
                        "ortho_scale": scale,
                        "seed": seed,
                        "job_id": f"full_candidate_seed16|P1-OrthoFlow3|{episode['episode_id']}|{candidate['eta_index']:03d}|{seed}",
                    })
    pilot_candidate_selection = {row["episode_id"]: {candidate["eta_index"] for candidate in row["candidates"]} for row in episode_output if row["episode_id"] in pilot_ids}
    cached_pilot_seed = []
    for row in read_rows("P1_pilot_seed16_shard*.jsonl"):
        if int(row["eta_index"]) in pilot_candidate_selection[row["episode_id"]]:
            cached_pilot_seed.append(row)
    expected_cached = sum(len(values) for values in pilot_candidate_selection.values()) * 16
    if len(cached_pilot_seed) != expected_cached:
        raise RuntimeError("full P1 pilot seed cache mismatch")
    cached_pilot_seed.sort(key=lambda row: (row["episode_id"], int(row["eta_index"]), int(row["seed"])))
    (OUT / "raw/P1_full_seed16_cached_pilot.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in cached_pilot_seed))
    positive = [row for row in episode_output if row["basin_exists"]]
    p0 = json.loads((ETA3 / "long_run_v2/timeout_basin_matrix.json").read_text())["episodes"]
    summary = {
        "schema": "eta_basis_redesign_full_global_v1",
        "P0-3D": {
            "basin_existence": sum(row["basin_exists"] for row in p0),
            "positive_basin_fraction": stats(row["rho"] for row in p0 if row["basin_exists"]),
            "episodes": p0,
        },
        "P1-OrthoFlow3": {
            "basin_existence": len(positive),
            "positive_basin_fraction": stats(row["basin_fraction"] for row in positive),
            "episodes": episode_output,
            "collision_rollouts": sum(row["wall_collision"] or row["agent_collision"] for row in rows),
        },
    }
    (OUT / "full_global_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (OUT / "jobs/P1_full_seed16_pending.json").write_text(json.dumps({"jobs": pending_jobs}, indent=2, sort_keys=True) + "\n")
    manifest = json.loads((OUT / "run_manifest.json").read_text())
    manifest.update({"state": "full_seed_jobs_registered", "p1_full_global_completed": len(rows), "p1_full_basin_existence": len(positive), "p1_full_seed_cached_pilot": len(cached_pilot_seed), "p1_full_seed_pending_jobs": len(pending_jobs)})
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"P0_existence": summary["P0-3D"]["basin_existence"], "P1_existence": len(positive), "P1_positive_median_fraction": summary["P1-OrthoFlow3"]["positive_basin_fraction"]["median"], "cached_seed": len(cached_pilot_seed), "pending_seed": len(pending_jobs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
