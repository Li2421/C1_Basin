#!/usr/bin/env python3
"""Quantify the behavioral mechanism in saved representative trace pairs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta3_basin"


def first_crossings(positions: np.ndarray, directions: np.ndarray, planes: np.ndarray) -> np.ndarray:
    crossed = directions[None] * (positions[:, :, 0] - planes[None]) >= 0
    result = []
    for agent in range(4):
        indices = np.flatnonzero(crossed[:, agent])
        result.append(int(indices[0]) if len(indices) else -1)
    return np.asarray(result, dtype=int)


def summarize(data) -> dict:
    positions = np.asarray(data["positions"], dtype=np.float64)
    dt = float(data["dt"])
    directions = -np.sign(positions[0, :, 0]).astype(int)
    first_planes = np.where(directions > 0, -0.79, 0.79)
    second_planes = np.where(directions > 0, 2.01, -2.01)
    first = first_crossings(positions, directions, first_planes)
    second = first_crossings(positions, directions, second_planes)
    speeds = np.linalg.norm(np.diff(positions, axis=0), axis=-1) / dt
    valid_transit = (first >= 0) & (second >= 0)
    transit = np.where(valid_transit, (second - first) * dt, np.nan)
    final_step = len(positions) - 1
    return {
        "steps": final_step,
        "duration_seconds": final_step * dt,
        "first_resource_crossing_seconds_per_agent": np.where(first >= 0, first * dt, np.nan).tolist(),
        "second_resource_crossing_seconds_per_agent": np.where(second >= 0, second * dt, np.nan).tolist(),
        "last_second_resource_clearance_seconds": float(np.max(second[second >= 0]) * dt) if np.any(second >= 0) else None,
        "mean_between_resources_seconds": float(np.nanmean(transit)) if np.any(valid_transit) else None,
        "mean_waiting_seconds_per_agent": float(np.mean(np.sum(speeds < 0.025, axis=0) * dt)),
        "waiting_seconds_per_agent": (np.sum(speeds < 0.025, axis=0) * dt).tolist(),
        "mean_g_norm": float(np.mean(data["g_norm"])),
        "mean_second_projection_norm": float(np.mean(data["second_projection_norm"])),
        "final_total_goal_error": float(data["total_goal_error"][-1]),
    }


def main() -> int:
    metadata = json.loads((STUDY / "representatives/metadata.json").read_text())["episodes"]
    episodes = []
    for item in metadata:
        summaries = {}
        for label in ("eta0", "eta_success"):
            data = np.load(ROOT / item["traces"][label]["npz"])
            summaries[label] = summarize(data)
            summaries[label]["terminal"] = item["traces"][label]["terminal"]
            summaries[label]["eta"] = item["traces"][label]["eta"]
        episodes.append({"episode_id": item["episode_id"], "regime": item["regime"], "basin_type": item["basin_type"], "traces": summaries})
    output = {"schema": "double_bottleneck_eta3_representative_analysis_v1", "episodes": episodes}
    (STUDY / "representative_trajectory_analysis.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
