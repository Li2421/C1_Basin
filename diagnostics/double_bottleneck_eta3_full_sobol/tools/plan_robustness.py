#!/usr/bin/env python3
"""Materialize the pre-registered local-eta and MACFlow-seed jobs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc


ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / "diagnostics/double_bottleneck_eta3_full_sobol"
LOW = np.asarray((0.5, -0.5, 0.0), dtype=np.float64)
HIGH = np.asarray((1.25, 0.5, 0.75), dtype=np.float64)
RADIUS = 0.03125
SEEDS = (1103, 1201, 1301, 1409, 1511, 1601, 1709, 1801)


def base_job(episode: dict, *, stage: str, parameter_id: str, sample_type: str,
             theta: list[float], seed: int, job_id: str) -> dict:
    return {
        **episode,
        "pilot_stratum": stage,
        "stage": stage,
        "representation": "P0-3D",
        "parameter_id": parameter_id,
        "sample_type": sample_type,
        "theta": theta,
        "seed": seed,
        "job_id": job_id,
    }


def main() -> int:
    selected = json.loads((STUDY / "selected_success_representatives.json").read_text())["episodes"]
    catalog = {
        row["episode_id"]: row
        for row in json.loads((STUDY / "episode_catalog.json").read_text())["episodes"]
    }

    # All offsets live inside the registered radius in normalized eta coordinates.
    axes = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            offset = np.zeros(3, dtype=np.float64)
            offset[axis] = sign * RADIUS
            axes.append(offset)
    cube = 2.0 * qmc.Sobol(d=3, scramble=True, seed=920031).random_base2(m=5) - 1.0
    norms = np.linalg.norm(cube, axis=1)
    directions = cube / np.maximum(norms[:, None], 1e-12)
    # A fixed radial transform gives space-filling interior samples, not only a shell.
    radial = ((np.arange(32, dtype=np.float64) + 0.5) / 32.0) ** (1.0 / 3.0)
    offsets = axes + [RADIUS * radial[index] * directions[index] for index in range(32)]

    local_jobs, stochastic_jobs, centers = [], [], []
    for episode_selection in selected:
        episode_id = episode_selection["episode_id"]
        episode = catalog[episode_id]
        for center_index, representative in enumerate(episode_selection["representatives"]):
            center_id = f"{episode_id}|{representative['parameter_id']}|{representative['role']}"
            center_theta = np.asarray(representative["theta"], dtype=np.float64)
            center_unit = (center_theta - LOW) / (HIGH - LOW)
            center_record = {
                "center_id": center_id,
                "episode_id": episode_id,
                "source_parameter_id": representative["parameter_id"],
                "role": representative["role"],
                "theta": center_theta.tolist(),
                "timeout_coverage": representative["timeout_coverage"],
                "controls_preserved": representative["controls_preserved"],
            }
            centers.append(center_record)
            seen = set()
            for local_index, offset in enumerate(offsets):
                local_unit = np.clip(center_unit + offset, 0.0, 1.0)
                local_theta = LOW + local_unit * (HIGH - LOW)
                key = tuple(np.round(local_theta, 14))
                if key in seen or np.allclose(local_theta, center_theta, atol=1e-14, rtol=0):
                    continue
                seen.add(key)
                pid = f"LOCAL_C{center_index:02d}_{local_index:03d}"
                job_id = f"local|{episode_id}|{representative['parameter_id']}|{representative['role']}|{pid}"
                local_jobs.append(base_job(
                    episode,
                    stage="local_robustness",
                    parameter_id=pid,
                    sample_type=f"center={center_id}",
                    theta=local_theta.tolist(),
                    seed=int(episode["seed"]),
                    job_id=job_id,
                ))
            for seed_index, seed in enumerate(SEEDS):
                pid = f"SEED_{seed:04d}"
                job_id = f"stochastic|{episode_id}|{representative['parameter_id']}|{representative['role']}|{pid}"
                stochastic_jobs.append(base_job(
                    episode,
                    stage="stochastic_robustness",
                    parameter_id=pid,
                    sample_type=f"center={center_id}",
                    theta=center_theta.tolist(),
                    seed=seed,
                    job_id=job_id,
                ))

    if len({row["job_id"] for row in local_jobs}) != len(local_jobs):
        raise RuntimeError("duplicate local jobs")
    if len({row["job_id"] for row in stochastic_jobs}) != len(stochastic_jobs):
        raise RuntimeError("duplicate stochastic jobs")
    (STUDY / "jobs/local_robustness.json").write_text(json.dumps({
        "schema": "double_bottleneck_eta3_local_jobs_v1",
        "normalized_radius": RADIUS,
        "offset_construction": "6 axis endpoints plus 32 scrambled-Sobol directions with fixed interior radial transform; clip to global domain",
        "centers": centers,
        "jobs": local_jobs,
    }, indent=2, sort_keys=True) + "\n")
    (STUDY / "jobs/stochastic_robustness.json").write_text(json.dumps({
        "schema": "double_bottleneck_eta3_stochastic_jobs_v1",
        "seeds": SEEDS,
        "centers": centers,
        "jobs": stochastic_jobs,
    }, indent=2, sort_keys=True) + "\n")
    (STUDY / "robustness_plan.json").write_text(json.dumps({
        "schema": "double_bottleneck_eta3_robustness_plan_v1",
        "positive_episodes": len(selected),
        "representative_centers": len(centers),
        "local_jobs": len(local_jobs),
        "stochastic_jobs": len(stochastic_jobs),
        "local_radius_normalized": RADIUS,
        "macflow_seeds": SEEDS,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"positive_episodes": len(selected), "centers": len(centers), "local_jobs": len(local_jobs), "stochastic_jobs": len(stochastic_jobs)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
