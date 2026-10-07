"""Summarize Phase 1 and freeze the single allowed adaptive directional step."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from scipy.stats import beta

from .setup import CURRENT, HERE, STATES, existing_groups, key, sha, dump


FRESH_SEEDS = list(range(95106001, 95106033))


def classify(successes: int, n: int):
    lo = float(beta.ppf(0.05, successes, n - successes + 1)) if successes else 0.0
    hi = float(beta.ppf(0.95, successes + 1, n - successes)) if successes < n else 1.0
    if n >= 16 and lo >= 0.8:
        return "SUCCESS_CELL", [lo, hi]
    if n >= 16 and hi <= 0.2:
        return "FAILURE_CELL", [lo, hi]
    return "UNKNOWN_CELL", [lo, hi]


def phase2_points():
    return {
        "D1_pair231": [
            (0.578125, -0.3125, -0.125),
            (0.578125, -0.34375, -0.125),
            (0.578125, -0.375, -0.125),
            (0.578125, -0.40625, -0.125),
            (0.5703125, -0.34375, -0.125),
            (0.5703125, -0.375, -0.125),
        ],
        "D2_pair228": [
            (0.375, -0.53125, 0.0),
            (0.375, -0.5, 0.0),
            (0.375, -0.46875, 0.0),
            (0.375, -0.4375, 0.0),
            (0.359375, -0.5, 0.0),
            (0.359375, -0.46875, 0.0),
            (0.390625, -0.5, 0.0),
        ],
        "D4_pair227": [
            (0.359375, -0.4375, -0.0625),
            (0.375, -0.40625, -0.0625),
            (0.3515625, -0.4375, -0.0625),
            (0.375, -0.390625, -0.0625),
            (0.3671875, -0.421875, -0.0625),
            (0.3671875, -0.4375, -0.0625),
            (0.375, -0.421875, -0.0625),
            (0.359375, -0.421875, -0.0625),
        ],
    }


def main():
    protocol = json.loads((HERE / "protocol.json").read_text())
    groups = existing_groups()
    for path in sorted((HERE / "raw").glob("phase1_*/manifest.json")):
        for row in json.loads(path.read_text())["records"]:
            groups[(row["state_id"], key(row["eta"]))][row["seed"]] = row
    cells = []
    for state, points in protocol["phase1_points"].items():
        for eta_values in points:
            eta = key(eta_values)
            rows = list(groups[(state, eta)].values())
            counts = Counter(row["outcome"] for row in rows)
            label, interval = classify(counts["success"], len(rows))
            duration = np.asarray([0.05 * row["steps"] for row in rows])
            costs = np.asarray([row["J_def"] for row in rows])
            cells.append({
                "state_id": state, "eta": list(eta), "n": len(rows),
                "counts": {name: counts[name] for name in ("success", "deadlock", "timeout", "collision")},
                "classification": label, "Q_S_one_sided_95": interval,
                "mean_J_def": float(costs.mean()), "std_J_def": float(costs.std(ddof=1)),
                "mean_episode_steps": float(np.mean([row["steps"] for row in rows])),
                "mean_per_step_squared_deformation_diagnostic": float(np.mean(costs / duration)),
            })
    dump("phase1_analysis.json", {
        "cells": cells,
        "adaptive_decision": {
            "D1_pair231": "eta1 decrease reaches the SUCCESS/UNKNOWN transition; continue one midpoint step while testing whether more-negative eta2 lowers J within success.",
            "D2_pair228": "eta1 decrease and eta2 increase jointly lower J but approach rare deadlock; take one further step and controls.",
            "D4_pair227": "both eta1 decrease and eta2 increase lower J until separate success transitions; refine both and their diagonal. eta3 decrease was rejected because J increased.",
        },
    })
    points = {state: [key(point) for point in values] for state, values in phase2_points().items()}
    jobs = [
        {"state_id": state, "eta": list(eta), "seed": seed, "cell_id": "directional_phase2"}
        for state in STATES for eta in points[state] for seed in FRESH_SEEDS
    ]
    plan = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "fresh_seed_set": FRESH_SEEDS,
        "common_random_numbers": True,
        "points": {state: [list(point) for point in points[state]] for state in STATES},
        "new_rollouts": len(jobs),
        "phase1_analysis_sha256": sha(HERE / "phase1_analysis.json"),
        "protocol_sha256": sha(HERE / "protocol.json"),
        "search_expansion_limit": "This is the one additional directional step permitted by the predeclared stopping rule; no further eta expansion will follow.",
    }
    dump("phase2_plan.json", plan)
    dump("phase2_jobs.json", jobs)
    for state in STATES:
        dump(f"phase2_{state}_jobs.json", [job for job in jobs if job["state_id"] == state])
    print(json.dumps({
        "new_rollouts": len(jobs),
        "per_state": {state: sum(job["state_id"] == state for job in jobs) for state in STATES},
        "points": {state: len(points[state]) for state in STATES},
    }, indent=2))


if __name__ == "__main__":
    main()
