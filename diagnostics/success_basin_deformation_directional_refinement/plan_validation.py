"""Freeze fresh matched-seed validation of only statistically close minima."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .setup import HERE, STATES, dump, sha


SEEDS = list(range(95107001, 95107033))
CANDIDATES = {
    "D1_pair231": [
        (0.5703125, -0.34375, -0.125),
        (0.5703125, -0.375, -0.125),
        (0.578125, -0.40625, -0.125),
    ],
    "D2_pair228": [
        (0.359375, -0.5, 0.0),
        (0.375, -0.46875, 0.0),
        (0.375, -0.5, 0.0),
    ],
    "D4_pair227": [
        (0.375, -0.421875, -0.0625),
        (0.359375, -0.4375, -0.0625),
        (0.3671875, -0.4375, -0.0625),
    ],
}


def main():
    jobs = [
        {"state_id": state, "eta": list(eta), "seed": seed, "cell_id": "candidate_validation"}
        for state in STATES for eta in CANDIDATES[state] for seed in SEEDS
    ]
    plan = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "Only the three statistically close Phase-2 minimum candidates per state; no eta expansion.",
        "candidates": {state: [list(eta) for eta in CANDIDATES[state]] for state in STATES},
        "fresh_seeds": SEEDS,
        "common_random_numbers": True,
        "new_rollouts": len(jobs),
        "phase2_plan_sha256": sha(HERE / "phase2_plan.json"),
        "search_expansion": False,
    }
    dump("validation_plan.json", plan)
    dump("validation_jobs.json", jobs)
    for state in STATES:
        dump(f"validation_{state}_jobs.json", [job for job in jobs if job["state_id"] == state])
    print(json.dumps({"new_rollouts": len(jobs), "per_state": len(jobs) // 3}, indent=2))


if __name__ == "__main__":
    main()
