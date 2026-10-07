"""Freeze fresh-seed validation of refined winners and nearest competitors."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from .setup import HERE, sha, write


POLICIES = {
    "D1_pair231": [
        [0.59375, -0.3125, -0.125],
        [0.59375, -0.3125, -0.1875],
        [0.625, -0.375, -0.125],
        [0.75, 0.0, 0.0],
    ],
    "D2_pair228": [
        [0.5, -0.5625, -0.0625],
        [0.4375, -0.5625, 0.0],
        [0.4375, -0.625, 0.0],
        [0.5, -0.5, 0.0],
    ],
    "D4_pair227": [
        [0.40625, -0.4375, -0.0625],
        [0.375, -0.4375, -0.0625],
        [0.375, -0.4375, 0.0],
        [0.5, -0.5, 0.0],
    ],
}


def main():
    seeds = list(range(95105001, 95105017))
    jobs = []
    for state_id, policies in POLICIES.items():
        for index, eta in enumerate(policies):
            for seed in seeds:
                jobs.append({
                    "state_id": state_id, "eta": eta, "seed": seed,
                    "cell_id": f"validation_candidate_{index}",
                    "candidate_index": index,
                })
    plan = {
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "Phase-2 refined winner, two closest mean-J successful competitors, and the pre-refinement winner for each state.",
        "policies": POLICIES,
        "fresh_seeds": seeds,
        "common_random_numbers_within_state": True,
        "new_rollouts": len(jobs),
        "phase2_plan_sha256": sha(HERE / "phase2_plan.json"),
        "search_expansion": False,
    }
    write("validation_plan.json", plan)
    write("validation_jobs.json", jobs)
    for state_id in POLICIES:
        write(
            f"validation_{state_id}_jobs.json",
            [job for job in jobs if job["state_id"] == state_id],
        )
    print(json.dumps({"new_rollouts": len(jobs), "per_state": 64}, indent=2))


if __name__ == "__main__":
    main()
