"""Freeze the original finite full-horizon oracle search for zero-insufficient fresh32 states."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/direct_eta_basin_geometry_audit_v1"
RAW = HERE / "raw/fresh32_zero_prediction.jsonl"

CANDIDATES = (
    (0.5703125, -0.375, -0.125),
    (0.578125, -0.40625, -0.125),
    (0.578125, -0.34375, -0.125),
    (0.40625, -0.5, 0.0),
    (0.4375, -0.53125, 0.0),
    (0.375, -0.4375, -0.0625),
    (0.40625, -0.4375, -0.0625),
    (1.0, 0.0, 0.25),
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_sha(value: dict) -> str:
    body = {key: val for key, val in value.items() if key != "content_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    destination = HERE / "fresh32_oracle_plan.json"
    if destination.exists():
        raise RuntimeError("refusing to overwrite frozen plan")
    source_plan = json.loads((HERE / "fresh32_zero_prediction_plan.json").read_text())
    rows = [json.loads(line) for line in RAW.read_text().splitlines()]
    if len(rows) != 4096:
        raise RuntimeError(f"ZERO/prediction stage incomplete: {len(rows)}/4096")
    zero = defaultdict(list)
    for row in rows:
        if row["condition"] == "ZERO":
            zero[int(row["episode_index"])].append(row)
    if set(zero) != {int(row["episode_index"]) for row in source_plan["selected_episodes"]}:
        raise RuntimeError("ZERO episode coverage mismatch")
    zero_counts = {
        episode: sum(bool(row["success"]) for row in values)
        for episode, values in zero.items()
    }
    if any(len(values) != 64 for values in zero.values()):
        raise RuntimeError("ZERO condition is not complete matched64")
    needed = {episode for episode, count in zero_counts.items() if count < 63}
    selected = {
        int(row["episode_index"]): row
        for row in source_plan["selected_episodes"]
        if int(row["episode_index"]) in needed
    }
    seeds = [int(seed) for seed in source_plan["robust_seeds"]]
    arms = []
    for episode_index in sorted(selected):
        episode = selected[episode_index]
        for candidate_index, eta in enumerate(CANDIDATES):
            arms.append(
                {
                    "arm_id": f"E{episode_index:03d}__C{candidate_index}",
                    "episode_index": episode_index,
                    "source_id": episode["source_id"],
                    "initial_positions": episode["initial_positions"],
                    "rng_namespace": 500000 + episode_index,
                    "candidate_priority": candidate_index,
                    "eta": list(eta),
                    "seeds": seeds,
                }
            )
    plan = {
        "schema": "fresh32_original_finite_oracle_plan_v1",
        "status": "FROZEN_BEFORE_ORACLE_SEARCH",
        "oracle_interpretation": "original general-WIDE finite eta domain: zero followed by exact frozen eight active candidates",
        "success_rule": "B63 iff complete 64 matched futures and >=63 successes; exact stop after second physical failure",
        "selection_rule": "active candidates evaluated only when already-frozen ZERO matched64 result is not B63",
        "zero_results_sha256": sha(RAW),
        "zero_success_counts": {str(key): val for key, val in sorted(zero_counts.items())},
        "zero_insufficient_episode_indices": sorted(needed),
        "source_plan_sha256": sha(HERE / "fresh32_zero_prediction_plan.json"),
        "domain_resolution_sha256": sha(HERE / "oracle_domain_resolution.md"),
        "candidate_etas": [list(value) for value in CANDIDATES],
        "arms": arms,
        "maximum_candidate_continuations": len(arms) * 64,
        "global_new_rollout_cap_for_this_stage": 7528,
        "prior_committed_new_rollouts": 7472,
        "overall_new_rollout_cap": 15000,
    }
    plan["content_sha256"] = semantic_sha(plan)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(
        json.dumps(
            {
                "zero_B63": 32 - len(needed),
                "zero_insufficient": len(needed),
                "active_candidate_arms": len(arms),
                "maximum_candidate_continuations": len(arms) * 64,
                "stage_cap": plan["global_new_rollout_cap_for_this_stage"],
                "plan_sha256": sha(destination),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
