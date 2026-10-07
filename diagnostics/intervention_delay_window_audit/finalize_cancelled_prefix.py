"""Validate and manifest job256's complete 32-seed prefix after safe cancel."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = HERE / "audit_plan.json"
    selection_path = HERE / "adaptive_to128_plan.json"
    stage = HERE / "raw/adaptive128_d01_shard0"
    records_path = stage / "records.jsonl"
    plan = json.loads(plan_path.read_text())
    selection = json.loads(selection_path.read_text())
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    states = set(selection["state_ids"])
    observed_seeds = sorted({int(r["seed"]) for r in records})
    prefix_count = len(observed_seeds)
    seeds = set(selection["new_seeds"][:prefix_count])
    keys = {(r["state_id"], int(r["delay_steps"]), int(r["seed"])) for r in records}
    assert observed_seeds == selection["new_seeds"][:prefix_count]
    assert len(records) == len(keys) == 10 * prefix_count * 2
    assert {r["state_id"] for r in records} == states
    assert {int(r["seed"]) for r in records} == seeds
    assert {int(r["delay_steps"]) for r in records} == {0, 1}
    assert not any(r.get("execution_error") is not None for r in records)
    manifest = {
        "stage": f"adaptive128_d01_valid_prefix{prefix_count}",
        "partial_but_complete_seed_prefix": True,
        "source_job": 256,
        "source_job_final_state": f"CANCELLED_BY_AUDIT_AGENT_AFTER_{prefix_count}_COMPLETE_SEED_ROUNDS",
        "cancel_reason": "cache audit found pre-existing conceptual branch-I d0 tuples were not reused for one state; numerical records remain valid; cancel prevented further unnecessary executions",
        "semantic_validity_impact": "none",
        "cache_efficiency_impact": f"{prefix_count} avoidable d0 executions for p018 occurred before detection",
        "delays": [0, 1],
        "state_ids": sorted(states),
        "seed_count": prefix_count,
        "seed_start_index": 0,
        "seed_stop_index_exclusive": prefix_count,
        "records": len(records),
        "new_rollouts": sum(bool(r.get("physical_rollout_executed", not r.get("reused"))) for r in records),
        "reused_rollouts": sum(not bool(r.get("physical_rollout_executed", not r.get("reused"))) for r in records),
        "reuse_kinds": dict(Counter(r.get("reuse_kind") for r in records if r.get("reused"))),
        "physical_steps_new": sum(int(r["steps"]) for r in records if r.get("physical_rollout_executed", not r.get("reused"))),
        "elapsed_s_to_last_complete_round_upper_bound": 197.0,
        "device": ["cuda:0"],
        "audit_plan_sha256": sha(plan_path),
        "adaptive_selection_sha256": sha(selection_path),
        "adaptive_selection_rule": selection["selection_rule"],
        "records_sha256": sha(records_path),
    }
    assert manifest["audit_plan_sha256"] == selection["audit_plan_sha256"]
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
