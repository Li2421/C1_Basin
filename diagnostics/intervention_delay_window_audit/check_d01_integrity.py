"""Integrity checks for the completed d=0/1 frozen rollout stage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = HERE / "audit_plan.json"
    plan_hash = sha(plan_path)
    plan = json.loads(plan_path.read_text())
    stage = HERE / "raw/initial64_d01_shard0"
    manifest = json.loads((stage / "manifest.json").read_text())
    records_path = stage / "records.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    expected = len(plan["states"]) * len(plan["seeds_initial"]) * 2
    keys = {(row["state_id"], int(row["delay_steps"]), int(row["seed"])) for row in records}
    by_key = {(row["state_id"], int(row["delay_steps"]), int(row["seed"])): row for row in records}
    assert manifest["audit_plan_sha256"] == plan_hash
    assert manifest["records_sha256"] == sha(records_path)
    assert len(records) == expected == 7552 and len(keys) == expected
    assert {row["delay_steps"] for row in records} == {0, 1}
    assert {row["seed"] for row in records} == set(plan["seeds_initial"])
    assert not any(row.get("execution_error") is not None for row in records)

    matched_first_flow_safe = 0
    zero_eta_exact = 0
    d1_exact_safe = 0
    for state in plan["states"]:
        state_id = state["state_id"]
        for seed in plan["seeds_initial"]:
            d0, d1 = by_key[(state_id, 0, seed)], by_key[(state_id, 1, seed)]
            f0, f1 = d0["first_step"], d1["first_step"]
            assert np.array_equal(np.asarray(f0["u_flow"]), np.asarray(f1["u_flow"]))
            assert np.array_equal(np.asarray(f0["u_safe"]), np.asarray(f1["u_safe"]))
            matched_first_flow_safe += 1
            assert np.max(np.abs(np.asarray(f1["u_exec"]) - np.asarray(f1["u_safe"]))) < 1e-12
            assert np.max(np.abs(np.asarray(f1["g_raw"]))) < 1e-12
            d1_exact_safe += 1
            if state["eta_best"] == [0.0, 0.0, 0.0]:
                for field in ("outcome", "success", "steps", "terminal_step", "J_total"):
                    assert d0[field] == d1[field], (state_id, seed, field)
                zero_eta_exact += 1
    result = {
        "passed": True,
        "audit_plan_sha256": plan_hash,
        "records_sha256": sha(records_path),
        "state_count": len(plan["states"]),
        "seed_count": len(plan["seeds_initial"]),
        "logical_records": len(records),
        "unique_logical_tuple_keys": len(keys),
        "new_physical_rollouts": manifest["new_rollouts"],
        "reused_or_exactly_materialized_records": manifest["reused_rollouts"],
        "execution_errors": 0,
        "matched_first_step_flow_and_safe_pairs": matched_first_flow_safe,
        "d1_exact_u_safe_first_steps": d1_exact_safe,
        "zero_eta_exact_d0_d1_pairs": zero_eta_exact,
        "downstream_semantics": plan["semantics"]["downstream_policy"],
        "receding_oracle_requery_available": False,
    }
    (HERE / "d01_integrity_checks.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
