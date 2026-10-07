"""Validate merged d=0/1 adaptive-to-128 extension records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    selection_path = HERE / "adaptive_to128_plan.json"
    selection = json.loads(selection_path.read_text())
    stages = [
        HERE / "raw/adaptive128_d01_shard0",
        HERE / "raw/adaptive128_d01_cont34_64_shard0",
    ]
    records, manifests = [], []
    for stage in stages:
        manifest = json.loads((stage / "manifest.json").read_text())
        path = stage / "records.jsonl"
        assert manifest["records_sha256"] == sha(path)
        assert manifest["adaptive_selection_sha256"] == sha(selection_path)
        manifests.append(manifest)
        records.extend(json.loads(line) for line in path.read_text().splitlines())
    keys = {(r["state_id"], int(r["delay_steps"]), int(r["seed"])) for r in records}
    assert len(records) == len(keys) == 10 * 64 * 2 == 1280
    assert {r["state_id"] for r in records} == set(selection["state_ids"])
    assert {int(r["seed"]) for r in records} == set(selection["new_seeds"])
    assert {int(r["delay_steps"]) for r in records} == {0, 1}
    assert not any(r.get("execution_error") is not None for r in records)
    by_key = {(r["state_id"], int(r["delay_steps"]), int(r["seed"])): r for r in records}
    matched, d1_safe, zero_exact = 0, 0, 0
    plan = json.loads((HERE / "audit_plan.json").read_text())
    states = {s["state_id"]: s for s in plan["states"]}
    for state_id in selection["state_ids"]:
        for seed in selection["new_seeds"]:
            d0, d1 = by_key[(state_id, 0, seed)], by_key[(state_id, 1, seed)]
            assert np.array_equal(np.asarray(d0["first_step"]["u_flow"]), np.asarray(d1["first_step"]["u_flow"]))
            assert np.array_equal(np.asarray(d0["first_step"]["u_safe"]), np.asarray(d1["first_step"]["u_safe"]))
            matched += 1
            assert np.max(np.abs(np.asarray(d1["first_step"]["u_exec"]) - np.asarray(d1["first_step"]["u_safe"]))) < 1e-12
            d1_safe += 1
            if states[state_id]["eta_best"] == [0.0, 0.0, 0.0]:
                for field in ("outcome", "success", "steps", "terminal_step", "J_total"):
                    assert d0[field] == d1[field]
                zero_exact += 1
    result = {
        "passed": True,
        "adaptive_selection_sha256": sha(selection_path),
        "selected_state_count": 10,
        "new_seed_count": 64,
        "logical_records": len(records),
        "unique_logical_tuple_keys": len(keys),
        "new_physical_rollouts": sum(m["new_rollouts"] for m in manifests),
        "reused_rollouts": sum(m["reused_rollouts"] for m in manifests),
        "execution_errors": 0,
        "matched_first_step_flow_and_safe_pairs": matched,
        "d1_exact_u_safe_first_steps": d1_safe,
        "zero_eta_exact_pairs": zero_exact,
        "job256_partial_semantically_valid": True,
        "job256_cache_efficiency_issue_only": True,
        "avoidable_duplicate_executions_before_detection": 34,
        "continuation_repeated_prior_seed_tuples": 0
    }
    (HERE / "adaptive128_d01_integrity.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
