"""Strict integrity audit for the frozen d=32/64 coarse shard."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
STAGE = HERE / "raw/coarse_long_shard0"
EXPECTED_DELAYS = {32, 64}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = HERE / "audit_plan.json"
    index_path = HERE / "tuple_index.jsonl"
    manifest_path = STAGE / "manifest.json"
    records_path = STAGE / "records.jsonl"
    spec_path = STAGE / "stage_spec.json"
    plan = json.loads(plan_path.read_text())
    by_id = {state["state_id"]: state for state in plan["states"]}
    expected = {}
    for line in index_path.read_text().splitlines():
        item = json.loads(line)
        if int(item["delay_steps"]) in EXPECTED_DELAYS:
            expected[(item["state_id"], int(item["delay_steps"]), int(item["seed"]))] = item
    manifest = json.loads(manifest_path.read_text())
    spec = json.loads(spec_path.read_text())
    rows = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    observed = {}
    duplicate_count = 0
    for row in rows:
        key = (row["state_id"], int(row["delay_steps"]), int(row["seed"]))
        duplicate_count += int(key in observed)
        observed[key] = row

    physical_mismatch = 0
    metadata_mismatch = 0
    delayed_action_mismatch = 0
    for key, row in observed.items():
        item = expected.get(key)
        if item is None:
            metadata_mismatch += 1
            continue
        should_be_physical = item["status"] == "NEW"
        physical_mismatch += int(bool(row.get("physical_rollout_executed")) != should_be_physical)
        state = by_id[key[0]]
        metadata_mismatch += int(row["category"] != state["category"])
        metadata_mismatch += int(row["source_group"] != state["source_group"])
        metadata_mismatch += int(int(row["y_long"]) != int(state["y_long"]))
        metadata_mismatch += int([float(v) for v in row["eta_best"]] != [float(v) for v in state["eta_best"]])
        first = row.get("first_step") or {}
        raw = first.get("g_raw") or []
        delayed_action_mismatch += int(not raw or max(abs(float(value)) for value in raw) > 1e-12)

    zero_ids = {state_id for state_id, state in by_id.items() if state["eta_best"] == [0.0, 0.0, 0.0]}
    zero_pair_mismatch = 0
    compare = ("outcome", "success", "steps", "terminal_step", "J_total")
    for state_id in zero_ids:
        for seed in by_id[state_id]["seeds_initial"]:
            a = observed[(state_id, 32, int(seed))]
            b = observed[(state_id, 64, int(seed))]
            zero_pair_mismatch += int(any(a[field] != b[field] for field in compare))

    expected_keys = set(expected)
    checks = {
        "audit_plan_sha256": sha(plan_path),
        "tuple_index_sha256": sha(index_path),
        "status_complete": manifest.get("status") == "complete",
        "plan_hash_matches": manifest.get("audit_plan_sha256") == sha(plan_path),
        "index_hash_matches": manifest.get("tuple_index_sha256") == sha(index_path),
        "records_hash_matches": manifest.get("records_sha256") == sha(records_path),
        "spec_hash_matches": manifest.get("stage_spec_sha256") == sha(spec_path),
        "state_count": len(by_id),
        "expected_tuple_count": len(expected_keys),
        "observed_record_count": len(rows),
        "observed_unique_tuple_count": len(observed),
        "exact_cartesian_key_match": set(observed) == expected_keys,
        "duplicate_tuple_count": duplicate_count,
        "execution_error_count": sum(row.get("execution_error") is not None for row in rows),
        "physical_execution_status_mismatch_count": physical_mismatch,
        "state_metadata_mismatch_count": metadata_mismatch,
        "delayed_first_action_nonzero_correction_count": delayed_action_mismatch,
        "zero_eta_state_count": len(zero_ids),
        "zero_eta_d32_d64_semantic_mismatch_count": zero_pair_mismatch,
        "new_physical_rollouts": sum(bool(row.get("physical_rollout_executed")) for row in rows),
        "reused_or_materialized_records": sum(not bool(row.get("physical_rollout_executed")) for row in rows),
        "outcomes": dict(Counter(row["outcome"] for row in rows)),
    }
    checks["all_required_checks_pass"] = all([
        checks["status_complete"], checks["plan_hash_matches"], checks["index_hash_matches"],
        checks["records_hash_matches"], checks["spec_hash_matches"], checks["state_count"] == 277,
        checks["expected_tuple_count"] == 35_456, checks["observed_record_count"] == 35_456,
        checks["observed_unique_tuple_count"] == 35_456, checks["exact_cartesian_key_match"],
        duplicate_count == 0, checks["execution_error_count"] == 0, physical_mismatch == 0,
        metadata_mismatch == 0, delayed_action_mismatch == 0, len(zero_ids) == 157,
        zero_pair_mismatch == 0, checks["new_physical_rollouts"] == 15_360,
        checks["reused_or_materialized_records"] == 20_096,
    ])
    if not checks["all_required_checks_pass"]:
        raise AssertionError(checks)
    output = HERE / "coarse_long_integrity.json"
    output.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

