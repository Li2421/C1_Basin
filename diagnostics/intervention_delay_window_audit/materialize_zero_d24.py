"""Materialize eta=0 d=2/4 records from canonical completed d=0 records.

For eta_best=(0,0,0), every tested delay executes the identical action
sequence.  This utility performs no simulation and refuses incomplete or
hash-mismatched source data.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "raw/initial64_d01_shard0"
OUTPUT = HERE / "raw/initial64_d24_zero_materialized"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    plan_path = HERE / "audit_plan.json"
    plan_hash = sha(plan_path)
    plan = json.loads(plan_path.read_text())
    source_manifest_path = SOURCE / "manifest.json"
    source_records_path = SOURCE / "records.jsonl"
    if not source_manifest_path.exists():
        raise RuntimeError("canonical d=0/1 manifest is not complete yet")
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("audit_plan_sha256") != plan_hash:
        raise AssertionError("canonical source plan hash mismatch")
    if sha(source_records_path) != source_manifest.get("records_sha256"):
        raise AssertionError("canonical source record hash mismatch")
    if source_manifest.get("delays") != [0, 1]:
        raise AssertionError("canonical source does not contain d=0/1")

    zero_states = {
        state["state_id"]: state
        for state in plan["states"]
        if int(state["y_long"]) == 0
    }
    seeds = {int(seed) for seed in plan["seeds_initial"]}
    canonical = {}
    for line in source_records_path.read_text().splitlines():
        row = json.loads(line)
        if row["state_id"] not in zero_states or int(row["delay_steps"]) != 0:
            continue
        key = (row["state_id"], int(row["seed"]))
        if key in canonical:
            raise AssertionError(f"duplicate canonical tuple: {key}")
        if row.get("execution_error") is not None:
            raise AssertionError(f"canonical execution error: {key}")
        if [float(value) for value in row["eta_best"]] != [0.0, 0.0, 0.0]:
            raise AssertionError(f"nonzero eta in zero cohort: {key}")
        canonical[key] = row

    expected_keys = {(state_id, seed) for state_id in zero_states for seed in seeds}
    if set(canonical) != expected_keys:
        missing = sorted(expected_keys - set(canonical))[:5]
        extra = sorted(set(canonical) - expected_keys)[:5]
        raise AssertionError(f"canonical inventory mismatch; missing={missing}, extra={extra}")

    rows = []
    for state_id in zero_states:
        for seed in sorted(seeds):
            source = canonical[(state_id, seed)]
            for delay in (2, 4):
                row = dict(source)
                row.update({
                    "delay_steps": delay,
                    "reused": True,
                    "reuse_kind": "zero_eta_exact_materialization",
                    "reuse_source": str(source_records_path),
                    "physical_rollout_executed": False,
                    "materialized_from_delay": 0,
                    "J_delay_prefix": 0.0,
                    "J_after_switch": float(source["J_total"]),
                })
                rows.append(row)

    keys = {(row["state_id"], int(row["delay_steps"]), int(row["seed"])) for row in rows}
    expected_count = len(zero_states) * len(seeds) * 2
    if len(rows) != expected_count or len(keys) != expected_count:
        raise AssertionError((len(rows), len(keys), expected_count))
    OUTPUT.mkdir(parents=True, exist_ok=False)
    records_path = OUTPUT / "records.jsonl"
    records_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    manifest = {
        "stage": "initial64_d24_zero_materialized",
        "status": "VALID_EXACT_MATERIALIZATION",
        "audit_plan_sha256": plan_hash,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha(source_manifest_path),
        "source_records_sha256": sha(source_records_path),
        "delays": [2, 4],
        "state_ids": list(zero_states),
        "state_count": len(zero_states),
        "seed_count": len(seeds),
        "records": len(rows),
        "new_rollouts": 0,
        "reused_rollouts": len(rows),
        "physical_steps_new": 0,
        "physical_rollout_executed": False,
        "reuse_kinds": dict(Counter(row["reuse_kind"] for row in rows)),
        "records_sha256": sha(records_path),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
