"""Verify a completed resumable delay stage against its frozen invocation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--expected-delays", required=True)
    parser.add_argument("--expected-new", type=int)
    args = parser.parse_args()
    expected_delays = {int(value) for value in args.expected_delays.split(",")}
    output = HERE / "raw" / f"{args.stage}_shard{args.shard}"
    manifest_path, records_path, spec_path = output / "manifest.json", output / "records.jsonl", output / "stage_spec.json"
    manifest = json.loads(manifest_path.read_text())
    spec = json.loads(spec_path.read_text())
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    keys = [(row["state_id"], int(row["delay_steps"]), int(row["seed"])) for row in rows]
    checks = {
        "status_complete": manifest.get("status") == "complete",
        "manifest_records_hash_ok": manifest.get("records_sha256") == sha(records_path),
        "manifest_spec_hash_ok": manifest.get("stage_spec_sha256") == sha(spec_path),
        "plan_hash_ok": manifest.get("audit_plan_sha256") == sha(HERE / "audit_plan.json"),
        "index_hash_ok": manifest.get("tuple_index_sha256") == sha(HERE / "tuple_index.jsonl"),
        "record_count": len(rows),
        "record_count_matches": len(rows) == int(manifest["records"]) == int(spec["tuples"]),
        "unique_tuple_count": len(set(keys)),
        "no_duplicate_tuples": len(keys) == len(set(keys)),
        "delays_exact": {int(row["delay_steps"]) for row in rows} == expected_delays,
        "execution_error_count": sum(row.get("execution_error") is not None for row in rows),
        "new_physical_rollouts": sum(bool(row.get("physical_rollout_executed")) for row in rows),
        "manifest_new_physical_matches": sum(bool(row.get("physical_rollout_executed")) for row in rows) == int(manifest["new_physical_rollouts"]),
        "outcomes": dict(Counter(row["outcome"] for row in rows)),
        "all_delayed_new_first_actions_are_zero_correction": all(
            int(row["delay_steps"]) == 0
            or not row.get("physical_rollout_executed")
            or max(abs(float(value)) for value in row["first_step"]["g_raw"]) <= 1e-12
            for row in rows
        ),
    }
    required = [
        checks["status_complete"], checks["manifest_records_hash_ok"], checks["manifest_spec_hash_ok"],
        checks["plan_hash_ok"], checks["index_hash_ok"], checks["record_count_matches"],
        checks["no_duplicate_tuples"], checks["delays_exact"], checks["execution_error_count"] == 0,
        checks["manifest_new_physical_matches"], checks["all_delayed_new_first_actions_are_zero_correction"],
    ]
    if args.expected_new is not None:
        checks["expected_new_matches"] = checks["new_physical_rollouts"] == args.expected_new
        required.append(checks["expected_new_matches"])
    checks["all_required_checks_pass"] = all(required)
    if not checks["all_required_checks_pass"]:
        raise AssertionError(checks)
    output_path = HERE / f"{args.stage}_shard{args.shard}_integrity.json"
    output_path.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
