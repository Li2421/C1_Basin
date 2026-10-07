"""Verify every frozen adaptive job and its provenance chain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    job_manifest_path = HERE / "adaptive_job_manifest.json"
    selection_path = HERE / "adaptive_selection.json"
    plan_path = HERE / "audit_plan.json"
    index_path = HERE / "tuple_index.jsonl"
    jobs = json.loads(job_manifest_path.read_text())
    parent_hash = sha(selection_path)
    checks = {
        "job_manifest_sha256": sha(job_manifest_path),
        "adaptive_selection_sha256": parent_hash,
        "parent_selection_hash_matches": jobs["adaptive_selection_sha256"] == parent_hash,
        "audit_plan_hash_matches": jobs["audit_plan_sha256"] == sha(plan_path),
        "tuple_index_hash_matches": jobs["tuple_index_sha256"] == sha(index_path),
        "job_count": len(jobs["jobs"]),
        "completed_job_count": 0,
        "total_records": 0,
        "total_new_physical_rollouts": 0,
        "execution_error_count": 0,
        "duplicate_adaptive_tuple_count": 0,
        "job_results": [],
    }
    all_keys = set()
    for job in jobs["jobs"]:
        derived = Path(job["selection_path"])
        payload = json.loads(derived.read_text())
        output = HERE / "raw" / f"{job['stage']}_shard0"
        manifest_path = output / "manifest.json"
        records_path = output / "records.jsonl"
        manifest = json.loads(manifest_path.read_text())
        rows = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
        keys = {(row["state_id"], int(row["delay_steps"]), int(row["seed"])) for row in rows}
        duplicates = len(rows) - len(keys)
        cross_duplicates = len(all_keys & keys)
        all_keys.update(keys)
        result = {
            "stage": job["stage"],
            "selection_hash_matches": sha(derived) == job["selection_sha256"] == manifest.get("selection_sha256"),
            "parent_selection_hash_matches": payload.get("parent_adaptive_selection_sha256") == parent_hash,
            "plan_hash_matches": manifest.get("audit_plan_sha256") == sha(plan_path),
            "index_hash_matches": manifest.get("tuple_index_sha256") == sha(index_path),
            "status_complete": manifest.get("status") == "complete",
            "records_hash_matches": manifest.get("records_sha256") == sha(records_path),
            "record_count": len(rows),
            "record_count_matches": len(rows) == job["expected_records"] == int(manifest["records"]),
            "new_physical_rollouts": sum(bool(row.get("physical_rollout_executed")) for row in rows),
            "new_physical_matches": sum(bool(row.get("physical_rollout_executed")) for row in rows) == job["expected_new_physical_rollouts"],
            "delays_match": {int(row["delay_steps"]) for row in rows} == set(job["delays"]),
            "duplicate_tuple_count": duplicates,
            "cross_job_duplicate_tuple_count": cross_duplicates,
            "execution_error_count": sum(row.get("execution_error") is not None for row in rows),
            "delayed_new_first_actions_zero_correction": all(
                int(row["delay_steps"]) == 0
                or not row.get("physical_rollout_executed")
                or max(abs(float(value)) for value in row["first_step"]["g_raw"]) <= 1e-12
                for row in rows
            ),
        }
        result["pass"] = all([
            result["selection_hash_matches"], result["parent_selection_hash_matches"],
            result["plan_hash_matches"], result["index_hash_matches"], result["status_complete"],
            result["records_hash_matches"], result["record_count_matches"],
            result["new_physical_matches"], result["delays_match"],
            duplicates == 0, cross_duplicates == 0, result["execution_error_count"] == 0,
            result["delayed_new_first_actions_zero_correction"],
        ])
        checks["job_results"].append(result)
        checks["completed_job_count"] += int(result["status_complete"])
        checks["total_records"] += len(rows)
        checks["total_new_physical_rollouts"] += result["new_physical_rollouts"]
        checks["execution_error_count"] += result["execution_error_count"]
        checks["duplicate_adaptive_tuple_count"] += duplicates + cross_duplicates
    checks["all_required_checks_pass"] = all([
        checks["parent_selection_hash_matches"], checks["audit_plan_hash_matches"],
        checks["tuple_index_hash_matches"],
        checks["completed_job_count"] == checks["job_count"],
        checks["execution_error_count"] == 0,
        checks["duplicate_adaptive_tuple_count"] == 0,
        all(result["pass"] for result in checks["job_results"]),
    ])
    if not checks["all_required_checks_pass"]:
        raise AssertionError(checks)
    output = HERE / "adaptive_execution_integrity.json"
    output.write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

