"""Integrity checks for valid d=2/4 initial stages; no statistics."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_stage(name: str) -> tuple[dict, list[dict], Path]:
    root = HERE / "raw" / name
    manifest = json.loads((root / "manifest.json").read_text())
    records_path = root / "records.jsonl"
    if sha(records_path) != manifest["records_sha256"]:
        raise AssertionError(f"record hash mismatch: {name}")
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    if len(rows) != manifest["records"]:
        raise AssertionError(f"manifest count mismatch: {name}")
    return manifest, rows, records_path


def main() -> None:
    plan_path = HERE / "audit_plan.json"
    plan_hash = sha(plan_path)
    plan = json.loads(plan_path.read_text())
    plan_by_id = {state["state_id"]: state for state in plan["states"]}
    seeds = {int(seed) for seed in plan["seeds_initial"]}

    nz_manifest, nz_rows, nz_path = load_stage("initial64_d24_nonzero_shard0")
    z_manifest, z_rows, z_path = load_stage("initial64_d24_zero_materialized")
    adaptive_manifest, adaptive_rows, adaptive_path = load_stage("adaptive_to128_d24_nonzero_shard0")
    adaptive_plan_path = HERE / "adaptive_to128_plan.json"
    adaptive_plan = json.loads(adaptive_plan_path.read_text())
    to256a_manifest, to256a_rows, to256a_path = load_stage("adaptive_to256_d24_nonzero_shard0")
    to256b_manifest, to256b_rows, to256b_path = load_stage("adaptive_to256_d24_nonzero_second64_shard0")
    to256_plan_path = HERE / "adaptive_to256_plan.json"
    to256_plan = json.loads(to256_plan_path.read_text())
    nonzero_ids = {sid for sid, meta in plan_by_id.items() if int(meta["y_long"]) == 1}
    zero_ids = {sid for sid, meta in plan_by_id.items() if int(meta["y_long"]) == 0}
    expected_nz = {(sid, delay, seed) for sid in nonzero_ids for delay in (2, 4) for seed in seeds}
    expected_z = {(sid, delay, seed) for sid in zero_ids for delay in (2, 4) for seed in seeds}
    key = lambda row: (row["state_id"], int(row["delay_steps"]), int(row["seed"]))
    nz_keys = {key(row) for row in nz_rows}
    z_keys = {key(row) for row in z_rows}
    adaptive_ids = set(adaptive_plan["state_ids"])
    adaptive_seeds = {int(seed) for seed in adaptive_plan["new_seeds"]}
    expected_adaptive = {
        (sid, delay, seed)
        for sid in adaptive_ids
        for delay in (2, 4)
        for seed in adaptive_seeds
    }
    adaptive_keys = {key(row) for row in adaptive_rows}
    to256_ids = set(to256_plan["state_ids"])
    to256_seeds = {int(seed) for seed in to256_plan["new_seeds"]}
    expected_to256 = {
        (sid, delay, seed)
        for sid in to256_ids
        for delay in (2, 4)
        for seed in to256_seeds
    }
    to256a_keys = {key(row) for row in to256a_rows}
    to256b_keys = {key(row) for row in to256b_rows}
    to256_keys = to256a_keys | to256b_keys
    checks = {
        "audit_plan_sha256": plan_hash,
        "nonzero_manifest_plan_hash_ok": nz_manifest["audit_plan_sha256"] == plan_hash,
        "zero_manifest_plan_hash_ok": z_manifest["audit_plan_sha256"] == plan_hash,
        "nonzero_state_count": len(nonzero_ids),
        "zero_state_count": len(zero_ids),
        "seed_count": len(seeds),
        "nonzero_record_count": len(nz_rows),
        "zero_materialized_record_count": len(z_rows),
        "nonzero_unique_key_count": len(nz_keys),
        "zero_unique_key_count": len(z_keys),
        "nonzero_exact_cartesian_inventory": nz_keys == expected_nz and len(nz_rows) == len(expected_nz),
        "zero_exact_cartesian_inventory": z_keys == expected_z and len(z_rows) == len(expected_z),
        "adaptive_manifest_plan_hash_ok": adaptive_manifest["audit_plan_sha256"] == plan_hash,
        "adaptive_selection_hash_ok": adaptive_manifest.get("adaptive_selection_sha256") == sha(adaptive_plan_path),
        "adaptive_state_count": len(adaptive_ids),
        "adaptive_seed_count": len(adaptive_seeds),
        "adaptive_record_count": len(adaptive_rows),
        "adaptive_unique_key_count": len(adaptive_keys),
        "adaptive_exact_cartesian_inventory": adaptive_keys == expected_adaptive and len(adaptive_rows) == len(expected_adaptive),
        "adaptive_execution_error_count": sum(row.get("execution_error") is not None for row in adaptive_rows),
        "adaptive_physical_rollout_count": sum(bool(row.get("physical_rollout_executed")) for row in adaptive_rows),
        "adaptive_outcomes": dict(Counter(row["outcome"] for row in adaptive_rows)),
        "adaptive_records_sha256": sha(adaptive_path),
        "to256_manifest_plan_hash_ok": all(
            manifest["audit_plan_sha256"] == plan_hash
            for manifest in (to256a_manifest, to256b_manifest)
        ),
        "to256_selection_hash_ok": all(
            manifest.get("adaptive_selection_sha256") == sha(to256_plan_path)
            for manifest in (to256a_manifest, to256b_manifest)
        ),
        "to256_state_count": len(to256_ids),
        "to256_seed_count": len(to256_seeds),
        "to256_part_record_counts": [len(to256a_rows), len(to256b_rows)],
        "to256_record_count": len(to256a_rows) + len(to256b_rows),
        "to256_unique_key_count": len(to256_keys),
        "to256_parts_overlap_key_count": len(to256a_keys & to256b_keys),
        "to256_exact_cartesian_inventory": to256_keys == expected_to256 and len(to256a_rows) + len(to256b_rows) == len(expected_to256),
        "to256_execution_error_count": sum(
            row.get("execution_error") is not None for row in to256a_rows + to256b_rows
        ),
        "to256_physical_rollout_count": sum(
            bool(row.get("physical_rollout_executed")) for row in to256a_rows + to256b_rows
        ),
        "to256_outcomes": dict(Counter(row["outcome"] for row in to256a_rows + to256b_rows)),
        "to256_part_records_sha256": [sha(to256a_path), sha(to256b_path)],
        "cross_stage_duplicate_key_count": len(nz_keys & z_keys),
        "initial_adaptive_duplicate_key_count": len((nz_keys | z_keys) & adaptive_keys),
        "prior_to256_duplicate_key_count": len((nz_keys | z_keys | adaptive_keys) & to256_keys),
        "nonzero_execution_error_count": sum(row.get("execution_error") is not None for row in nz_rows),
        "zero_execution_error_count": sum(row.get("execution_error") is not None for row in z_rows),
        "nonzero_physical_rollout_count": sum(bool(row.get("physical_rollout_executed")) for row in nz_rows),
        "zero_physical_rollout_count": sum(bool(row.get("physical_rollout_executed")) for row in z_rows),
        "zero_all_exact_materialization": all(
            row.get("reuse_kind") == "zero_eta_exact_materialization"
            and row.get("physical_rollout_executed") is False
            and [float(value) for value in row["eta_best"]] == [0.0, 0.0, 0.0]
            for row in z_rows
        ),
        "nonzero_outcomes": dict(Counter(row["outcome"] for row in nz_rows)),
        "zero_outcomes": dict(Counter(row["outcome"] for row in z_rows)),
        "nonzero_records_sha256": sha(nz_path),
        "zero_records_sha256": sha(z_path),
        "invalid_partial_directory_present": (HERE / "raw/initial64_d24_shard0/INVALID_EXCLUDED.json").exists(),
        "invalid_partial_has_manifest": (HERE / "raw/initial64_d24_shard0/manifest.json").exists(),
    }
    required = [
        checks["nonzero_manifest_plan_hash_ok"],
        checks["zero_manifest_plan_hash_ok"],
        checks["nonzero_exact_cartesian_inventory"],
        checks["zero_exact_cartesian_inventory"],
        checks["adaptive_manifest_plan_hash_ok"],
        checks["adaptive_selection_hash_ok"],
        checks["adaptive_exact_cartesian_inventory"],
        checks["adaptive_execution_error_count"] == 0,
        checks["to256_manifest_plan_hash_ok"],
        checks["to256_selection_hash_ok"],
        checks["to256_exact_cartesian_inventory"],
        checks["to256_parts_overlap_key_count"] == 0,
        checks["to256_execution_error_count"] == 0,
        checks["cross_stage_duplicate_key_count"] == 0,
        checks["initial_adaptive_duplicate_key_count"] == 0,
        checks["prior_to256_duplicate_key_count"] == 0,
        checks["nonzero_execution_error_count"] == 0,
        checks["zero_execution_error_count"] == 0,
        checks["zero_physical_rollout_count"] == 0,
        checks["zero_all_exact_materialization"],
        checks["invalid_partial_directory_present"],
        not checks["invalid_partial_has_manifest"],
    ]
    checks["all_required_checks_pass"] = all(required)
    if not checks["all_required_checks_pass"]:
        raise AssertionError(checks)
    (HERE / "d24_integrity_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
