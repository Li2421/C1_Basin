"""Freeze inputs and validate reusable H1/L1 oracle-cadence results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/home/zhihan/research/Basin_C1")
HERE = ROOT / "diagnostics/strict_deadlock_oracle_burst_length_v1"
CAPACITY = ROOT / "diagnostics/strict_deadlock_success_basin_capacity_v1"
CADENCE = ROOT / "diagnostics/strict_deadlock_oracle_cadence_v1"
ROBUST_SEEDS = list(range(95310001, 95310065))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def eta_key(value: object) -> tuple[float, float, float]:
    if isinstance(value, str):
        value = json.loads(value)
    return tuple(round(float(item), 8) for item in value)  # type: ignore[arg-type]


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def main() -> None:
    cadence_manifest = json.loads((CADENCE / "manifest.json").read_text())
    cadence_integrity = json.loads((CADENCE / "integrity_audit.json").read_text())
    if cadence_manifest.get("status") != "COMPLETE" or cadence_integrity.get("status") != "PASS":
        raise RuntimeError("source cadence audit is not complete/pass")
    if cadence_manifest.get("classification") != "ORACLE_CADENCE_MISMATCH_CONFIRMED":
        raise RuntimeError("unexpected source cadence classification")
    frozen = json.loads((CADENCE / "source_state_manifest.json").read_text())
    etas = json.loads((CADENCE / "eta_manifest.json").read_text())
    if frozen["count"] != 17 or len(etas["etas"]) != 17:
        raise RuntimeError("expected 17 frozen states and etas")
    eta_by_state = {row["state_id"]: eta_key(row["eta"]) for row in etas["etas"]}
    h1_reuse = {row["state_id"]: row for row in cadence_integrity["H1_robust_reuse"]}
    reuse = []
    for state in frozen["states"]:
        state_id = state["state_id"]
        state_path = Path(state["state_file"])
        if sha256(state_path) != state["state_sha256"]:
            raise RuntimeError(f"state changed: {state_id}")
        source_h1 = Path(h1_reuse[state_id]["robust_path"])
        if sha256(source_h1) != h1_reuse[state_id]["robust_path_sha256"]:
            raise RuntimeError(f"H1 source changed: {state_id}")
        h1_rows = [
            row for row in rows(source_h1)
            if not row.get("state_complete") and eta_key(row["eta"]) == eta_by_state[state_id]
        ]
        h1_rows.sort(key=lambda row: int(row["seed"]))
        l1_path = CADENCE / "raw" / f"{state_id}.jsonl"
        expected_l1_hash = cadence_manifest["raw_state_files"][l1_path.name]
        if sha256(l1_path) != expected_l1_hash:
            raise RuntimeError(f"L1 source changed: {state_id}")
        l1_all = rows(l1_path)
        l1_rows = sorted(
            [row for row in l1_all if row["condition"] == "H8" and row["flow_mode"] == "robust"],
            key=lambda row: int(row["seed"]),
        )
        exact = {(row["condition"], row["flow_mode"]): row for row in l1_all[:2]}
        if [int(row["seed"]) for row in h1_rows] != ROBUST_SEEDS:
            raise RuntimeError(f"H1 seed mismatch: {state_id}")
        if [int(row["seed"]) for row in l1_rows] != ROBUST_SEEDS:
            raise RuntimeError(f"L1 seed mismatch: {state_id}")
        if sum(row["outcome"] == "success" for row in h1_rows) != 64:
            raise RuntimeError(f"H1 no longer 64/64: {state_id}")
        if exact[("H1", "exact")]["outcome"] != "success":
            raise RuntimeError(f"exact H1 no longer success: {state_id}")
        reuse.append({
            "case_id": state["case_id"], "state_id": state_id,
            "eta": list(eta_by_state[state_id]),
            "H1_robust_path": str(source_h1), "H1_robust_sha256": sha256(source_h1),
            "L1_raw_path": str(l1_path), "L1_raw_sha256": sha256(l1_path),
            "H1_robust_successes": 64,
            "L1_robust_successes": sum(row["outcome"] == "success" for row in l1_rows),
            "H1_exact": exact[("H1", "exact")], "L1_exact": exact[("H8", "exact")],
        })

    source_manifest = {
        "schema": "strict_deadlock_oracle_burst_source_v1",
        "capacity_directory": str(CAPACITY), "cadence_directory": str(CADENCE),
        "capacity_manifest_sha256": sha256(CAPACITY / "manifest.json"),
        "cadence_manifest_sha256": sha256(CADENCE / "manifest.json"),
        "count": 17, "historical": 11, "fresh_unseen": 6,
        "starting_rule": "earliest robust state from capacity audit",
        "states": frozen["states"], "etas": etas["etas"],
        "reuse": reuse,
    }
    burst_config = {
        "schema": "strict_deadlock_oracle_burst_config_v1",
        "dense_reference": "H1", "trigger_period": 8,
        "tested_burst_lengths": [1, 2, 4, 6, 8],
        "new_burst_lengths": [2, 4, 6, 8],
        "trigger_rule": "absolute global timestep modulo 8 equals zero",
        "burst_rule": "a trigger observed at or after query start activates the current and next L-1 physical steps; no artificial trigger at query start",
        "initial_burst_latch": 0,
        "special_phase_case": {
            "state_id": "old_r106__S_8s", "query_step": 279, "query_phase_mod8": 7,
            "semantics": "inactive at query step 279; first trigger at global step 280, including for L8",
        },
        "robust_seeds": ROBUST_SEEDS,
        "continuation_horizon": "remaining global horizon through absolute step 850",
        "dt": 0.05, "eta_search": False,
        "new_rollouts_expected": 17 * (4 + 4 * 64),
        "reused_rollouts": {"H1_robust": 1088, "L1_robust": 1088, "H1_exact": 17, "L1_exact": 17},
    }
    integrity = {
        "status": "PREPARED_REUSE_GATE_PASS", "states": 17,
        "state_hashes_pass": True, "eta_hashes_match": True,
        "H1_reuse_valid": True, "L1_reuse_valid": True,
        "H1_robust_reused": 1088, "L1_robust_reused": 1088,
        "H1_all_64_of_64": True,
        "eta_search_performed": False, "Gphi_loaded_or_evaluated": False,
        "source_code_paths": cadence_integrity["source_files"],
        "source_code_sha256": cadence_integrity["source_sha256"],
        "source_cadence_integrity_sha256": sha256(CADENCE / "integrity_audit.json"),
    }
    write_json(HERE / "source_manifest.json", source_manifest)
    write_json(HERE / "burst_config.json", burst_config)
    write_json(HERE / "integrity_audit.json", integrity)
    print(json.dumps({
        "status": "PASS", "states": 17, "new_rollouts_expected": burst_config["new_rollouts_expected"],
        "reused_H1_L1_robust": 2176,
    }, indent=2))


if __name__ == "__main__":
    main()
