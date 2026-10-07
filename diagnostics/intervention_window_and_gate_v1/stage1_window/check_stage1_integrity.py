"""Static/inventory integrity checks for the frozen Stage-1 infrastructure."""

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
    index_path = HERE / "tuple_index.jsonl"
    inventory = json.loads((HERE / "tuple_inventory.json").read_text())
    plan = json.loads(plan_path.read_text())
    entries = [json.loads(line) for line in index_path.read_text().splitlines()]
    keys = [(row["state_id"], int(row["delay_steps"]), int(row["seed"])) for row in entries]
    state_ids = {state["state_id"] for state in plan["states"]}
    counts = Counter((int(row["delay_steps"]), row["status"]) for row in entries)
    zero_ids = {state["state_id"] for state in plan["states"] if state["eta_best"] == [0.0, 0.0, 0.0]}
    checks = {
        "audit_plan_sha256": sha(plan_path),
        "tuple_index_sha256": sha(index_path),
        "plan_hash_matches_inventory": inventory["audit_plan_sha256"] == sha(plan_path),
        "index_hash_matches_inventory": inventory["tuple_index_sha256"] == sha(index_path),
        "state_count": len(plan["states"]),
        "source_group_count": len({state["source_group"] for state in plan["states"]}),
        "zero_eta_state_count": len(zero_ids),
        "nonzero_eta_state_count": len(plan["states"]) - len(zero_ids),
        "hard13_count": sum(bool(state["is_hard13_anchor"]) for state in plan["states"]),
        "every_state_has_64_unique_seeds": all(len(state["seeds_initial"]) == len(set(state["seeds_initial"])) == 64 for state in plan["states"]),
        "index_record_count": len(entries),
        "index_unique_key_count": len(set(keys)),
        "index_exact_cartesian": len(entries) == len(set(keys)) == len(plan["states"]) * 64 * len(plan["coarse_delays"]),
        "index_state_ids_exact": {row["state_id"] for row in entries} == state_ids,
        "index_delays_exact": {int(row["delay_steps"]) for row in entries} == set(plan["coarse_delays"]),
        "zero_eta_new_tuple_count": sum(row["status"] == "NEW" and row["state_id"] in zero_ids for row in entries),
        "source_paths_exist": all(row["status"] != "REUSE" or Path(row["source_path"]).exists() for row in entries),
        "by_delay": {
            str(delay): {status.lower(): counts[(delay, status)] for status in ("REUSE", "NEW")}
            for delay in plan["coarse_delays"]
        },
        "checkpoint_hash_ok": sha(Path(plan["checkpoint"])) == plan["checkpoint_sha256"],
        "fixed_eta_downstream_explicit": plan["semantics"]["receding_oracle_requery_available"] is False,
        "selection_is_full_stable_pool": plan["selection"]["stable_state_count"] == 277,
    }
    required = [
        checks["plan_hash_matches_inventory"], checks["index_hash_matches_inventory"],
        checks["state_count"] == 277, checks["source_group_count"] == 117,
        checks["zero_eta_state_count"] == 157, checks["nonzero_eta_state_count"] == 120,
        checks["hard13_count"] == 13, checks["every_state_has_64_unique_seeds"],
        checks["index_exact_cartesian"], checks["index_state_ids_exact"], checks["index_delays_exact"],
        checks["zero_eta_new_tuple_count"] == 0, checks["source_paths_exist"],
        checks["checkpoint_hash_ok"], checks["fixed_eta_downstream_explicit"],
        checks["selection_is_full_stable_pool"],
    ]
    checks["all_required_checks_pass"] = all(required)
    if not checks["all_required_checks_pass"]:
        raise AssertionError(checks)
    (HERE / "infrastructure_integrity.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    print(json.dumps(checks, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
