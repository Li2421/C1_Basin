"""Freeze Stage-1's complete oracle-stable state pool and matched seed sets."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/home/zhihan/research/Basin_C1")
DIAG = ROOT / "diagnostics"
HERE = Path(__file__).resolve().parent
CONF = DIAG / "gphi_gate_confidence_aware_v1/oracle_confidence_dataset.csv"
HARD = DIAG / "hard_stable_boundary_crossval/difficult_stable_states.csv"
V4 = DIAG / "gphi_training_dataset_v4"
PRIOR = DIAG / "intervention_delay_window_audit"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    stable = [
        row for row in csv_rows(CONF)
        if row["oracle_confidence_class"].startswith("ORACLE_STABLE")
    ]
    if len(stable) != 277:
        raise AssertionError(f"expected 277 stable states, got {len(stable)}")
    hard_ids = {row["state_id"] for row in csv_rows(HARD)}
    if len(hard_ids) != 13 or not hard_ids <= {row["state_id"] for row in stable}:
        raise AssertionError("hard-13 mismatch")

    labels: dict[str, dict] = {}
    for version in ("v2", "v3", "v4"):
        obj = json.loads((DIAG / f"gphi_training_dataset_{version}/label_statistics.json").read_text())
        labels.update(obj.get("per_state", {}))
        labels.update(obj.get("new_states", {}))
    eta_by_id = {
        row["state_id"]: tuple(float(value) for value in labels[row["state_id"]]["eta_best"])
        for row in stable
    }

    # Inventory exact saved eta_best tuples across all dataset generations.
    saved_seeds: dict[str, set[int]] = {row["state_id"]: set() for row in stable}
    for version in ("v1", "v2", "v3", "v4"):
        for path in (DIAG / f"gphi_training_dataset_{version}/raw").rglob("records.jsonl"):
            for line in path.read_text().splitlines():
                record = json.loads(line)
                state_id = record.get("state_id")
                if state_id not in eta_by_id or record.get("execution_error") is not None:
                    continue
                if tuple(float(value) for value in record.get("eta", [])) == eta_by_id[state_id]:
                    saved_seeds[state_id].add(int(record["seed"]))

    protocol = json.loads((V4 / "protocol.json").read_text())
    v4_seeds = [int(value) for value in protocol["oracle_seed_default"]]
    if len(v4_seeds) != 64 or len(set(v4_seeds)) != 64:
        raise AssertionError("V4 seed inventory mismatch")
    prior_plan = json.loads((PRIOR / "audit_plan.json").read_text())
    prior_ids = {item["state_id"] for item in prior_plan["states"]}

    frozen = []
    for source in sorted(stable, key=lambda row: row["state_id"]):
        state_id = source["state_id"]
        eta = list(eta_by_id[state_id])
        y_long = int(source["original_gate_label"])
        if (y_long == 0) != (eta == [0.0, 0.0, 0.0]):
            raise AssertionError((state_id, y_long, eta))
        state_path = V4 / source["state_file"]
        if sha(state_path) != source["state_sha256"]:
            raise AssertionError(f"state hash mismatch: {state_id}")
        if state_id in prior_ids:
            seeds = v4_seeds
            seed_source = "prior_delay_audit_matched_seeds"
        elif len(saved_seeds[state_id]) == 64:
            seeds = sorted(saved_seeds[state_id])
            seed_source = "existing_eta_best_rollout_seeds"
        else:
            # Only D1_pair231 lacks a complete saved eta_best arm.  Reuse the
            # already validated V4 namespace rather than inventing new seeds.
            seeds = v4_seeds
            seed_source = "v4_protocol_fallback_no_complete_eta_best_arm"
        if len(seeds) != 64 or len(set(seeds)) != 64:
            raise AssertionError((state_id, seed_source, len(seeds)))
        frozen.append({
            "state_id": state_id,
            "selection_role": "FULL_ORACLE_STABLE_POOL",
            "is_hard13_anchor": state_id in hard_ids,
            "category": source["category"],
            "y_long": y_long,
            "oracle_confidence_class": source["oracle_confidence_class"],
            "source_group": source["source_group"],
            "source_trajectory": source["source_trajectory"],
            "state_file": str(state_path),
            "state_sha256": source["state_sha256"],
            "step": int(source["step"]),
            "rng_namespace": int(source["rng_namespace"]),
            "eta_best": json.dumps(eta, separators=(",", ":")),
            "seed_set_source": seed_source,
            "seeds_initial": json.dumps(seeds, separators=(",", ":")),
            "Q0_enlarged": float(source["Q0"]),
            "Q0_wilson95_lower": float(source["wilson95_lower"]),
            "Q0_wilson95_upper": float(source["wilson95_upper"]),
        })
    write_csv(HERE / "frozen_state_list.csv", frozen)

    states = []
    for row in frozen:
        item = dict(row)
        item["eta_best"] = json.loads(item["eta_best"])
        item["seeds_initial"] = json.loads(item["seeds_initial"])
        states.append(item)
    counts = Counter((row["category"], int(row["y_long"])) for row in frozen)
    plan = {
        "study": "intervention_window_and_gate_v1_stage1_window",
        "selection_frozen_before_new_delay_results": True,
        "selection": {
            "rule": "complete existing oracle-stable pool; no gate error, geometry, or prior delay result used",
            "stable_state_count": len(states),
            "source_group_count": len({row["source_group"] for row in frozen}),
            "hard13_named_diagnostic_count": len(hard_ids),
            "stratum_counts": {f"{category}|{label}": count for (category, label), count in sorted(counts.items())},
        },
        "coarse_delays": [0, 4, 8, 16, 32, 64],
        "adaptive_extension_delay": 128,
        "local_refinement_allowed": "any integer delay in [0,128], frozen before its rollout",
        "adaptive_seed_counts": [64, 128, 256],
        "seed_policy": {
            "per_state_seed_count": 64,
            "distinct_seed_sets": len({tuple(item["seeds_initial"]) for item in states}),
            "reuse_existing_eta_best_or_prior_delay_seeds": True,
            "matched_within_state_across_delays": True,
        },
        "semantics": {
            "delay_prefix": "for local steps k<d execute exactly first hard-projected u_safe; DiagnosticCorrector disabled",
            "downstream_policy": "from k>=d use frozen eta_best DiagnosticCorrector with unchanged FlowBC and both hard projections",
            "same_after_switch": "identical fixed-eta_best downstream for every delay; only d changes",
            "matched_randomness": "same state rng_namespace, continuation seed, and absolute step determine Flow keys across delays",
            "receding_oracle_requery_available": False,
            "limitation": "same fixed-eta_best approximation as prior delay audit; not arbitrary-state receding oracle re-query",
            "zero_eta_identity": "eta_best=0 makes every delay action-identical; materialize all delays from one canonical tuple",
        },
        "reuse_policy": {
            "dataset_d0": "exact saved eta_best tuple from V1--V4",
            "prior_d0_d4": "exact valid prior delay-audit tuple with matching state/seed/checkpoint/environment",
            "zero_eta_all_delays": "semantic alias of one canonical eta=0 tuple",
            "float_duplicate_tolerance": 1e-7,
            "no_duplicate_physical_tuple_execution": True,
        },
        "states": states,
        "environment": protocol["environment"],
        "checkpoint": protocol["checkpoint"],
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "frozen_hashes": {
            "oracle_confidence_dataset.csv": sha(CONF),
            "difficult_stable_states.csv": sha(HARD),
            "v4_protocol.json": sha(V4 / "protocol.json"),
            "prior_delay_audit_plan.json": sha(PRIOR / "audit_plan.json"),
            "frozen_state_list.csv": sha(HERE / "frozen_state_list.csv"),
        },
    }
    (HERE / "audit_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "states": len(states), "hard13": len(hard_ids),
        "source_groups": plan["selection"]["source_group_count"],
        "counts": plan["selection"]["stratum_counts"],
        "distinct_seed_sets": plan["seed_policy"]["distinct_seed_sets"],
        "plan_sha256": sha(HERE / "audit_plan.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
