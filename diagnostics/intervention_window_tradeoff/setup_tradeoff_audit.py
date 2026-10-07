"""Freeze the complete oracle-stable pool and tradeoff-audit protocol."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path("/home/zhihan/research/Basin_C1")
DIAG = ROOT / "diagnostics"
HERE = Path(__file__).resolve().parent
V4 = DIAG / "gphi_training_dataset_v4"
CONF = DIAG / "gphi_gate_confidence_aware_v1/oracle_confidence_dataset.csv"
HARD = DIAG / "hard_stable_boundary_crossval/difficult_stable_states.csv"
PRIOR = DIAG / "intervention_delay_window_audit"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict]:
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
    confidence = [
        row for row in read_csv(CONF)
        if row["oracle_confidence_class"].startswith("ORACLE_STABLE")
    ]
    if len(confidence) != 277:
        raise AssertionError(f"expected 277 oracle-stable states, got {len(confidence)}")
    hard_ids = {row["state_id"] for row in read_csv(HARD)}
    if len(hard_ids) != 13 or not hard_ids <= {row["state_id"] for row in confidence}:
        raise AssertionError("hard-13 inventory mismatch")

    label_info: dict[str, dict] = {}
    for version in ("v2", "v3", "v4"):
        obj = json.loads((DIAG / f"gphi_training_dataset_{version}/label_statistics.json").read_text())
        label_info.update(obj.get("per_state", {}))
        label_info.update(obj.get("new_states", {}))

    protocol = json.loads((V4 / "protocol.json").read_text())
    seeds = [int(value) for value in protocol["oracle_seed_default"]]
    if len(seeds) != 64 or len(set(seeds)) != 64:
        raise AssertionError("expected 64 unique frozen oracle seeds")

    # Freeze per-state matched seeds to maximize exact reuse.  States already
    # audited at d=0/4 retain that audit's seed set.  Every other state uses
    # its 64 already-saved eta_best rollout seeds from V1--V4.
    prior_plan = json.loads((PRIOR / "audit_plan.json").read_text())
    prior_ids = {item["state_id"] for item in prior_plan["states"]}
    existing_eta_seeds: dict[str, set[int]] = {row["state_id"]: set() for row in confidence}
    eta_by_id = {
        row["state_id"]: tuple(float(value) for value in label_info[row["state_id"]]["eta_best"])
        for row in confidence
    }
    for version in ("v1", "v2", "v3", "v4"):
        for records_path in (DIAG / f"gphi_training_dataset_{version}/raw").rglob("records.jsonl"):
            for line in records_path.read_text().splitlines():
                record = json.loads(line)
                state_id = record.get("state_id")
                if state_id not in eta_by_id or record.get("execution_error") is not None:
                    continue
                eta = tuple(float(value) for value in record.get("eta", []))
                if eta == eta_by_id[state_id]:
                    existing_eta_seeds[state_id].add(int(record["seed"]))

    rows = []
    for source in sorted(confidence, key=lambda row: row["state_id"]):
        state_id = source["state_id"]
        labels = label_info[state_id]
        eta = [float(value) for value in labels["eta_best"]]
        y_long = int(source["original_gate_label"])
        if (y_long == 0) != (eta == [0.0, 0.0, 0.0]):
            raise AssertionError((state_id, y_long, eta))
        state_path = V4 / source["state_file"]
        if sha(state_path) != source["state_sha256"]:
            raise AssertionError(f"state hash mismatch: {state_id}")
        if state_id in prior_ids:
            state_seeds = seeds
            seed_set_source = "prior_delay_audit_matched_seeds"
        else:
            state_seeds = sorted(existing_eta_seeds[state_id])
            seed_set_source = "existing_eta_best_rollout_seeds"
            if len(state_seeds) != 64:
                # D1_pair231 has no saved eta_best arm in V1--V4.  Freeze the
                # already validated V4 continuation seed set rather than
                # inventing a new seed namespace.
                state_seeds = seeds
                seed_set_source = "v4_protocol_fallback_no_saved_eta_best"
        if len(state_seeds) != 64 or len(set(state_seeds)) != 64:
            raise AssertionError((state_id, seed_set_source, len(state_seeds)))
        rows.append({
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
            "seed_set_source": seed_set_source,
            "seeds_initial": json.dumps(state_seeds, separators=(",", ":")),
            "Q0_enlarged": float(source["Q0"]),
            "Q0_wilson95_lower": float(source["wilson95_lower"]),
            "Q0_wilson95_upper": float(source["wilson95_upper"]),
        })
    write_csv(HERE / "frozen_state_list.csv", rows)

    states = []
    for row in rows:
        item = dict(row)
        item["eta_best"] = json.loads(item["eta_best"])
        item["seeds_initial"] = json.loads(item["seeds_initial"])
        states.append(item)
    counts = Counter((row["category"], int(row["y_long"])) for row in rows)
    plan = {
        "study": "intervention_window_tradeoff",
        "selection_frozen_before_new_delay_results": True,
        "selection": {
            "rule": "complete existing oracle-stable pool; no geometry, gate error, or prior delay result used",
            "stable_state_count": len(states),
            "hard13_named_diagnostic_count": len(hard_ids),
            "source_group_count": len({row["source_group"] for row in rows}),
            "stratum_counts": {f"{c}|{y}": n for (c, y), n in sorted(counts.items())},
        },
        "coarse_delays": [0, 4, 8, 16, 32, 64],
        "adaptive_extension_delay": 128,
        "adaptive_seed_counts": [64, 128, 256],
        "seed_policy": {
            "per_state_seed_count": 64,
            "prior_delay_states_retain_prior_seed_set": True,
            "other_states_use_existing_eta_best_seed_set": True,
            "matched_within_state_across_delays": True,
            "distinct_seed_sets": len({tuple(item["seeds_initial"]) for item in states}),
        },
        "semantics": {
            "delay_prefix": "for local physical steps k < d execute exactly first hard-projected u_safe; DiagnosticCorrector disabled",
            "downstream_policy": "from local step k >= d use the state's frozen eta_best DiagnosticCorrector plus unchanged FlowBC and both hard projections",
            "same_after_switch": "identical fixed-eta_best downstream implementation for every branch; only d changes",
            "matched_randomness": "same state rng_namespace, continuation seed, and absolute environment step determine Flow keys across delays",
            "receding_oracle_requery_available": False,
            "limitation": "fixed eta_best is the same validated approximation used by the prior delay audit; it is not arbitrary-state receding oracle re-query",
            "zero_eta_identity": "eta_best=0 makes every delay branch action-identical; materialize all delays from one canonical tuple",
        },
        "reuse_policy": {
            "v4_d0": "reuse exact V4 (state_id, eta_best, seed) continuation",
            "prior_d0_d4": "reuse exact valid tuples from intervention_delay_window_audit when state, seed, checkpoint and environment match",
            "zero_eta_all_delays": "materialize any delay from one canonical eta_best=0 tuple",
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
        "states": len(states),
        "hard13": len(hard_ids),
        "source_groups": plan["selection"]["source_group_count"],
        "counts": plan["selection"]["stratum_counts"],
        "seed_count": len(seeds),
        "distinct_seed_sets": plan["seed_policy"]["distinct_seed_sets"],
        "plan_sha256": sha(HERE / "audit_plan.json"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
