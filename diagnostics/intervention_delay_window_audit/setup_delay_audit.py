"""Freeze a result-independent intervention-delay audit cohort and protocol."""

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
GENERIC_PER_STRATUM = 10
SELECTION_NAMESPACE = "step_aligned_delay_audit_v1"


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


def stable_hash(state_id: str) -> str:
    return hashlib.sha256(f"{SELECTION_NAMESPACE}:{state_id}".encode()).hexdigest()


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    confidence = [
        row for row in read_csv(CONF)
        if row["oracle_confidence_class"].startswith("ORACLE_STABLE")
    ]
    if len(confidence) != 277:
        raise AssertionError(f"expected frozen stable pool of 277, got {len(confidence)}")
    hard_ids = {row["state_id"] for row in read_csv(HARD)}
    if len(hard_ids) != 13 or not hard_ids <= {row["state_id"] for row in confidence}:
        raise AssertionError("hard-13 inventory mismatch")

    label_info: dict[str, dict] = {}
    for version in ("v2", "v3", "v4"):
        obj = json.loads((DIAG / f"gphi_training_dataset_{version}/label_statistics.json").read_text())
        label_info.update(obj.get("per_state", {}))
        label_info.update(obj.get("new_states", {}))

    # Generic cohort: fixed-hash top K within every category x long-label
    # stratum.  No geometry, prior error, gate output, or recovery anchor is
    # used in ranking.  Hard-13 are appended only as named diagnostics.
    strata: dict[tuple[str, int], list[dict]] = {}
    for row in confidence:
        key = (row["category"], int(row["original_gate_label"]))
        strata.setdefault(key, []).append(row)
    expected_strata = {
        ("NORMAL", 0), ("NORMAL", 1), ("PRE_DEADLOCK", 1),
        ("RECOVERY", 0), ("RECOVERY", 1),
    }
    if set(strata) != expected_strata:
        raise AssertionError(sorted(strata))
    generic_ids = set()
    for key, candidates in strata.items():
        ranked = sorted(candidates, key=lambda row: (stable_hash(row["state_id"]), row["state_id"]))
        generic_ids.update(row["state_id"] for row in ranked[:GENERIC_PER_STRATUM])
    selected_ids = generic_ids | hard_ids

    protocol = json.loads((V4 / "protocol.json").read_text())
    seeds = [int(value) for value in protocol["oracle_seed_default"]]
    if len(seeds) != 64 or len(set(seeds)) != 64:
        raise AssertionError("expected 64 unique pre-existing oracle seeds")

    rows = []
    for source in sorted(confidence, key=lambda row: row["state_id"]):
        state_id = source["state_id"]
        if state_id not in selected_ids:
            continue
        labels = label_info[state_id]
        eta = [float(value) for value in labels["eta_best"]]
        y_long = int(source["original_gate_label"])
        if (y_long == 0) != (eta == [0.0, 0.0, 0.0]):
            raise AssertionError((state_id, y_long, eta))
        state_path = V4 / source["state_file"]
        if sha(state_path) != source["state_sha256"]:
            raise AssertionError(f"state hash mismatch: {state_id}")
        rows.append({
            "state_id": state_id,
            "selection_role": (
                "GENERIC_HASH_STRATIFIED_AND_HARD13" if state_id in generic_ids and state_id in hard_ids
                else "GENERIC_HASH_STRATIFIED" if state_id in generic_ids
                else "NAMED_HARD13_ANCHOR"
            ),
            "is_generic_cohort": state_id in generic_ids,
            "is_hard13_anchor": state_id in hard_ids,
            "selection_hash": stable_hash(state_id),
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
            "Q0_enlarged": float(source["Q0"]),
            "Q0_wilson95_lower": float(source["wilson95_lower"]),
            "Q0_wilson95_upper": float(source["wilson95_upper"]),
        })
    if len(generic_ids) != 50 or len(rows) != len(selected_ids):
        raise AssertionError((len(generic_ids), len(rows), len(selected_ids)))
    write_csv(HERE / "frozen_state_list.csv", rows)

    state_objects = []
    for row in rows:
        obj = dict(row)
        obj["eta_best"] = json.loads(obj["eta_best"])
        state_objects.append(obj)
    counts = Counter((row["category"], int(row["y_long"])) for row in rows)
    plan = {
        "study": "step_aligned_intervention_delay_window",
        "selection_frozen_before_rollout": True,
        "stable_pool_size": len(confidence),
        "selection": {
            "generic_rule": "within each existing category x y_long stratum, take the 10 lowest SHA256(step_aligned_delay_audit_v1:state_id)",
            "generic_rule_uses_prior_failures": False,
            "generic_rule_uses_geometry": False,
            "generic_state_count": len(generic_ids),
            "named_hard13_anchors_appended_after_generic_selection": True,
            "named_hard13_count": len(hard_ids),
            "selected_unique_state_count": len(rows),
            "stratum_counts": {f"{category}|{label}": count for (category, label), count in sorted(counts.items())},
            "namespace": SELECTION_NAMESPACE,
        },
        "delays": [0, 1, 2, 4],
        "seeds_initial": seeds,
        "adaptive_seed_counts": [64, 128, 256],
        "semantics": {
            "delay_prefix": "for local physical steps k < d execute exactly the first hard-projected Safety baseline u_safe,k; do not apply DiagnosticCorrector",
            "downstream_policy": "from local step k >= d use the audited state's frozen eta_best DiagnosticCorrector, FlowBC, first hard projection, and second hard projection",
            "same_after_switch": "the downstream implementation is identical for every delay branch; only d changes",
            "d0": "eta_best mechanism available immediately; eta_best=0 naturally executes no additional correction",
            "matched_randomness": "same state rng_namespace, continuation seed, and absolute environment step determine each Flow key for all delays",
            "receding_oracle_requery_available": False,
            "fallback_reason": "the validated success-basin oracle is an offline fixed-eta continuation search and exposes no online eta re-query for arbitrary reached augmented states; inventing one would change the frozen oracle",
            "fallback": "closest existing oracle-consistent fixed-eta_best downstream; identical across branches",
        },
        "reuse_policy": {
            "d0": "reuse exact existing V4 (state_id, eta_best, seed) continuation tuples",
            "zero_eta_all_delays": "when eta_best is exactly zero, all delay branches are identical and may reuse the same exact tuple",
            "d1": "reuse exact prior conceptual-audit Branch-N tuples when plan checkpoint/environment/state/seed hashes match",
            "no_duplicate_valid_tuples": True,
        },
        "states": state_objects,
        "environment": protocol["environment"],
        "checkpoint": protocol["checkpoint"],
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "frozen_hashes": {
            "oracle_confidence_dataset.csv": sha(CONF),
            "difficult_stable_states.csv": sha(HARD),
            "v4_protocol.json": sha(V4 / "protocol.json"),
            "frozen_state_list.csv": sha(HERE / "frozen_state_list.csv"),
        },
    }
    (HERE / "audit_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "selected_states": len(rows),
        "generic_states": len(generic_ids),
        "hard13_anchors": len(hard_ids),
        "hard13_overlap_generic": len(generic_ids & hard_ids),
        "counts": {f"{k[0]}|{k[1]}": v for k, v in sorted(counts.items())},
        "seeds": len(seeds),
        "plan_sha256": sha(HERE / "audit_plan.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
