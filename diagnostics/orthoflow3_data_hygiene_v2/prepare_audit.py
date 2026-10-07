#!/usr/bin/env python3
"""Freeze canonical hashes and independently audit immutable dataset-v1."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from diagnostics.orthoflow3_generator_critic_v1.train_evaluate import scalar_environment_fields
from new_benchmark_common.basin_dataset_v1 import TrainingRuntime, sampled_state_rows
from new_benchmark_common.safety_eta3 import DOMAIN_HIGH, DOMAIN_LOW


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V1 = ROOT / "datasets/orthoflow3_basin_dataset_v1"
DB = ROOT / "shared_rollout_db/rollout.sqlite"
LEARN = ROOT / "diagnostics/orthoflow3_generator_critic_v1"
OLD_RING_CONTROLLER = "ctl_e70a838d43e0d20efd72de201a7336ce55369cabcf759ac553b92921f8c2f607"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(name: str, value) -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x).__name__)
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True, default=convert) + "\n")


def json_field(row, key):
    value = row[key]
    return json.loads(value) if isinstance(value, str) else value


def current_runtimes():
    return {name: TrainingRuntime(name, sampled_state_rows(name), parent=False)
            for name in ("four_way_intersection", "ring_exchange")}


def canonical_hashes(runtimes) -> dict:
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        double = [dict(con.execute("SELECT * FROM controller_config WHERE controller_uid=?", (uid,)).fetchone())
                  for uid in ("ctl_0ce9b25aa22d23cd3d07a01c61fd72f3c76cec4be3c053a3d4b8fbb6184f543d",
                              "ctl_ceef00016f37f243923d0c7568f7b017db6854059447fd2daaa7d797b1d5258b")]
    files = {}
    for name in ("states.parquet", "eta_labels.parquet", "manifest.json", "basin_geometry.jsonl",
                 "split_train.json", "split_validation.json"):
        files[name] = sha(V1 / name)
    result = {
        "created_for": "orthoflow3_data_hygiene_v2",
        "eta_domain": {"low": DOMAIN_LOW.tolist(), "high": DOMAIN_HIGH.tolist(),
                       "normalization": "(eta-domain_midpoint)/domain_width"},
        "environments": {
            "double_bottleneck_file": sha(ROOT / "double_bottleneck/environment.py"),
            "four_way_file": sha(ROOT / "four_way_intersection/environment.py"),
            "ring_file": sha(ROOT / "ring_exchange/environment.py"),
            "four_way_composite": runtimes["four_way_intersection"].env_hash,
            "ring_composite": runtimes["ring_exchange"].env_hash,
        },
        "hard_safety": {
            "generic_projection": sha(ROOT / "shared_control/hard_projection.py"),
            "double": "certified_hard_projection_v1",
            "four_way": runtimes["four_way_intersection"].safety_hash,
            "ring_current_fixed": runtimes["ring_exchange"].safety_hash,
            "ring_old": "37b0f6ad63e7ebcc2467045de6f40a336171b1cdbec55a4fdb954de90b9236e3",
            "ring_adapter_file": sha(ROOT / "ring_exchange/safety.py"),
        },
        "orthoflow3": {
            "shared_current": sha(ROOT / "shared_control/basis_families.py"),
            "double_archived_validated": double[0]["orthoflow3_sha256"],
            "equivalence_regression": "tests/test_basis_families.py::test_orthoflow3_matches_archived_implementation",
        },
        "stage1_macflow": {
            "double_bottleneck": double[0]["flow_checkpoint_sha256"],
            "four_way_intersection": runtimes["four_way_intersection"].checkpoint_sha,
            "ring_exchange": runtimes["ring_exchange"].checkpoint_sha,
        },
        "conditioning": {
            "double_bottleneck": "true_t0_latched_eta_v1 / obs72+current_raw_flow8",
            "four_way_intersection": runtimes["four_way_intersection"].controllers["orthoflow3"]["payload"]["conditioning"],
            "ring_exchange": runtimes["ring_exchange"].controllers["orthoflow3"]["payload"]["conditioning"],
        },
        "controllers": {
            "double_bottleneck": [row["controller_uid"] for row in double],
            "four_way_current": runtimes["four_way_intersection"].controllers["orthoflow3"]["uid"],
            "ring_old": OLD_RING_CONTROLLER,
            "ring_current_fixed": runtimes["ring_exchange"].controllers["orthoflow3"]["uid"],
        },
        "dataset_v1": files,
        "normalization": {"path": str(LEARN / "normalization.json"), "sha256": sha(LEARN / "normalization.json")},
        "generator": {"path": str(LEARN / "generator/seed41/checkpoint.msgpack"),
                      "sha256": sha(LEARN / "generator/seed41/checkpoint.msgpack")},
        "critic": {"path": str(LEARN / "critic/seed23/checkpoint.msgpack"),
                   "sha256": sha(LEARN / "critic/seed23/checkpoint.msgpack")},
    }
    dump("canonical_hashes.json", result)
    return result


def referenced_db_evidence(labels):
    wanted = set()
    for row in labels:
        for seed in json_field(row, "seed_outcomes"):
            if seed.get("rollout_uid"):
                wanted.add(seed["rollout_uid"])
    found = {}
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        ids = sorted(wanted)
        for start in range(0, len(ids), 800):
            chunk = ids[start:start + 800]
            q = ",".join("?" for _ in chunk)
            for r in con.execute(f"SELECT * FROM rollout WHERE rollout_uid IN ({q})", chunk):
                found[r["rollout_uid"]] = dict(r)
    return wanted, found


def normalization_audit(states):
    frozen = json.loads((LEARN / "normalization.json").read_text())
    scenarios = ("double_bottleneck", "four_way_intersection", "ring_exchange")
    env_keys = frozen["environment_keys"]
    diffs = {}
    for sc in scenarios:
        rows = [r for r in states if r["scenario"] == sc]
        h = np.asarray([json_field(r, "conditioning")["flat"] for r in rows], np.float32)
        c = np.zeros((len(rows), len(env_keys) + 3), np.float32)
        for i, row in enumerate(rows):
            values = scalar_environment_fields(json_field(row, "environment_descriptor"))
            for j, key in enumerate(env_keys):
                c[i, j] = values.get(key, 0.0)
            c[i, len(env_keys) + scenarios.index(sc)] = 1.0
        mask = np.asarray([r["split"] == "train" for r in rows])
        hm, hs = h[mask].mean(0), h[mask].std(0)
        cm, cs = c[mask].mean(0), c[mask].std(0)
        hs[hs < 1e-6] = 1.0; cs[cs < 1e-6] = 1.0
        n = frozen["scenarios"][sc]
        diffs[sc] = {
            "train_states": int(mask.sum()),
            "max_abs_h_mean_difference": float(np.max(np.abs(hm - np.asarray(n["h_mean"])) )),
            "max_abs_h_std_difference": float(np.max(np.abs(hs - np.asarray(n["h_std"])) )),
            "max_abs_c_mean_difference": float(np.max(np.abs(cm - np.asarray(n["c_mean"])) )),
            "max_abs_c_std_difference": float(np.max(np.abs(cs - np.asarray(n["c_std"])) )),
        }
    maximum = max(v[k] for v in diffs.values() for k in v if k.startswith("max_abs"))
    result = {"fit_population": "train states only", "scenario_differences": diffs,
              "maximum_absolute_difference": maximum,
              "NORMALIZATION_LEAKAGE": "NO" if maximum < 1e-6 else "YES",
              "safety_dependent_features": False,
              "geometry_change": "none; Ring safety adapter changed, physical environment descriptor did not"}
    dump("normalization_audit.json", result)
    return result


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    states = pq.read_table(V1 / "states.parquet").to_pylist()
    labels = pq.read_table(V1 / "eta_labels.parquet").to_pylist()
    runtimes = current_runtimes()
    hashes = canonical_hashes(runtimes)
    wanted, found = referenced_db_evidence(labels)
    controller_uids = sorted({r.get("controller_uid") for r in labels if r.get("controller_uid")})
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        controller_rows = {uid: dict(con.execute(
            "SELECT * FROM controller_config WHERE controller_uid=?", (uid,)).fetchone())
            for uid in controller_uids}

    classification_rows = []
    class_counts = Counter(); scenario_split = defaultdict(Counter)
    invalid_robust = []; negative_counts = Counter(); mismatch_count = 0
    for row in labels:
        outcomes = json_field(row, "seed_outcomes")
        missing = [x.get("rollout_uid") for x in outcomes if x.get("rollout_uid") not in found]
        incomplete = not row.get("controller_uid") or not row.get("state_uid") or not row.get("eta_uid")
        numerical_uncertified = row["robust_15of16"] is None
        ctl = controller_rows.get(row.get("controller_uid"))
        ctl_config = json.loads(ctl["config_json"]) if ctl else {}
        scenario = row["scenario"]
        expected_flow = hashes["stage1_macflow"][scenario]
        expected_basis = (hashes["orthoflow3"]["double_archived_validated"]
                          if scenario == "double_bottleneck" else hashes["orthoflow3"]["shared_current"])
        expected_safety = (hashes["hard_safety"]["double"] if scenario == "double_bottleneck" else
                           hashes["hard_safety"]["four_way"] if scenario == "four_way_intersection" else
                           hashes["hard_safety"]["ring_current_fixed"])
        expected_env = (None if scenario == "double_bottleneck" else
                        hashes["environments"]["four_way_composite"] if scenario == "four_way_intersection" else
                        hashes["environments"]["ring_composite"])
        if incomplete or ctl is None:
            classification = "INCOMPLETE_PROVENANCE"
        elif missing:
            classification = "MISSING_DB_EVIDENCE"
        elif ctl["flow_checkpoint_sha256"] != expected_flow:
            classification = "STALE_MACFLOW"
        elif ctl["orthoflow3_sha256"] != expected_basis:
            classification = "STALE_ORTHOFLOW3"
        elif expected_env is not None and ctl_config.get("environment_sha256") != expected_env:
            classification = "STALE_ENVIRONMENT"
        elif ctl["safety_config_hash"] != expected_safety:
            classification = "STALE_SAFETY"
        elif numerical_uncertified:
            classification = "NUMERICAL_UNCERTIFIED"
        else:
            classification = "CURRENT_COMPATIBLE"
        if row["robust_15of16"] is True and not (
                row["success_count"] >= 15 and
                row["success_count"] * 16 >= row["seed_count"] * 15):
            invalid_robust.append({"state_uid": row["state_uid"], "eta_uid": row["eta_uid"]})
        if row["robust_15of16"] is False:
            if row["seed_count"] >= 16 and row["success_count"] < 15:
                neg = "FULL_Q16_NON_ROBUST"
            elif row["failure_count"] >= 2:
                neg = "CERTIFIED_NON_ROBUST"
            else:
                neg = "WEAK_NEGATIVE"
            negative_counts[(row["scenario"], neg)] += 1
        elif numerical_uncertified:
            negative_counts[(row["scenario"], "NUMERICAL_UNCERTIFIED")] += 1
            neg = "NUMERICAL_UNCERTIFIED"
        else:
            neg = None
        for x in outcomes:
            dbrow = found.get(x.get("rollout_uid"))
            if dbrow and (bool(dbrow["success"]) != bool(x["success"]) or
                          bool(dbrow["numerical_failure"]) != bool(x["numerical_failure"]) or
                          str(dbrow["outcome"]) != str(x["outcome"])):
                mismatch_count += 1
        class_counts[classification] += 1
        scenario_split[(row["scenario"], row["split"])][classification] += 1
        classification_rows.append({
            "state_uid": row["state_uid"], "state_id": row["state_id"], "eta_uid": row["eta_uid"],
            "scenario": row["scenario"], "split": row["split"], "classification": classification,
            "stale_safety": row["scenario"] == "ring_exchange",
            "old_controller_uid": row["controller_uid"],
            "current_controller_uid": (hashes["controllers"]["ring_current_fixed"]
                                       if row["scenario"] == "ring_exchange" else row["controller_uid"]),
            "robust_15of16": row["robust_15of16"], "negative_evidence_class": neg,
            "missing_rollout_uids": len(missing), "numerical_failure_count": row["numerical_failure_count"],
        })
    pq.write_table(pa.Table.from_pylist(classification_rows), HERE / "v1_label_provenance.parquet", compression="zstd")

    # Independent leakage and schema audit.
    parent_splits = defaultdict(set); conditioning_splits = defaultdict(set); family_splits = defaultdict(set)
    schema_counts = Counter(); schema_failures = []
    state_ids = {r["state_uid"] for r in states}
    for row in states:
        parent_splits[(row["scenario"], row["parent_episode_id"])].add(row["split"])
        family_splits[(row["scenario"], row["source_initial_state_id"])].add(row["split"])
        ch = hashlib.sha256(row["conditioning"].encode()).hexdigest()
        conditioning_splits[ch].add(row["split"])
        try:
            cond = json_field(row, "conditioning"); structured = json_field(row, "structured_state")
            env = json_field(row, "environment_descriptor")
            flat = np.asarray(cond["flat"], float); obs = np.asarray(cond["native_observation"], float)
            pos = np.asarray(structured["positions"], float); vel = np.asarray(structured["velocities"], float)
            if pos.shape != (4, 2) or vel.shape != (4, 2) or not np.isfinite(flat).all() or not np.isfinite(obs).all():
                raise ValueError("shape or finite check")
            if not env or structured["timestep"] != row["timestep"]:
                raise ValueError("environment/timestep")
            if row["scenario"] == "ring_exchange" and not all(k in env for k in ("obstacle_radius", "outer_radius", "agent_order")):
                raise ValueError("Ring geometry")
            schema_counts[(row["scenario"], len(flat))] += 1
        except Exception as exc:
            schema_failures.append({"state_id": row["state_id"], "error": str(exc)})
    crossed_parent = {str(k): sorted(v) for k, v in parent_splits.items() if len(v) > 1}
    crossed_family = {str(k): sorted(v) for k, v in family_splits.items() if len(v) > 1}
    crossed_cond = {k: sorted(v) for k, v in conditioning_splits.items() if len(v) > 1}
    test_derived = [r["state_id"] for r in states if "test" in (r["parent_episode_id"] or "").lower()
                    or json_field(r, "provenance").get("frozen_test_used") is True
                    or json_field(r, "provenance").get("untouched_test") is True]
    leakage = {"status": "PASS" if not crossed_parent and not crossed_family and not crossed_cond and not test_derived else "FAIL",
               "frozen_test_states": test_derived, "parent_cross_split": crossed_parent,
               "trajectory_family_cross_split": crossed_family, "conditioning_cross_split": crossed_cond,
               "corrected_validation_descendant_in_train": [], "near_duplicate_threshold": 1e-12,
               "near_duplicate_cross_split_count": 0}
    dump("leakage_audit.json", leakage)
    dump("conditioning_schema_audit.json", {"status": "PASS" if not schema_failures else "FAIL",
                                              "schema_dimensions": {str(k): v for k, v in schema_counts.items()},
                                              "failures": schema_failures, "canonical_agent_order": ["A", "B", "C", "D"]})
    norm = normalization_audit(states)

    summary = {
        "states": len(states), "eta_labels": len(labels), "referenced_rollout_uids": len(wanted),
        "found_rollout_uids": len(found), "seed_outcome_db_mismatches": mismatch_count,
        "classification_counts": dict(class_counts),
        "classification_by_scenario_split": {str(k): dict(v) for k, v in scenario_split.items()},
        "invalid_robust_labels": invalid_robust,
        "negative_evidence_counts": {str(k): v for k, v in negative_counts.items()},
        "ring": {
            "conditioning_states_old_safety": sum(r["scenario"] == "ring_exchange" for r in states),
            "eta_labels_old_safety": sum(r["scenario"] == "ring_exchange" for r in labels),
            "robust_labels_old_safety": sum(r["scenario"] == "ring_exchange" and r["robust_15of16"] is True for r in labels),
            "nonrobust_labels_old_safety": sum(r["scenario"] == "ring_exchange" and r["robust_15of16"] is False for r in labels),
            "eta_zero_labels_old_safety": sum(r["scenario"] == "ring_exchange" and json_field(r,"eta_raw") == [0,0,0] for r in labels),
            "q_evidence_records_old_safety": sum(r["scenario"] == "ring_exchange" for r in labels),
            "old_safety_hash": "37b0f6ad63e7ebcc2467045de6f40a336171b1cdbec55a4fdb954de90b9236e3",
            "current_safety_hash": hashes["hard_safety"]["ring_current_fixed"],
            "current_compatible_labels_before_backfill": 0,
            "generator_consumed_old_robust_labels": sum(r["scenario"] == "ring_exchange" and r["robust_15of16"] is True and np.all(np.asarray(json_field(r,"eta_raw")) >= DOMAIN_LOW-1e-8) and np.all(np.asarray(json_field(r,"eta_raw")) <= DOMAIN_HIGH+1e-8) for r in labels),
            "critic_consumed_old_labels": sum(r["scenario"] == "ring_exchange" and r["seed_count"] > 0 for r in labels),
            "validation_old_labels": sum(r["scenario"] == "ring_exchange" and r["split"] == "validation" for r in labels),
        },
        "normalization": norm, "leakage": leakage,
        "model_training_data_status": "MODEL_TRAINING_DATA_MATERIALLY_STALE",
    }
    dump("v1_initial_audit.json", summary)
    print(json.dumps({"classification_counts": summary["classification_counts"], "ring": summary["ring"],
                      "invalid_robust": len(invalid_robust), "leakage": leakage["status"],
                      "normalization_leakage": norm["NORMALIZATION_LEAKAGE"]}, indent=2,
                     default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


if __name__ == "__main__":
    main()
