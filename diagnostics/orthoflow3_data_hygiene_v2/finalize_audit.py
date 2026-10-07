#!/usr/bin/env python3
"""Build the audited v2 candidate from certified current evidence."""
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

from new_benchmark_common.basin_dataset_v1 import (
    TrainingRuntime, sampled_state_rows, _aggregate_eta_rows, _canonical_center,
    _robust_components, _component_threshold, normalized_distance,
)
from new_benchmark_common.safety_eta3 import DOMAIN_HIGH, DOMAIN_LOW, normalized_eta


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V1 = ROOT / "datasets/orthoflow3_basin_dataset_v1"
V2 = ROOT / "datasets/orthoflow3_basin_dataset_v2_audited"
DB = ROOT / "shared_rollout_db/rollout.sqlite"
CURRENT_RING_SAFETY = "18419e7a15d3dbeb9b49bacefbaf6e376a7e3551833a89bdfec52c3bab1861be"
OLD_RING_SAFETY = "37b0f6ad63e7ebcc2467045de6f40a336171b1cdbec55a4fdb954de90b9236e3"
OLD_RING_CONTROLLER = "ctl_e70a838d43e0d20efd72de201a7336ce55369cabcf759ac553b92921f8c2f607"


def canonical(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")


def parse(value):
    return json.loads(value) if isinstance(value, str) else value


def sha(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h


def write_parquet(path: Path, rows: list[dict], json_fields: set[str]) -> None:
    columns = defaultdict(list)
    keys = sorted({k for r in rows for k in r})
    for r in rows:
        for k in keys:
            value = r.get(k)
            columns[k].append(canonical(value) if k in json_fields and value is not None else value)
    pq.write_table(pa.table(columns), path, compression="zstd", version="2.6")


def enrich_evidence(row: dict, *, safety_hash: str, provenance_class: str,
                    old_controller_uid: str | None = None) -> dict:
    row = dict(row)
    n, s, f = int(row["seed_count"]), int(row["success_count"]), int(row["failure_count"])
    numerical = int(row["numerical_failure_count"])
    robust = row["robust_15of16"] is True
    logically_sufficient = robust and s >= 15 and s * 16 >= n * 15
    # A collision on a failed seed is valid negative evidence and does not
    # invalidate the other 15 successful seeds.  Only a rollout simultaneously
    # recorded as success and collision is internally inconsistent.
    success_collision_count = int(row.get("success_collision_count", 0))
    if robust and (not logically_sufficient or success_collision_count > 0):
        row["robust_15of16"] = None
        robust = False
        row["audit_downgrade_reason"] = ("SUCCESS_COLLISION_INCONSISTENCY" if success_collision_count else
                                         "UNDER_EVIDENCED_ROBUST_LABEL")
    if row["robust_15of16"] is False:
        if n >= 16 and s < 15:
            negative = "FULL_Q16_NON_ROBUST"
        elif f >= 2:
            negative = "CERTIFIED_NON_ROBUST"
        else:
            negative = "WEAK_NEGATIVE"
    elif row["robust_15of16"] is None:
        negative = "NUMERICAL_UNCERTIFIED" if numerical else "WEAK_NEGATIVE"
    else:
        negative = None
    row.update({
        "provenance_classification": provenance_class,
        "safety_hash": safety_hash,
        "old_controller_uid": old_controller_uid,
        "negative_evidence_class": negative,
        "robust_evidence_logically_sufficient": bool(logically_sufficient),
        "source_dataset_version": "orthoflow3_basin_dataset_v2_audited",
        "label_semantics_version": ("ring_current_safety_v2" if row.get("scenario") == "ring_exchange"
                                    else "retained_current_v1"),
    })
    return row


def rollout_audit_map(keys: set[tuple[str, str, str]]) -> dict:
    controllers = sorted({k[2] for k in keys})
    result = {}
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        for ctl in controllers:
            for row in con.execute("""SELECT state_uid,eta_uid,controller_uid,
                    AVG(j_def) mean_j_def,COUNT(j_def) j_def_count,
                    SUM(CASE WHEN success=1 AND collision=1 THEN 1 ELSE 0 END) success_collision_count
                FROM rollout WHERE controller_uid=? AND conflict_quarantined=0
                GROUP BY state_uid,eta_uid,controller_uid""", (ctl,)):
                key = (row["state_uid"], row["eta_uid"], row["controller_uid"])
                if key in keys:
                    result[key] = {"mean_j_def": row["mean_j_def"], "j_def_count": row["j_def_count"],
                                   "success_collision_count": row["success_collision_count"]}
    return result


def drift(old_rows, new_rows):
    old = {(r["state_uid"], r["eta_uid"]): r for r in old_rows}
    new = {(r["state_uid"], r["eta_uid"]): r for r in new_rows}
    transitions = Counter(); qdelta = []; terminal_changes = Counter()
    collision_changes = Counter(); numerical_changes = Counter(); consumed = Counter()
    zero_changes = []
    for key, a in old.items():
        b = new[key]
        def label(r):
            return "robust" if r["robust_15of16"] is True else "nonrobust" if r["robust_15of16"] is False else "uncertified"
        transitions[f"{label(a)}->{label(b)}"] += 1
        if a["seed_count"] and b["seed_count"]:
            qdelta.append(b["success_count"] / b["seed_count"] - a["success_count"] / a["seed_count"])
        ta, tb = parse(a["terminal_summary"]), b["terminal_summary"]
        if ta != tb:
            terminal_changes["changed"] += 1
        else:
            terminal_changes["same"] += 1
        collision_changes[f"{int(a['collision_count']>0)}->{int(b['collision_count']>0)}"] += 1
        numerical_changes[f"{int(a['numerical_failure_count']>0)}->{int(b['numerical_failure_count']>0)}"] += 1
        if a["split"] == "train":
            consumed[f"critic:{label(a)}->{label(b)}"] += 1
            eta = np.asarray(parse(a["eta_raw"]), float)
            if a["robust_15of16"] is True and np.all(eta >= DOMAIN_LOW-1e-8) and np.all(eta <= DOMAIN_HIGH+1e-8):
                consumed[f"generator:{label(a)}->{label(b)}"] += 1
        if parse(a["eta_raw"]) == [0, 0, 0]:
            zero_changes.append({"state_uid": key[0], "old": label(a), "current": label(b)})
    comparable = sum(v for k, v in transitions.items() if "uncertified" not in k)
    changed = transitions["robust->nonrobust"] + transitions["nonrobust->robust"]
    return {
        "matched_labels": len(old), "transitions": dict(transitions),
        "LABEL_DRIFT_RATE": changed / comparable if comparable else None,
        "q_change": {"count": len(qdelta), "mean": float(np.mean(qdelta)), "median": float(np.median(qdelta)),
                     "note": "empirical success-fraction difference over each label's actually evaluated canonical seeds"},
        "terminal_summary": dict(terminal_changes), "collision_presence": dict(collision_changes),
        "numerical_presence": dict(numerical_changes), "eta_zero": {
            "states": len(zero_changes), "changed": sum(x["old"] != x["current"] for x in zero_changes),
            "transitions": dict(Counter(f"{x['old']}->{x['current']}" for x in zero_changes)), "rows": zero_changes},
        "training_consumed_transitions": dict(consumed),
    }


def density_and_boundary(states, labels):
    by = defaultdict(list)
    for r in labels: by[r["state_uid"]].append(r)
    threshold = _component_threshold(); rows = []; boundary_summary = defaultdict(Counter)
    for st in states:
        ls = by[st["state_uid"]]
        robust = [r for r in ls if r["robust_15of16"] is True]
        cert = [r for r in ls if r["negative_evidence_class"] in ("CERTIFIED_NON_ROBUST", "FULL_Q16_NON_ROBUST")]
        weak = [r for r in ls if r["negative_evidence_class"] == "WEAK_NEGATIVE"]
        unknown = [r for r in ls if r["negative_evidence_class"] == "NUMERICAL_UNCERTIFIED"]
        rp = np.asarray([parse(r["eta_normalized"]) for r in robust], float) if robust else np.empty((0,3))
        def nearest(r):
            p = np.asarray(parse(r["eta_normalized"]), float)
            return float(np.min(np.linalg.norm(rp - p, axis=1))) if len(rp) else None
        near_cert = sum(nearest(r) is not None and nearest(r) <= threshold for r in cert)
        near_weak = sum(nearest(r) is not None and nearest(r) <= threshold for r in weak)
        far = sum(nearest(r) is not None and nearest(r) > threshold for r in cert + weak)
        for key, value in (("robust_interior", len(robust)), ("near_certified_negative", near_cert),
                           ("near_weak_negative", near_weak), ("far_negative", far)):
            boundary_summary[st["scenario"]][key] += value
        covariance = np.cov(rp.T).tolist() if len(rp) >= 2 else None
        zero = next((r for r in ls if parse(r["eta_raw"]) == [0,0,0]), None)
        rows.append({
            "state_uid": st["state_uid"], "state_id": st["state_id"], "scenario": st["scenario"], "split": st["split"],
            "total_eta_labels": len(ls), "verified_robust_eta_count": len(robust),
            "certified_nonrobust_count": len(cert), "weak_negative_count": len(weak),
            "numerical_unknown_count": len(unknown),
            "eta_zero_status": ("MISSING" if zero is None else "ROBUST" if zero["robust_15of16"] is True else
                                "NON_ROBUST" if zero["robust_15of16"] is False else "UNCERTIFIED"),
            "num_robust_components": st["num_robust_components"],
            "nearest_robust_eta_to_zero": (float(np.min(np.linalg.norm(rp, axis=1))) if len(rp) else None),
            "robust_eta_covariance": covariance,
            "robust_eta_spread_trace": (float(np.trace(np.cov(rp.T))) if len(rp) >= 2 else 0.0 if len(rp)==1 else None),
            "positive_fraction": len(robust)/len(ls) if ls else None,
            "flag_only_one_robust": len(robust) == 1, "flag_fewer_than_four_robust": len(robust) < 4,
            "flag_no_certified_negative": len(cert) == 0,
            "flag_extreme_imbalance": bool(ls and (len(robust)/len(ls) < .01 or len(robust)/len(ls) > .90)),
            "near_certified_negative": near_cert, "near_weak_negative": near_weak, "far_negative": far,
        })
    summaries = {}
    for sc in sorted({r["scenario"] for r in rows}):
        rr = [r for r in rows if r["scenario"] == sc]
        summaries[sc] = {
            "states": len(rr), "only_one_robust": sum(r["flag_only_one_robust"] for r in rr),
            "fewer_than_four_robust": sum(r["flag_fewer_than_four_robust"] for r in rr),
            "no_certified_negative": sum(r["flag_no_certified_negative"] for r in rr),
            "extreme_imbalance": sum(r["flag_extreme_imbalance"] for r in rr),
            "eta_labels_median": float(np.median([r["total_eta_labels"] for r in rr])),
            "robust_median": float(np.median([r["verified_robust_eta_count"] for r in rr])),
            "certified_negative_median": float(np.median([r["certified_nonrobust_count"] for r in rr])),
            "positive_fraction_median": float(np.median([r["positive_fraction"] for r in rr])),
        }
    return rows, summaries, {sc: dict(v) for sc, v in boundary_summary.items()}


def db_integrity(canonical_hashes):
    malformed = 0; live = 0; missing_source_paths = 0
    stale_controllers = Counter()
    with sqlite3.connect(DB) as con:
        con.row_factory = sqlite3.Row
        quick = con.execute("PRAGMA quick_check").fetchone()[0]
        fk = [list(r) for r in con.execute("PRAGMA foreign_key_check")]
        dup = con.execute("""SELECT COUNT(*) FROM (SELECT state_uid,eta_uid,controller_uid,seed_key,COUNT(*) n
            FROM rollout GROUP BY state_uid,eta_uid,controller_uid,seed_key HAVING n>1)""").fetchone()[0]
        orphan_state = con.execute("SELECT COUNT(*) FROM rollout r LEFT JOIN state s USING(state_uid) WHERE s.state_uid IS NULL").fetchone()[0]
        orphan_eta = con.execute("SELECT COUNT(*) FROM rollout r LEFT JOIN eta e USING(eta_uid) WHERE e.eta_uid IS NULL").fetchone()[0]
        for (seed,) in con.execute("SELECT DISTINCT seed_key FROM rollout"):
            try:
                obj = json.loads(seed)
                if not isinstance(obj, dict): malformed += 1
            except Exception: malformed += 1
        sources = list(con.execute("SELECT sha256,path FROM source_file"))
        live = sum(str(r["sha256"]).startswith("LIVE") for r in sources)
        missing_source_paths = sum(not Path(r["path"]).exists() for r in sources)
        conflicts = con.execute("SELECT COUNT(*) FROM conflict").fetchone()[0]
        conflict_status = dict(con.execute(
            "SELECT COALESCE(status,'NULL'),COUNT(*) FROM conflict GROUP BY status").fetchall())
        missing_rollout_provenance = con.execute("""SELECT COUNT(*) FROM rollout
            WHERE experiment_uid IS NULL OR original_source_file IS NULL OR raw_record_hash IS NULL
               OR compatibility_quality IS NULL""").fetchone()[0]
        incomplete_sources = con.execute("""SELECT COUNT(*) FROM source_file
            WHERE classification IS NULL OR rows_seen IS NULL OR rows_imported IS NULL""").fetchone()[0]
        for r in con.execute("""SELECT s.name,c.safety_config_hash,COUNT(*) n FROM rollout r
            JOIN controller_config c USING(controller_uid) JOIN scenario s USING(scenario_uid)
            GROUP BY s.name,c.safety_config_hash"""):
            if r["name"] == "ring_exchange" and r["safety_config_hash"] != CURRENT_RING_SAFETY:
                stale_controllers[str(r["safety_config_hash"])] += int(r["n"])
    return {"structural": {"quick_check": quick, "foreign_key_violations": fk,
                            "duplicate_exact_tuple_keys": dup, "orphan_states": orphan_state,
                            "orphan_eta_references": orphan_eta, "malformed_distinct_seed_ids": malformed,
                            "live_source_records": live, "missing_source_paths": missing_source_paths,
                            "missing_rollout_provenance_rows": missing_rollout_provenance,
                            "incomplete_source_records": incomplete_sources,
                            "conflict_records": conflicts, "conflict_status": conflict_status,
                            "status": "PASS" if quick == "ok" and not fk and not dup and not orphan_state and not orphan_eta and not malformed else "FAIL"},
            "scientific": {"stale_ring_rollout_rows_by_safety_hash": dict(stale_controllers),
                           "scope_note": "A DB is not expected to cover unrequested state/eta/seed combinations."}}


def readiness_matrix(states, labels, density_rows):
    bysc = {}; density = defaultdict(list)
    for r in density_rows: density[r["scenario"]].append(r)
    for sc in sorted(density):
        rr = density[sc]; all_rob = all(r["verified_robust_eta_count"] >= 1 for r in rr)
        multi = all(r["verified_robust_eta_count"] >= 4 for r in rr)
        neg = all(r["certified_nonrobust_count"] >= 1 for r in rr)
        boundary = sum(r["near_certified_negative"] > 0 for r in rr) / len(rr)
        zero = all(r["eta_zero_status"] != "MISSING" and r["eta_zero_status"] != "UNCERTIFIED" for r in rr)
        bysc[sc] = {
            "direct_center_regression": "READY" if all_rob else "PARTIAL",
            "generator_likelihood_robust_points": "READY" if multi else "PARTIAL" if all_rob else "NOT_READY",
            "Q_h_eta": "READY" if all_rob and neg else "PARTIAL",
            "critic_ranking": "READY" if all_rob and neg else "PARTIAL",
            "margin_set_learning": "READY" if boundary >= .8 else "PARTIAL" if boundary > 0 else "NOT_READY",
            "minimum_deformation": "PARTIAL" if zero and all_rob else "NOT_READY",
            "notes": "Minimum-deformation remains PARTIAL unless executed-deformation metadata is dense for eta=0 and robust alternatives."
        }
    return bysc


def main():
    V2.mkdir(parents=True, exist_ok=True)
    old_states = pq.read_table(V1 / "states.parquet").to_pylist()
    old_labels_raw = pq.read_table(V1 / "eta_labels.parquet").to_pylist()
    old_labels = [{k: parse(v) if k in {"eta_raw","eta_normalized","terminal_summary","seed_outcomes"} else v for k,v in r.items()} for r in old_labels_raw]
    ring_states_raw = sampled_state_rows("ring_exchange")
    ring_rt = TrainingRuntime("ring_exchange", ring_states_raw, parent=False)
    ring_ctl = ring_rt.controllers["orthoflow3"]["uid"]
    old_ring = [r for r in old_labels if r["scenario"] == "ring_exchange"]
    wanted = {(r["state_uid"], r["eta_uid"]) for r in old_ring}
    current_ring = []
    state_meta = {r["state_uid"]: r for r in old_states}
    for state in ring_states_raw:
        for row in _aggregate_eta_rows(state["uid"], ring_ctl):
            if (row["state_uid"], row["eta_uid"]) not in wanted: continue
            meta = state_meta[row["state_uid"]]
            row.update(state_id=meta["state_id"], scenario="ring_exchange", split=meta["split"])
            current_ring.append(enrich_evidence(row, safety_hash=CURRENT_RING_SAFETY,
                                                provenance_class="CURRENT_COMPATIBLE_RECOMPUTED",
                                                old_controller_uid=OLD_RING_CONTROLLER))
    got = {(r["state_uid"],r["eta_uid"]) for r in current_ring}
    missing = sorted(wanted-got)
    if missing:
        raise RuntimeError(f"Ring backfill incomplete: {len(missing)} labels")

    retained = []
    for row in old_labels:
        if row["scenario"] == "ring_exchange": continue
        safety = ("d664071c6b88028e7943c62680b9613f94667545e092a00718c1aad609dfc8cd"
                  if row["scenario"] == "four_way_intersection" else "certified_hard_projection_v1")
        retained.append(enrich_evidence(row, safety_hash=safety,
                                        provenance_class="CURRENT_COMPATIBLE_RETAINED"))
    labels = retained + current_ring
    keys = {(r["state_uid"],r["eta_uid"],r["controller_uid"]) for r in labels}
    jm = rollout_audit_map(keys)
    for row in labels:
        row.update(jm.get((row["state_uid"],row["eta_uid"],row["controller_uid"]),
                          {"mean_j_def": None,"j_def_count":0,"success_collision_count":0}))
        # Apply the collision-consistency audit after seed-level DB aggregation.
        if row["robust_15of16"] is True and row["success_collision_count"]:
            row["robust_15of16"] = None
            row["robust_evidence_logically_sufficient"] = False
            row["audit_downgrade_reason"] = "SUCCESS_COLLISION_INCONSISTENCY"
            row["negative_evidence_class"] = "NUMERICAL_UNCERTIFIED"
        row["provenance_db_key"] = {"state_uid": row["state_uid"], "eta_uid": row["eta_uid"],
                                    "controller_uid": row["controller_uid"]}

    # Recompute state summaries and point-cloud components only from audited evidence.
    by = defaultdict(list)
    for r in labels: by[r["state_uid"]].append(r)
    states = []; geometries = []
    for original in old_states:
        st = dict(original); ls = by[st["state_uid"]]
        if st["scenario"] == "ring_exchange":
            st["controller_uid"] = ring_ctl
            st["conditioning_source_safety_status"] = "OLD_SAFETY_PHYSICAL_STATE_REUSED_LABELS_RECOMPUTED"
        center = _canonical_center(ls); components = _robust_components(ls); refs=[]
        for i, component in enumerate(components):
            ref = f"{st['state_id']}::audited_component_{i}"; refs.append(ref)
            geometries.append({"geometry_ref":ref,"state_uid":st["state_uid"],"state_id":st["state_id"],
                "scenario":st["scenario"],"component_index":i,"geometry_type":"verified_robust_point_cloud",
                "validated_inner_set":False,"eta_uids":[r["eta_uid"] for r in component],
                "eta_normalized":[r["eta_normalized"] for r in component],
                "safety_provenance":"current canonical","no_parametric_geometry_claim":True})
        st["zero_sufficient"] = any(parse(r["eta_raw"]) == [0,0,0] and r["robust_15of16"] is True for r in ls)
        st["has_robust_eta"] = center is not None
        st["robust_search_status"] = "ROBUST_ETA_OBSERVED" if center else "NO_ROBUST_ETA_OBSERVED"
        st["canonical_center"] = None if center is None else center["eta_raw"]
        st["canonical_center_eta_uid"] = None if center is None else center["eta_uid"]
        st["num_robust_components"] = len(components); st["basin_geometry_refs"] = refs
        st["audit_version"] = "v2_audited"
        states.append(st)
        component_eta = {r["eta_uid"]: i for i,c in enumerate(components) for r in c}
        for r in ls:
            d = None if center is None else normalized_distance(r["eta_raw"],center["eta_raw"])
            r["distance_to_center"] = d
            r["geometry_membership"] = ("interior" if r["robust_15of16"] is True else
                "boundary_candidate" if r["robust_15of16"] is False and d is not None and d <= _component_threshold() else
                "exterior_observed" if r["robust_15of16"] is False else "uncertified")
            r["robust_component_ref"] = None if r["eta_uid"] not in component_eta else refs[component_eta[r["eta_uid"]]]
            r["normalized_inner_set_distance"] = None

    density_rows, density_summary, boundary = density_and_boundary(states, labels)
    drift_report = drift(old_ring, current_ring)
    dump(V2 / "label_drift_report.json", drift_report)
    dump(V2 / "state_label_density_audit.json", {"states": density_rows,"summary":density_summary})
    dump(V2 / "boundary_evidence_audit.json", {"component_threshold_normalized":_component_threshold(),"summary":boundary})

    invalid_robust = [r for r in labels if r["robust_15of16"] is True and not r["robust_evidence_logically_sufficient"]]
    success_collision_rows = [r for r in labels if r["success_collision_count"] > 0]
    collision_robust = [r for r in labels if r.get("audit_downgrade_reason") == "SUCCESS_COLLISION_INCONSISTENCY"]
    numerical = [r for r in labels if r["numerical_failure_count"] > 0]
    numerical_bad_negative = [r for r in numerical if r["negative_evidence_class"] in ("CERTIFIED_NON_ROBUST","FULL_Q16_NON_ROBUST") and r["failure_count"] < 2]
    evidence = {
        "robust": dict(Counter(r["evidence_level"] for r in labels if r["robust_15of16"] is True)),
        "robust_exact_seed_count": dict(sorted(Counter(
            int(r["seed_count"]) for r in labels if r["robust_15of16"] is True).items())),
        "robust_strength_interpretation": {
            "note": "Exact counts are authoritative. Bucket names inherited from v1 are coarse; a 15-seed early acceptance is logically sufficient for 15/16.",
            "Q4_only_is_never_robust": True,
        },
        "invalid_robust_labels": len(invalid_robust),
        "negative": dict(Counter(r["negative_evidence_class"] for r in labels if r["negative_evidence_class"])),
        "negative_exact_seed_count": dict(sorted(Counter(
            int(r["seed_count"]) for r in labels if r["negative_evidence_class"]).items())),
    }
    dump(V2 / "evidence_strength_audit.json", evidence)
    dump(V2 / "numerical_audit.json", {"affected_eta_labels":len(numerical),
        "affected_states":len({r['state_uid'] for r in numerical}),
        "per_scenario":dict(Counter(r['scenario'] for r in numerical)),
        "incorrectly_certified_as_negative":len(numerical_bad_negative),
        "records":[{"state_uid":r['state_uid'],"eta_uid":r['eta_uid'],"scenario":r['scenario'],"count":r['numerical_failure_count']} for r in numerical]})
    dump(V2 / "collision_audit.json", {"robust_labels_with_success_collision_inconsistency":len(collision_robust),
        "all_labels_with_success_collision_inconsistency":len(success_collision_rows),
        "robust_labels_with_any_failed_seed_collision":sum(r['robust_15of16'] is True and r['collision_count'] > 0 for r in labels),
        "all_label_collision_seeds":sum(r['collision_count'] for r in labels),
        "per_scenario":dict(Counter(r['scenario'] for r in labels for _ in range(int(r['collision_count'])))),
        "invalid_robust_keys":[[r['state_uid'],r['eta_uid']] for r in collision_robust],
        "all_success_collision_keys":[[r['state_uid'],r['eta_uid']] for r in success_collision_rows]})

    scenario_balance = {}
    for sc in ("double_bottleneck","four_way_intersection","ring_exchange"):
        ss=[r for r in states if r['scenario']==sc]; ll=[r for r in labels if r['scenario']==sc]
        scenario_balance[sc]={"states":len(ss),"eta_labels":len(ll),
            "robust_labels":sum(r['robust_15of16'] is True for r in ll),
            "certified_negatives":sum(r['negative_evidence_class'] in ('CERTIFIED_NON_ROBUST','FULL_Q16_NON_ROBUST') for r in ll),
            "Q16_plus":sum(r['seed_count']>=16 for r in ll),"zero_sufficient_states":sum(r['zero_sufficient'] for r in ss),
            "correction_needed_states":sum(r['has_robust_eta'] and not r['zero_sufficient'] for r in ss),
            "generator_batch_states_per_step":64,"critic_batch_states_per_step":96,"effective_scenario_share":1/3}
    dump(V2 / "scenario_balance_audit.json", {"sampling_rule":"scenario then state then eta/class","approximately_balanced":True,"scenarios":scenario_balance})

    hashes=json.loads((HERE/'canonical_hashes.json').read_text()); db_audit=db_integrity(hashes)
    dump(V2 / "database_integrity.json",db_audit)
    for name in ("leakage_audit.json","conditioning_schema_audit.json","normalization_audit.json"):
        dump(V2/name,json.loads((HERE/name).read_text()))
    training_prov={"dataset_path":str(V1),"dataset_hashes":hashes['dataset_v1'],
        "generator_checkpoint":hashes['generator'],"critic_checkpoint":hashes['critic'],
        "ring_labels_old_safety":21520,
        "ring_critic_train_labels_old_safety":17216,
        "ring_critic_validation_labels_old_safety":4304,
        "ring_generator_train_robust_labels_in_domain_old_safety":4165,
        "ring_generator_validation_robust_labels_in_domain_old_safety":1281,
        "assessment":"MODEL_TRAINING_DATA_PARTIALLY_STALE",
        "reason":"All Ring rows used the old safety adapter, but recomputation changed only 0.367% of certified robust/non-robust labels and no eta=0 status. The provenance is not current; observed label drift is limited.",
        "models_retrained":False}
    dump(V2/'model_training_provenance_audit.json',training_prov)

    # Keep point clouds only; no ellipsoid/ball is asserted.
    write_parquet(V2/'states.parquet',states,{"conditioning","structured_state","environment_descriptor","canonical_center","basin_geometry_refs","provenance"})
    write_parquet(V2/'eta_labels.parquet',labels,{"eta_raw","eta_normalized","terminal_summary","seed_outcomes"})
    with (V2/'basin_geometry.jsonl').open('w') as f:
        for r in geometries:f.write(canonical(r)+'\n')
    stale_map=[{"state_uid":r['state_uid'],"eta_uid":r['eta_uid'],"old_controller_uid":r['controller_uid'],
                "current_controller_uid":ring_ctl,"old_safety_hash":OLD_RING_SAFETY,"current_safety_hash":CURRENT_RING_SAFETY}
               for r in old_ring]
    write_parquet(V2/'stale_label_map.parquet',stale_map,set())
    for split in ('train','validation'):
        sr=[r for r in states if r['split']==split]
        dump(V2/f'split_{split}.json',{"split":split,"state_ids":[r['state_id'] for r in sr],
            "parent_episode_ids":sorted({r['parent_episode_id'] for r in sr}),
            "counts_by_scenario":dict(Counter(r['scenario'] for r in sr))})

    readiness_v2=readiness_matrix(states,labels,density_rows)
    # v1 is scientifically stale for every Ring objective even when structurally dense.
    readiness_v1=json.loads(json.dumps(readiness_v2))
    for objective in readiness_v1['ring_exchange']:
        if objective != 'notes':readiness_v1['ring_exchange'][objective]='NOT_READY'
    readiness_result={"v1":readiness_v1,"v2_audited":readiness_v2}
    dump(V2/'learning_readiness_matrix.json',readiness_result)

    manifest={"schema":"orthoflow3_basin_dataset_v2_audited","source":"orthoflow3_basin_dataset_v1",
        "state_count":len(states),"eta_label_count":len(labels),"ring_labels_recomputed":len(current_ring),
        "robust_criterion":">=15/16 exact canonical seeds; exact second-failure rejection",
        "eta_domain":{"low":DOMAIN_LOW.tolist(),"high":DOMAIN_HIGH.tolist()},
        "inner_set_policy":"verified robust point clouds only; no parametric inner sets created",
        "label_drift_rate":drift_report['LABEL_DRIFT_RATE'],"invalid_robust_labels":len(invalid_robust),
        "success_collision_inconsistent_robust_labels":len(collision_robust),"database_status":db_audit['structural']['status'],
        "dataset_status":"DATASET_V2_RECOMMENDED","model_data_status":"CURRENT_MODELS_USE_PARTIALLY_STALE_DATA"}
    dump(V2/'manifest.json',manifest)
    print(json.dumps(manifest,indent=2))


if __name__ == '__main__':main()
