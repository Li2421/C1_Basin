"""A second compatible future-Flow intervention on the frozen source pairs.

This adds a third controller condition at the same physical state and eta.
The first action, eta basis, plant, hard-safety and success semantics are
unchanged. Target K16 labels are never read by this module.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from . import intervention as base

OUT = Path(__file__).resolve().parent / "second_variant"
EXPERIMENT = "exp_orthoflow3_controller_intervention_second_variant_v1"
ALT = {
    "double_bottleneck": base.ROOT / "diagnostics/double_bottleneck_recovery_density_final/model/ckpt_primary_87_5pct.pkl",
    "four_way_intersection": base.ROOT / "diagnostics/four_way_intersection_stage1/base_u_v10_world_timeout_goal_recovery_source_balanced_wide_macflow/best.pkl",
    "ring_exchange": base.ROOT / "diagnostics/ring_exchange_stage1/base_u_v7_local_macflow/best.pkl",
}


def setup():
    base.OUT = OUT
    base.EXPERIMENT = EXPERIMENT


def prepare():
    from shared_rollout_db.src.rollout_db import canonical, connect, transaction, uid

    setup()
    if (OUT / "protocol.json").exists():
        print("second controller variant already frozen")
        return
    previous = base.read(OUT.parent / "balanced_expansion/pairs.json")
    first = base.read(OUT.parent / "protocol.json")
    profiles = {}
    with connect() as db, transaction(db):
        for scene, path in ALT.items():
            reference = first["profiles"][scene]
            checkpoint = pickle.load(path.open("rb"))
            assert checkpoint["config"]["environment_fingerprint"] == reference["environment_fingerprint"]
            digest = base.digest(path)
            assert digest not in (reference["base_flow_sha256"], reference["alternate_flow_sha256"])
            old = db.execute("SELECT * FROM controller_config WHERE controller_uid=?",
                             (reference["base_controller_uid"],)).fetchone()
            assert old is not None and old["compatibility_quality"] == "EXACT_PROFILE"
            payload = json.loads(old["config_json"])
            payload.update(flow_checkpoint_sha256=digest,
                           committed_t0_flow_sha256=reference["base_flow_sha256"],
                           conditioning=str(payload.get("conditioning", "")) +
                           "+committed_base_t0_then_alt_future_v1",
                           intervention_runtime_sha256=base.digest(base.__file__))
            new_uid = uid("ctl", payload)
            db.execute("""INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,
                flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,
                success_semantics_version,conditioning_version,rng_semantics_version,config_json,
                compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (new_uid, old["scenario_uid"], digest, old["orthoflow3_sha256"],
                 old["safety_config_hash"], old["horizon"], old["dt"],
                 old["success_semantics_version"], payload["conditioning"],
                 old["rng_semantics_version"], canonical(payload), "EXACT_PROFILE"))
            profiles[scene] = {**reference, "alternate_controller_uid": new_uid,
                               "alternate_flow_sha256": digest, "alternate_path": str(path)}
        db.execute("""INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,
            code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)""",
            (EXPERIMENT, "controller_intervention_second_variant_v1", str(OUT),
             base.digest(__file__), base.digest(base.__file__),
             canonical({"source_only": True, "same_state_eta_as_first_variant": True})))
    pairs = [{**r, "alternate_controller_uid": profiles[r["scene"]]["alternate_controller_uid"]}
             for r in previous]
    requests = [{"state_uid": r["state_uid"], "eta_uid": r["eta_uid"],
                 "controller_uid": r["alternate_controller_uid"],
                 "seed_keys": [canonical({"future_index": i}) for i in range(16)]}
                for r in pairs]
    protocol = {"schema": "source_only_second_compatible_future_flow_v1",
                "experiment_uid": EXPERIMENT,
                "intervention": first["intervention"],
                "source_only": True, "same_state_eta_as_first_variant": True,
                "controller_selection": "preregistered compatible earlier frozen checkpoints; no outcome selection",
                "environment_fingerprint_checked": True,
                "safety_or_success_changed": False, "basis_changed": False,
                "target_frozen_K16_untouched": True,
                "profiles": profiles,
                "code_sha256": base.digest(base.__file__),
                "wrapper_sha256": base.digest(__file__),
                "parent_pairs_sha256": base.digest(OUT.parent / "balanced_expansion/pairs.json")}
    base.write("protocol.json", protocol)
    base.write("pairs.json", pairs)
    base.write("planned_rollouts.json", {"requests": requests})
    print(json.dumps({"states": len({r["state_uid"] for r in pairs}),
                      "pairs": len(pairs), "requested": len(pairs) * 16}))


def analyze():
    from shared_rollout_db.src.rollout_db import connect
    from collections import defaultdict
    from itertools import combinations
    import csv

    setup()
    rows = []
    with connect(True) as db:
        for r in base.read(OUT / "pairs.json"):
            rec = db.execute("""SELECT success,numerical_failure,seed_key FROM rollout
                WHERE state_uid=? AND eta_uid=? AND controller_uid=?
                AND conflict_quarantined=0 AND compatibility_quality='EXACT_REUSE'""",
                (r["state_uid"], r["eta_uid"], r["alternate_controller_uid"])).fetchall()
            seeds = {json.loads(x["seed_key"]).get("future_index"): x for x in rec}
            valid = [seeds[i] for i in range(16) if i in seeds and not seeds[i]["numerical_failure"]]
            successes = sum(x["success"] for x in valid)
            rows.append({**r, "second_valid": len(valid), "second_success": successes,
                         "second_B15": len(valid) == 16 and successes >= 15,
                         "second_nonB15": len(valid) == 16 and successes < 15})
    with (OUT / "intervention_pairs.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    by = defaultdict(list)
    for r in rows:
        if r["second_valid"] == 16:
            by[(r["scene"], r["state_uid"])].append(r)
    summary = {}
    for scene in ALT:
        chosen = [r for r in rows if r["scene"] == scene]
        strong_reversals = 0
        certified_interval_reversals = 0
        for (sc, _), group in by.items():
            if sc != scene:
                continue
            for a, b in combinations(group, 2):
                a_lo = int(a["base_observed_success"]) / 16
                a_hi = (16 - int(a["base_observed_failure"])) / 16
                b_lo = int(b["base_observed_success"]) / 16
                b_hi = (16 - int(b["base_observed_failure"])) / 16
                second_delta = (a["second_success"] - b["second_success"]) / 16
                certified_interval_reversals += int(
                    (a_lo > b_hi and second_delta <= -.25) or
                    (b_lo > a_hi and second_delta >= .25))
                # The canonical source can be partial-count. A strict reversal
                # is certified only if both canonical pairs are full Q16.
                if not a["base_full_Q16"] or not b["base_full_Q16"]:
                    continue
                d0 = float(a["base_Q16"]) - float(b["base_Q16"])
                d1 = (a["second_success"] - b["second_success"]) / 16
                strong_reversals += int(d0 * d1 < 0 and abs(d0) >= .25 and abs(d1) >= .25)
        summary[scene] = {"pairs": len(chosen),
                          "full_Q16": sum(r["second_valid"] == 16 for r in chosen),
                          "numerical_unresolved": sum(r["second_valid"] < 16 for r in chosen),
                          "B15": sum(r["second_B15"] for r in chosen),
                          "strong_canonical_to_second_rank_reversals": strong_reversals,
                          "certified_interval_rank_reversals": certified_interval_reversals}
    base.write("summary.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "merge", "analyze"))
    parser.add_argument("--scene", choices=base.SCENES)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "analyze": analyze()
    else:
        setup()
        if args.action == "run": base.run(args.scene, args.shard, args.shards)
        else: base.merge()
