"""Source-only matched intervention expansion with partial-count negatives.

Only the alternate controller is newly executed. Canonical non-B15 labels
remain observed (s,f) intervals, never fabricated full Q16 outcomes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from . import intervention as base

OUT = Path(__file__).resolve().parent / "balanced_expansion"
EXPERIMENT = "exp_orthoflow3_controller_intervention_balanced_v1"


def setup():
    base.OUT = OUT
    base.EXPERIMENT = EXPERIMENT
    # The runtime/controller semantics are deliberately identical to the
    # pilot. Reuse that controller UID and the frozen runtime-code hash.


def read(path):
    return json.loads(Path(path).read_text())


def write(name, obj):
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def rank(value):
    return hashlib.sha256(("balanced-intervention-v1|" + str(value)).encode()).hexdigest()


def prepare():
    from shared_rollout_db.src.rollout_db import canonical, connect, transaction

    setup()
    if (OUT / "protocol.json").exists():
        print("balanced expansion already frozen")
        return
    pilot = read(OUT.parent / "protocol.json")
    states = read(base.OLD / "states.json")
    rows = pq.read_table(base.OLD / "all_pairs.parquet").to_pylist()
    by_state = defaultdict(list)
    for r in rows:
        if r["scenario"] in base.SCENES:
            by_state[r["state_index"]].append(r)
    chosen = []
    for scene in base.SCENES:
        for split, limit in (("train", 12), ("validation", 4)):
            available = []
            for i, state in enumerate(states):
                if state["scenario"] != scene or state["split"] != split:
                    continue
                z = by_state[i]
                pos = [r for r in z if r["full_standard_16"] and r["standard_success"] >= 15]
                neg = [r for r in z if r["standard_failure"] >= 2]
                if len(pos) >= 2 and len(neg) >= 2:
                    available.append((i, state, pos, neg))
            available.sort(key=lambda x: rank(x[1]["family"]))
            assert len(available) >= limit, (scene, split, len(available))
            for i, state, pos, neg in available[:limit]:
                positives = sorted(pos, key=lambda x: rank(x["eta_uid"]))[:2]
                # One early-clear failure plus one near-B15/late failure if
                # present. Both are selected on source TRAIN/VAL only.
                neg_low = min(neg, key=lambda x: (x["n"], rank(x["eta_uid"])))
                neg_high = max((x for x in neg if x["eta_uid"] != neg_low["eta_uid"]),
                               key=lambda x: (x["n"], rank(x["eta_uid"])))
                assert neg_low["eta_uid"] != neg_high["eta_uid"]
                for r in positives + [neg_low, neg_high]:
                    chosen.append({"scene": scene, "split": split, "state_index": i,
                                   "state_uid": state["state_uid"], "family": state["family"],
                                   "eta": r["eta"], "eta_uid": r["eta_uid"],
                                   "base_controller_uid": r["controller_uid"],
                                   "alternate_controller_uid": pilot["profiles"][scene]["alternate_controller_uid"],
                                   "base_observed_success": r["standard_success"],
                                   "base_observed_failure": r["standard_failure"],
                                   "base_observed_trials": r["standard_observed"],
                                   "base_B15": bool(r["full_standard_16"] and r["standard_success"] >= 15),
                                   "base_nonB15": bool(r["standard_failure"] >= 2),
                                   "base_full_Q16": bool(r["full_standard_16"]),
                                   "base_Q16": r["Q16"], "base_stopping_reason": r["stopping_reason"]})
    assert len(chosen) == 192
    requests = [{"state_uid": r["state_uid"], "eta_uid": r["eta_uid"],
                 "controller_uid": r["alternate_controller_uid"],
                 "seed_keys": [canonical({"future_index": i}) for i in range(16)]}
                for r in chosen]
    protocol = {"schema": "source_only_balanced_partial_intervention_v1",
                "experiment_uid": EXPERIMENT,
                "intervention": pilot["intervention"],
                "source_only": True, "target_scene_labels_used_for_training_or_selection": False,
                "scene_train_states": 12, "scene_val_states": 4, "eta_per_state": 4,
                "eta_selection": "two full-Q16 B15, one early non-B15, one late non-B15; source split only",
                "base_partial_count_policy": "retain observed s,f; no unrun seed imputation",
                "profiles": pilot["profiles"], "pilot_protocol_sha256": base.digest(OUT.parent / "protocol.json"),
                "source_pairs_sha256": base.digest(base.OLD / "all_pairs.parquet"),
                "code_sha256": base.digest(base.__file__),
                "experiment_module_sha256": base.digest(__file__),
                "safety_or_success_changed": False, "generator_changed": False,
                "target_frozen_K16_untouched": True}
    write("protocol.json", protocol)
    write("pairs.json", chosen)
    write("planned_rollouts.json", {"requests": requests})
    with connect() as db, transaction(db):
        db.execute("""INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,
            code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)""",
            (EXPERIMENT, "balanced_controller_intervention_v1", str(OUT),
             base.digest(OUT / "protocol.json"), base.digest(__file__),
             canonical({"source_only": True, "partial_base_counts": True})))
    print(json.dumps({"states": 48, "pairs": len(chosen), "requested_new_controller_slots": len(chosen) * 16}))


def analyze():
    from shared_rollout_db.src.rollout_db import connect

    result = []
    with connect(True) as db:
        for r in read(OUT / "pairs.json"):
            got = db.execute("""SELECT success,numerical_failure,seed_key FROM rollout WHERE state_uid=?
                AND eta_uid=? AND controller_uid=? AND conflict_quarantined=0
                AND compatibility_quality='EXACT_REUSE'""",
                (r["state_uid"], r["eta_uid"], r["alternate_controller_uid"])).fetchall()
            seeds = {json.loads(x["seed_key"]).get("future_index"): x for x in got}
            valid = [seeds[i] for i in range(16) if i in seeds and not seeds[i]["numerical_failure"]]
            s = sum(x["success"] for x in valid)
            result.append({**r, "alternate_valid": len(valid), "alternate_success": s,
                           "alternate_Q16_lower": s / 16,
                           "alternate_Q16_upper": (s + 16 - len(valid)) / 16,
                           "alternate_B15": len(valid) == 16 and s >= 15,
                           "alternate_nonB15": len(valid) == 16 and s < 15,
                           "base_Q16_lower": r["base_observed_success"] / 16,
                           "base_Q16_upper": (16 - r["base_observed_failure"]) / 16})
    with (OUT / "intervention_pairs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result[0]))
        writer.writeheader();writer.writerows(result)
    summary = {"pairs": len(result), "source_states": 48,
               "observed_new_records": sum(r["alternate_valid"] for r in result),
               "by_scene": {scene: {"pairs": sum(r["scene"] == scene for r in result),
                                     "base_positive": sum(r["scene"] == scene and r["base_B15"] for r in result),
                                     "base_negative": sum(r["scene"] == scene and r["base_nonB15"] for r in result),
                                     "negative_to_B15": sum(r["scene"] == scene and r["base_nonB15"] and r["alternate_B15"] for r in result),
                                     "positive_to_nonB15": sum(r["scene"] == scene and r["base_B15"] and r["alternate_nonB15"] for r in result),
                                     "numerical_unresolved": sum(r["scene"] == scene and r["alternate_valid"] < 16 for r in result)}
                            for scene in base.SCENES}}
    write("summary.json", summary)
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
