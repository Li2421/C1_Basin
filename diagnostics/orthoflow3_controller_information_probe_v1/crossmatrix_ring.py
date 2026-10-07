"""Outcome-blind shared-eta source TRAIN panel for a controller-data support test.

Only the existing Ring source TRAIN states and the first two compatible Flow
conditions enter training. The third controller and source VAL families remain
label-held-out in this experiment. This module prepares the cache manifest;
execution and merger are separate audited steps.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from shared_rollout_db.src.rollout_db import canonical

from .probe import OUT, SRC, read, write


DEST = OUT / "crossmatrix_ring_v2"
EXPERIMENT = "exp_orthoflow3_controller_info_crossmatrix_ring_v2"


def prepare():
    dest = DEST
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "protocol.json").exists():
        raise FileExistsError("Cross-matrix protocol already frozen")
    parent = [r for r in read(SRC / "balanced_expansion/pairs.json")
              if r["scene"] == "ring_exchange" and r["split"] == "train"]
    by_uid = {r["eta_uid"]: r["eta"] for r in parent}
    counts = Counter(r["eta_uid"] for r in parent)
    selected = [uid for uid, _ in sorted(counts.items(), key=lambda z: (-z[1], z[0]))[:4]]
    states = {r["state_uid"]: r for r in parent}
    assert len(states) == 12 and len(selected) == 4
    base_ids = {r["base_controller_uid"] for r in parent}
    first_ids = {r["alternate_controller_uid"] for r in parent}
    assert len(base_ids) == len(first_ids) == 1
    pairs = []
    for state_uid in sorted(states):
        template = states[state_uid]
        for eta_uid in selected:
            pairs.append({"scene": "ring_exchange", "split": "train", "state_uid": state_uid,
                          "state_index": template["state_index"], "family": template["family"],
                          "eta_uid": eta_uid, "eta": by_uid[eta_uid],
                          "base_controller_uid": template["base_controller_uid"],
                          "alternate_controller_uid": template["alternate_controller_uid"]})
    requests = [{"state_uid": p["state_uid"], "eta_uid": p["eta_uid"],
                 "controller_uid": p[key],
                 "seed_keys": [canonical({"future_index": i}) for i in range(8)]}
                for p in pairs for key in ("base_controller_uid", "alternate_controller_uid")]
    profile = read(SRC / "balanced_expansion/protocol.json")["profiles"]["ring_exchange"]
    assert next(iter(first_ids)) == profile["alternate_controller_uid"]
    write(dest / "pairs.json", pairs)
    write(dest / "planned_rollouts.json", {"requests": requests})
    write(dest / "protocol.json", {
        "experiment_uid": EXPERIMENT, "scene": "ring_exchange",
        "source_TRAIN_only": True, "heldout_source_family_VAL_states": 4,
        "heldout_third_Flow_labels_used_for_training": False,
        "controller_conditions": ["base", "first_compatible_future_Flow"],
        "eta_selection": "four most frequent exact eta UIDs among existing source TRAIN pairs, frequency then UID tie-break; no outcome ranking or VAL labels",
        "selected_eta_uid": selected, "selected_eta_existing_pair_frequency": [counts[x] for x in selected],
        "state_count": len(states), "eta_count": len(selected),
        "seed_keys": [canonical({"future_index": i}) for i in range(8)],
        "requested_continuations": len(requests) * 8,
        "base_controller_uid": next(iter(base_ids)),
        "alternate_controller_uid": next(iter(first_ids)),
        "first_variant_profile": profile,
        "safety_success_basis_semantics_unchanged": True,
        "cross_scene_target_TEST_untouched": True,
    })
    print(json.dumps({"pairs": len(pairs), "requests": len(requests),
                      "requested_continuations": len(requests) * 8,
                      "selected_eta_existing_pair_frequency": [counts[x] for x in selected]}))


def register():
    from shared_rollout_db.src.rollout_db import connect, transaction
    protocol = read(DEST / "protocol.json")
    assert protocol["experiment_uid"] == EXPERIMENT
    with connect() as db, transaction(db):
        db.execute("""INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,
            code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)""",
            (EXPERIMENT, "controller_info_crossmatrix_ring_v1", str(DEST),
             hashlib.sha256((DEST / "protocol.json").read_bytes()).hexdigest(),
             hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             canonical({"source_train_only": True,
                        "new_continuations_preflight": read(DEST / "cache_preflight.json")["summary"]["genuinely_missing"],
                        "third_controller_labels_heldout": True})))
    print(json.dumps({"experiment_registered": EXPERIMENT}))


def run(shard, shards):
    from diagnostics.orthoflow3_controller_intervention_generalization_v1 import intervention as parent
    from new_benchmark_common import basin_dataset_v1 as bd
    from shared_rollout_db.src.cache_writer import append_journal
    from shared_rollout_db.src.rollout_db import eta_identity

    protocol = read(DEST / "protocol.json")
    profile = protocol["first_variant_profile"]
    assert parent.digest(profile["alternate_path"]) == profile["alternate_flow_sha256"]
    assert profile["base_controller_uid"] == protocol["base_controller_uid"]
    assert profile["alternate_controller_uid"] == protocol["alternate_controller_uid"]
    pairs = read(DEST / "pairs.json")
    assert all(r["split"] == "train" for r in pairs)
    preflight = read(DEST / "cache_preflight.json")
    missing = {(r["state_uid"], r["eta_uid"], r["controller_uid"], seed)
               for r in preflight["details"] for seed in r["missing_seeds"]}
    jobs = []
    for row in pairs:
        for controller_uid in (row["base_controller_uid"], row["alternate_controller_uid"]):
            for seed in range(8):
                if (row["state_uid"], row["eta_uid"], controller_uid,
                    canonical({"future_index": seed})) in missing:
                    jobs.append((row, controller_uid, seed))
    assert len(jobs) == preflight["summary"]["genuinely_missing"]
    jobs = [job for i, job in enumerate(jobs) if i % shards == shard]
    parent.OUT = DEST
    alt_runtime, states, reset_alt = parent._runtime("ring_exchange", profile)
    base_runtime = bd.TrainingRuntime("ring_exchange", [], parent=False)
    assert base_runtime.controllers["orthoflow3"]["uid"] == protocol["base_controller_uid"]
    completed = 0
    for row, controller_uid, seed in jobs:
        eta = np.asarray(row["eta"], float)
        assert eta_identity(eta)[0] == row["eta_uid"]
        state = states[row["state_uid"]]
        if controller_uid == protocol["alternate_controller_uid"]:
            reset_alt()
            result = alt_runtime.rollout(state, eta, seed, "orthoflow3")
            result["controller_uid"] = controller_uid
            result["flow_checkpoint_sha256"] = profile["alternate_flow_sha256"]
            result["committed_t0_flow_sha256"] = profile["base_flow_sha256"]
            result["intervention"] = "canonical first action then first alternate future Flow"
        else:
            result = base_runtime.rollout(state, eta, seed, "orthoflow3")
            assert result["controller_uid"] == controller_uid
        assert result["state_uid"] == row["state_uid"]
        assert eta_identity(result["eta"])[0] == row["eta_uid"]
        result["experiment_uid"] = EXPERIMENT
        append_journal([{"schema": "controller_info_crossmatrix_ring_v2", "record": result}],
                       EXPERIMENT, f"ring_shard{shard}")
        completed += 1
        if completed % 16 == 0:
            print(json.dumps({"shard": shard, "completed": completed}), flush=True)
    print(json.dumps({"shard": shard, "complete": completed}), flush=True)


def merge():
    from shared_rollout_db.src.rollout_db import connect, eta_identity, transaction, uid
    protocol = read(DEST / "protocol.json")
    allowed = {(r["state_uid"], r["eta_uid"], c)
               for r in read(DEST / "pairs.json")
               for c in (r["base_controller_uid"], r["alternate_controller_uid"])}
    counts = Counter()
    journal = Path(__file__).resolve().parents[2] / "shared_rollout_db/journals" / EXPERIMENT
    paths = sorted(journal.glob("*.jsonl"))
    with connect() as db:
        for path in paths:
            lines = path.read_text().splitlines()
            source_uid = uid("src", {"path": str(path.resolve())})
            with transaction(db):
                db.execute("""INSERT OR IGNORE INTO source_file(source_uid,experiment_uid,path,sha256,
                    file_type,classification,rows_seen) VALUES(?,?,?,?,?,?,?)""",
                    (source_uid, EXPERIMENT, str(path.resolve()),
                     hashlib.sha256(path.read_bytes()).hexdigest(), ".jsonl", "SEED_EXACT", len(lines)))
                for line_number, line in enumerate(lines, 1):
                    item = json.loads(line)
                    assert item["schema"] == "controller_info_crossmatrix_ring_v2"
                    row = item["record"]
                    eta_uid = eta_identity(row["eta"])[0]
                    key = row["state_uid"], eta_uid, row["controller_uid"]
                    assert key in allowed and 0 <= row["future_index"] < 8
                    seed_key = canonical({"future_index": row["future_index"]})
                    rollout_uid = uid("roll", {"state": key[0], "eta": key[1],
                                              "controller": key[2], "seed": seed_key})
                    core = tuple(int(row[k]) for k in
                                 ("success", "deadlock", "timeout", "collision", "numerical_failure"))
                    old = db.execute("SELECT * FROM rollout WHERE rollout_uid=?", (rollout_uid,)).fetchone()
                    if old:
                        previous = tuple(old[k] for k in
                                         ("success", "deadlock", "timeout", "collision", "numerical_failure"))
                        if core != previous:
                            db.execute("UPDATE rollout SET conflict_quarantined=1 WHERE rollout_uid=?", (rollout_uid,))
                            counts["conflicts"] += 1
                            continue
                        counts["duplicates"] += 1
                    else:
                        db.execute("""INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,
                            continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,
                            episode_length,j_def,min_wall_distance,min_agent_distance,outcome,experiment_uid,
                            original_source_file,timestamp,compatibility_quality,raw_record_hash)
                            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (rollout_uid, *key, seed_key, seed_key, *core,
                             row.get("episode_length"), row.get("J_def"),
                             row.get("minimum_wall_clearance"), row.get("minimum_agent_clearance"),
                             row.get("outcome"), EXPERIMENT, str(path), row.get("timestamp"),
                             "EXACT_REUSE", hashlib.sha256(canonical(row).encode()).hexdigest()))
                        counts["inserted"] += 1
                    db.execute("INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)",
                               (rollout_uid, source_uid, line_number))
    write(DEST / "merge_audit.json", {"journals": len(paths), **counts})
    print(json.dumps({"journals": len(paths), **counts}))


def materialize():
    from shared_rollout_db.src.rollout_db import connect
    old = {(r["state_uid"], r["eta_uid"])
           for r in read(SRC / "balanced_expansion/pairs.json")
           if r["scene"] == "ring_exchange" and r["split"] == "train"}
    selected = [r for r in read(DEST / "pairs.json") if (r["state_uid"], r["eta_uid"]) not in old]
    output = []
    with connect(True) as db:
        for pair in selected:
            item = {"state_uid": pair["state_uid"], "state_index": pair["state_index"],
                    "eta_uid": pair["eta_uid"], "eta": pair["eta"]}
            for index, key in enumerate(("base_controller_uid", "alternate_controller_uid")):
                records = db.execute("""SELECT seed_key,success,numerical_failure,conflict_quarantined,
                    compatibility_quality FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=?""",
                    (pair["state_uid"], pair["eta_uid"], pair[key])).fetchall()
                valid = [r for r in records if r["seed_key"] in
                         {canonical({"future_index": i}) for i in range(8)}
                         and not r["numerical_failure"] and not r["conflict_quarantined"]
                         and r["compatibility_quality"] == "EXACT_REUSE"]
                item[f"controller{index}_success"] = sum(int(r["success"]) for r in valid)
                item[f"controller{index}_failure"] = len(valid) - item[f"controller{index}_success"]
                item[f"controller{index}_observed"] = len(valid)
            item["controller2_success"] = item["controller2_failure"] = 0
            item["controller2_observed"] = 0
            output.append(item)
    write(DEST / "pair_counts.json", {"new_pairs_only": output,
                                     "new_pair_count": len(output),
                                     "observed_trial_count": sum(r[f"controller{i}_observed"]
                                                                 for r in output for i in (0, 1)),
                                     "third_controller_label_used": False,
                                     "frozen_VAL_pair_count_unchanged": True})
    print(json.dumps({"new_pairs_only": len(output),
                      "observed_trials": sum(r[f"controller{i}_observed"]
                                             for r in output for i in (0, 1))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "register", "run", "merge", "materialize"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "register": register()
    elif args.action == "run": run(args.shard, args.shards)
    elif args.action == "merge": merge()
    else: materialize()
