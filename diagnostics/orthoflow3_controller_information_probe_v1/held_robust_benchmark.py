"""Prospective, score-blind Ring held-controller B15 selection benchmark.

The state/eta design is frozen before the new Flow checkpoint is opened. The
Flow is independently trained from existing expert transitions (no eta task
labels). No held-controller task outcome enters model or candidate selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from diagnostics.orthoflow3_controller_intervention_generalization_v1 import intervention as base
from shared_rollout_db.src.rollout_db import canonical, connect, transaction, uid

from .probe import OUT, SRC, read, write


DEST = OUT / "held_controller_robust_selection_v1"
FLOW = OUT / "held_controller_v11_flow/best.pkl"
EXPERIMENT = "exp_orthoflow3_ring_held_controller_robust_selection_v1"
SCENE = "ring_exchange"


def rank(value):
    return hashlib.sha256(("held-ring-b15-v1|" + str(value)).encode()).hexdigest()


def design():
    if (DEST / "design.json").exists():
        raise FileExistsError("Held-controller state/eta design is already frozen")
    old = [r for r in read(SRC / "balanced_expansion/pairs.json") if r["scene"] == SCENE]
    already = {r["state_uid"] for r in old if r["split"] == "validation"}
    states = read(SRC / "../orthoflow3_loso_partial_count_v1/states.json")
    candidates = [(i, s) for i, s in enumerate(states) if s["scenario"] == SCENE
                  and s["split"] == "validation" and s["state_uid"] not in already]
    candidates.sort(key=lambda z: rank(z[1]["family"]))
    assert len(candidates) == 12 and len({s["family"] for _, s in candidates}) == 12
    assert all(s["state_uid"] not in already for _, s in candidates)

    source = [r for r in old if r["split"] == "train"]
    count = Counter(r["eta_uid"] for r in source)
    coordinates = {r["eta_uid"]: np.asarray(r["eta"], float) for r in source}
    assert len(coordinates) >= 16
    selected = sorted(coordinates, key=lambda key: (-count[key], key))[:4]
    matrix = np.stack(list(coordinates.values()))
    scale = np.maximum(np.ptp(matrix, axis=0), .1)
    while len(selected) < 16:
        remaining = [key for key in coordinates if key not in selected]
        nearest = {key: min(float(np.linalg.norm((coordinates[key] - coordinates[anchor]) / scale))
                            for anchor in selected) for key in remaining}
        selected.append(sorted(remaining, key=lambda key: (-nearest[key], key))[0])
    pairs = [{"scene": SCENE, "split": "heldout_validation_family",
              "state_uid": s["state_uid"], "state_index": i, "family": s["family"],
              "eta_uid": eta_uid, "eta": coordinates[eta_uid].tolist()}
             for i, s in candidates for eta_uid in selected]
    write(DEST / "design.json", {
        "rule": "SHA256-ranked unused source-VAL families; 4 most frequent source-TRAIN eta, then 12 deterministic farthest-first eta normalized by source-TRAIN coordinate range",
        "predeclared_new_flow": "independently trained v11, seed 88123, same v10 expert dataset and training protocol, source dev NLL selects checkpoint",
        "controller_task_labels_seen": False, "critic_or_generator_scores_seen": False,
        "state_families": [s["family"] for _, s in candidates],
        "state_uids": [s["state_uid"] for _, s in candidates],
        "eta_uids": selected, "eta_coordinates": [coordinates[key].tolist() for key in selected],
        "states": len(candidates), "eta_per_state": len(selected),
        "trial_policy": "standard future_index 0..15", "B15": "at least 15 observed successes",
        "prior_source_validation_state_uids_excluded": sorted(already),
        "not_true_t0": True, "independent_source_families": True,
        "source_pair_manifest_sha256": base.digest(SRC / "balanced_expansion/pairs.json"),
    })
    write(DEST / "pair_design.json", pairs)
    print(json.dumps({"states": len(candidates), "eta": len(selected),
                      "pairs": len(pairs), "requests": 16 * len(pairs)}))


def register():
    if (DEST / "protocol.json").exists():
        raise FileExistsError("Held-controller protocol already frozen")
    assert FLOW.is_file(), "new independent Flow checkpoint must finish training first"
    d = read(DEST / "design.json")
    profile = read(SRC / "balanced_expansion/protocol.json")["profiles"][SCENE]
    flow_hash = base.digest(FLOW)
    excluded = {profile["base_flow_sha256"], profile["alternate_flow_sha256"],
                read(SRC / "second_variant/protocol.json")["profiles"][SCENE]["alternate_flow_sha256"],
                read(OUT / "fresh_controller_v8_v1/protocol.json")["profiles"][SCENE]["alternate_flow_sha256"]}
    assert flow_hash not in excluded
    with FLOW.open("rb") as handle:
        checkpoint = pickle.load(handle)
    assert checkpoint["config"]["environment_fingerprint"] == profile["environment_fingerprint"]
    assert checkpoint["config"]["flow_steps"] == 10
    with connect() as db, transaction(db):
        original = db.execute("SELECT * FROM controller_config WHERE controller_uid=?",
                              (profile["base_controller_uid"],)).fetchone()
        assert original is not None and original["compatibility_quality"] == "EXACT_PROFILE"
        payload = json.loads(original["config_json"])
        payload.update(flow_checkpoint_sha256=flow_hash,
                       committed_t0_flow_sha256=profile["base_flow_sha256"],
                       conditioning=str(payload.get("conditioning", "")) + "+committed_base_t0_then_alt_future_v1",
                       intervention_runtime_sha256=base.digest(base.__file__))
        controller_uid = uid("ctl", payload)
        db.execute("""INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,
            flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,
            success_semantics_version,conditioning_version,rng_semantics_version,config_json,
            compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (controller_uid, original["scenario_uid"], flow_hash, original["orthoflow3_sha256"],
             original["safety_config_hash"], original["horizon"], original["dt"],
             original["success_semantics_version"], payload["conditioning"],
             original["rng_semantics_version"], canonical(payload), "EXACT_PROFILE"))
        db.execute("""INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,
            code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)""",
            (EXPERIMENT, "independent_ring_held_controller_B15_selection_v1", str(DEST),
             base.digest(DEST / "design.json"), base.digest(__file__),
             canonical({"controller_task_labels_seen": False, "source_validation_families": 12})))
    pairs = [{**r, "base_controller_uid": profile["base_controller_uid"],
              "alternate_controller_uid": controller_uid} for r in read(DEST / "pair_design.json")]
    requests = [{"state_uid": p["state_uid"], "eta_uid": p["eta_uid"],
                 "controller_uid": controller_uid,
                 "seed_keys": [canonical({"future_index": k}) for k in range(16)]} for p in pairs]
    new_profile = {**profile, "alternate_controller_uid": controller_uid,
                   "alternate_flow_sha256": flow_hash, "alternate_path": str(FLOW)}
    write(DEST / "pairs.json", pairs)
    write(DEST / "planned_rollouts.json", {"requests": requests})
    write(DEST / "protocol.json", {
        "schema": "independent_held_flow_B15_selection_v1", "experiment_uid": EXPERIMENT,
        "scene": SCENE, "design_sha256": base.digest(DEST / "design.json"),
        "state_eta_design_frozen_before_new_flow_outcomes": True,
        "new_controller_not_in_critic_training_or_selection": True,
        "source_validation_families_disjoint_from_previous_probe": True,
        "candidate_pool_score_blind": True, "intervention": "canonical first step then independent v11 future Flow",
        "safety_success_basis_unchanged": True,
        "code_sha256": base.digest(base.__file__),
        "profiles": {SCENE: new_profile}, "requested_continuations": len(pairs) * 16,
    })
    print(json.dumps({"controller_uid": controller_uid, "pairs": len(pairs),
                      "requested_continuations": len(pairs) * 16}))


def setup():
    base.OUT = DEST
    base.EXPERIMENT = EXPERIMENT


def run(shard, shards):
    setup()
    base.run(SCENE, shard, shards)


def merge():
    setup()
    base.merge()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("design", "register", "run", "merge"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.action == "design": design()
    elif args.action == "register": register()
    elif args.action == "run": run(args.shard, args.shards)
    else: merge()
