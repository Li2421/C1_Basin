"""Prospective Ring Flow-v8 controller holdout on frozen source VAL pairs.

The v8 controller is chosen by checkpoint chronology and compatible
environment fingerprint, not by its outcomes. It supplies new independent
controller labels for the existing 4-state x 4-eta source VAL panel.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from diagnostics.orthoflow3_controller_intervention_generalization_v1 import intervention as base
from shared_rollout_db.src.rollout_db import canonical, connect, transaction, uid

from .probe import OUT, SRC, read, write


DEST = OUT / "fresh_controller_v8_v1"
EXPERIMENT = "exp_orthoflow3_controller_info_ring_v8_holdout_v1"
SCENE = "ring_exchange"
FLOW = base.ROOT / "diagnostics/ring_exchange_stage1/base_u_v8_local_macflow/best.pkl"


def setup():
    base.OUT = DEST
    base.EXPERIMENT = EXPERIMENT


def prepare():
    if (DEST / "protocol.json").exists():
        raise FileExistsError("Fresh-controller protocol already frozen")
    parent = read(SRC / "balanced_expansion/protocol.json")["profiles"][SCENE]
    old_pairs = [r for r in read(SRC / "balanced_expansion/pairs.json")
                 if r["scene"] == SCENE and r["split"] == "validation"]
    assert len(old_pairs) == 16 and len({r["state_uid"] for r in old_pairs}) == 4
    flow_hash = base.digest(FLOW)
    assert flow_hash not in {parent["base_flow_sha256"], parent["alternate_flow_sha256"],
                             read(SRC / "second_variant/protocol.json")["profiles"][SCENE]["alternate_flow_sha256"]}
    with FLOW.open("rb") as fh:
        checkpoint = pickle.load(fh)
    assert checkpoint["config"]["environment_fingerprint"] == parent["environment_fingerprint"]
    with connect() as db, transaction(db):
        original = db.execute("SELECT * FROM controller_config WHERE controller_uid=?",
                              (parent["base_controller_uid"],)).fetchone()
        assert original is not None and original["compatibility_quality"] == "EXACT_PROFILE"
        payload = json.loads(original["config_json"])
        payload.update(flow_checkpoint_sha256=flow_hash,
                       committed_t0_flow_sha256=parent["base_flow_sha256"],
                       conditioning=str(payload.get("conditioning", "")) + "+committed_base_t0_then_alt_future_v1",
                       intervention_runtime_sha256=base.digest(base.__file__))
        new_uid = uid("ctl", payload)
        db.execute("""INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,
            flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,
            success_semantics_version,conditioning_version,rng_semantics_version,config_json,
            compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (new_uid, original["scenario_uid"], flow_hash, original["orthoflow3_sha256"],
             original["safety_config_hash"], original["horizon"], original["dt"],
             original["success_semantics_version"], payload["conditioning"],
             original["rng_semantics_version"], canonical(payload), "EXACT_PROFILE"))
        db.execute("""INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,
            code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)""",
            (EXPERIMENT, "ring_v8_fresh_controller_holdout_v1", str(DEST),
             base.digest(__file__), base.digest(base.__file__),
             canonical({"source_validation_only": True, "controller_v8_labels_heldout_from_all_training": True})))
    profile = {**parent, "alternate_controller_uid": new_uid,
               "alternate_flow_sha256": flow_hash, "alternate_path": str(FLOW)}
    pairs = [{**r, "alternate_controller_uid": new_uid} for r in old_pairs]
    requests = [{"state_uid": r["state_uid"], "eta_uid": r["eta_uid"],
                 "controller_uid": new_uid,
                 "seed_keys": [canonical({"future_index": i}) for i in range(16)]} for r in pairs]
    write(DEST / "pairs.json", pairs)
    write(DEST / "planned_rollouts.json", {"requests": requests})
    write(DEST / "protocol.json", {
        "schema": "fresh_compatible_future_Flow_v8_holdout_v1", "experiment_uid": EXPERIMENT,
        "intervention": "canonical Flow at first step, v8 compatible frozen Flow thereafter",
        "source_only": True, "split": "validation", "states": 4, "eta_per_state": 4,
        "controller_choice": "predeclared v8 local checkpoint between existing v9 and v7 variants; no v8 task outcomes inspected",
        "target_frozen_K16_untouched": True, "ring_safety_success_basis_unchanged": True,
        "code_sha256": base.digest(base.__file__),
        "profiles": {SCENE: profile}, "requested_continuations": 256,
    })
    print(json.dumps({"experiment": EXPERIMENT, "controller_uid": new_uid,
                      "pairs": len(pairs), "requested_continuations": 256}))


def run(shard, shards):
    setup()
    base.run(SCENE, shard, shards)


def merge():
    setup()
    base.merge()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run", "merge"))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "run": run(args.shard, args.shards)
    else: merge()
