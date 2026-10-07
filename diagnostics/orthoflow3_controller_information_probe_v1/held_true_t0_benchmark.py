"""Prospective true-t0 held-controller Ring benchmark, designed without Q labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import jax
import numpy as np

from new_benchmark_common import basin_dataset_v1 as bd
from new_benchmark_common.macflow import load_checkpoint
from new_benchmark_common.safety_eta3 import ScenarioRuntime, state_token
from ring_exchange.environment import sample_initial_state
from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src.rollout_db import ROOT, canonical, connect, eta_identity, transaction, uid

from diagnostics.orthoflow3_controller_intervention_generalization_v1 import intervention as old
from .held_robust_benchmark import FLOW
from .probe import OUT, read, write


DEST = OUT / "held_controller_true_t0_v1"
EXPERIMENT = "exp_orthoflow3_ring_held_controller_true_t0_v1"
SCENE = "ring_exchange"
N_STATES = 48
K = 16
STAGE_SIZE = 24


def _content_hash(value): return hashlib.sha256(canonical(value).encode()).hexdigest()


def design():
    if (DEST / "design.json").exists(): raise FileExistsError("true-t0 design already frozen")
    base = bd.TrainingRuntime(SCENE, [], parent=False)
    old_eta = read(OUT / "held_controller_robust_selection_v1/design.json")
    selected = list(zip(old_eta["eta_uids"], old_eta["eta_coordinates"]))
    assert len(selected) == K and len(set(u for u, _ in selected)) == K
    states = []
    for i in range(N_STATES):
        seed = 6_000_000 + i
        x = sample_initial_state("test", seed, base.config)
        physical = {"positions": x.positions.tolist(), "velocities": x.velocities.tolist(),
                    "goals": x.goals.tolist(), "timestep": 0, "normalized_episode_time": 0.0}
        ch = bd.content_hash(physical)
        suid = uid("state", {"scenario": base.scenario_uid, "content": ch})
        states.append({"uid": suid, "alias": f"RING_HELD_T0_{i:03d}", "index": i,
                       "physical": physical, "content_hash": ch,
                       "source_group": f"held_true_t0_test_seed_{seed}",
                       "split": "held_test_stage1" if i < STAGE_SIZE else "held_test_stage2",
                       "provenance": {"distribution": "ring_exchange_test", "seed": seed,
                                      "outcome_blind": True, "true_t0": True,
                                      "candidate_pool": "frozen_source_train_16_eta"}})
    assert len({s["uid"] for s in states}) == N_STATES
    pairs = [{"state_uid": s["uid"], "state_index": s["index"], "eta_uid": u,
              "eta": e, "stage": 1 if s["index"] < STAGE_SIZE else 2}
             for s in states for u, e in selected]
    write(DEST / "states.json", states)
    write(DEST / "pair_design.json", pairs)
    write(DEST / "design.json", {
        "created_before_any_true_t0_held_controller_outcomes": True,
        "state_rule": "Ring test-split sample_initial_state seeds 6000000..6000047, no outcome filtering",
        "state_family_independence": "one distinct initial-condition RNG seed per state; prior Ring test used 4000000..4000059",
        "stages": {"first": [0,23], "second": [24,47]},
        "stage2_predeclared_trigger": "run if stage1 has fewer than 8 states with 1..4 B15 candidates and at least one confirmed non-B15, or fewer than 12 oracle-eligible mixed states; do not otherwise add more data",
        "candidate_rule": "reuse exactly the 16 score-blind source-TRAIN eta frozen before independent Flow v11 outcomes",
        "candidate_source_design_sha256": old.digest(OUT / "held_controller_robust_selection_v1/design.json"),
        "controller": str(FLOW), "controller_checkpoint_sha256": old.digest(FLOW),
        "true_t0": True, "score_or_outcome_based_selection": False,
        "states": N_STATES, "eta_per_state": K, "standard_seeds": list(range(16)),
    })
    print(json.dumps({"states":len(states),"pairs":len(pairs),"stage1_requested":STAGE_SIZE*K*16,
                      "stage2_max_requested":STAGE_SIZE*K*16}))


def register():
    if (DEST / "protocol.json").exists(): raise FileExistsError("protocol already frozen")
    design = read(DEST / "design.json")
    assert old.digest(FLOW) == design["controller_checkpoint_sha256"]
    states = read(DEST / "states.json")
    base = bd.TrainingRuntime(SCENE, states, parent=False)
    original_uid = base.controllers["orthoflow3"]["uid"]
    with connect() as db, transaction(db):
        original = db.execute("SELECT * FROM controller_config WHERE controller_uid=?", (original_uid,)).fetchone()
        assert original is not None
        payload = json.loads(original["config_json"])
        payload.update(flow_checkpoint_sha256=old.digest(FLOW),
                       committed_t0_flow_sha256=base.checkpoint_sha,
                       conditioning=str(payload.get("conditioning", "")) +
                       "+fresh_test_true_t0+committed_base_t0_then_alt_future_v1",
                       intervention_runtime_sha256=old.digest(old.__file__))
        held_uid = uid("ctl", payload)
        db.execute("""INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,
            flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,
            success_semantics_version,conditioning_version,rng_semantics_version,config_json,
            compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (held_uid, original["scenario_uid"], old.digest(FLOW), original["orthoflow3_sha256"],
             original["safety_config_hash"], original["horizon"], original["dt"],
             original["success_semantics_version"], payload["conditioning"],
             original["rng_semantics_version"], canonical(payload), "EXACT_PROFILE"))
        db.execute("""INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,
            code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)""",
            (EXPERIMENT, "ring_true_t0_independent_held_flow_B15_v1", str(DEST),
             old.digest(DEST / "design.json"), old.digest(__file__),
             canonical({"true_t0": True, "score_blind": True, "two_predeclared_stages": True})))
    pairs = [{**p,"alternate_controller_uid":held_uid,"base_controller_uid":original_uid}
             for p in read(DEST / "pair_design.json")]
    write(DEST / "pairs.json",pairs)
    for stage in (1,2):
        requests = [{"state_uid":p["state_uid"],"eta_uid":p["eta_uid"],
                     "controller_uid":held_uid,
                     "seed_keys":[canonical({"future_index":i}) for i in range(16)]}
                    for p in pairs if p["stage"]==stage]
        write(DEST / f"planned_stage{stage}.json",{"requests":requests})
    write(DEST / "protocol.json",{
        "schema":"ring_held_controller_true_t0_B15_v1",
        "design_sha256":old.digest(DEST / "design.json"),
        "experiment_uid":EXPERIMENT,"scenario":SCENE,"held_controller_uid":held_uid,
        "base_controller_uid":original_uid,"held_flow_sha256":old.digest(FLOW),
        "base_flow_sha256":base.checkpoint_sha,
        "environment_fingerprint":base.manifest["environment_fingerprint"],
        "state_and_candidate_design_before_outcomes":True,
        "frozen_model_scores_before_outcomes_required":True,
        "safety_basis_success_semantics_unchanged":True,
        "new_controller_not_in_critic_training":True,
        "stage1_requested":STAGE_SIZE*K*16,"stage2_max_requested":STAGE_SIZE*K*16,
        "runtime_sha256":old.digest(old.__file__),
    })
    print(json.dumps({"held_controller_uid":held_uid,"stage1_requested":STAGE_SIZE*K*16}))


def run(stage, shard, shards):
    from new_benchmark_common.safety_eta3 import ScenarioRuntime
    protocol = read(DEST / "protocol.json")
    assert old.digest(FLOW) == protocol["held_flow_sha256"]
    assert old.digest(old.__file__) == protocol["runtime_sha256"]
    states = read(DEST / "states.json")
    pairs = [p for p in read(DEST / "pairs.json") if p["stage"]==stage]
    pre = read(DEST / f"cache_preflight_stage{stage}.json")
    missing={(r["state_uid"],r["eta_uid"],r["controller_uid"],seed)
             for r in pre["details"] for seed in r["missing_seeds"]}
    jobs=[(p,seed) for p in pairs for seed in range(16)
          if (p["state_uid"],p["eta_uid"],protocol["held_controller_uid"],
              canonical({"future_index":seed})) in missing]
    assert len(jobs)==pre["summary"]["genuinely_missing"]
    jobs=[job for i,job in enumerate(jobs) if i%shards==shard]
    rt=bd.TrainingRuntime(SCENE,states,parent=False)
    assert rt.controllers["orthoflow3"]["uid"]==protocol["base_controller_uid"]
    held_agent,_=load_checkpoint(FLOW,expected_environment_fingerprint=protocol["environment_fingerprint"])
    base_agent=rt.agent
    counter={"calls":0}
    def flow(env,key):
        rt.agent=base_agent if counter["calls"]==0 else held_agent
        counter["calls"]+=1
        return ScenarioRuntime.flow_world(rt,env,key)
    rt.flow_world=flow
    completed=0
    for p,seed in jobs:
        counter["calls"]=0
        eta=np.asarray(p["eta"],float)
        result=rt.rollout(states[p["state_index"]],eta,seed,"orthoflow3")
        result.update(controller_uid=protocol["held_controller_uid"],
                      base_controller_uid=protocol["base_controller_uid"],
                      flow_checkpoint_sha256=protocol["held_flow_sha256"],
                      committed_t0_flow_sha256=protocol["base_flow_sha256"],
                      intervention="canonical first Flow action then independent held Flow",
                      experiment_uid=EXPERIMENT)
        assert result["state_uid"]==p["state_uid"] and eta_identity(result["eta"])[0]==p["eta_uid"]
        append_journal([{"schema":"ring_held_true_t0_v1","record":result}],EXPERIMENT,
                       f"stage{stage}_shard{shard}")
        completed+=1
        if completed%32==0:print(json.dumps({"stage":stage,"shard":shard,"completed":completed}),flush=True)
    print(json.dumps({"stage":stage,"shard":shard,"complete":completed}),flush=True)


def merge(stage):
    """Single-writer, idempotent merge; preserve numerical failures as observed."""
    protocol=read(DEST / "protocol.json")
    allowed={(p["state_uid"],p["eta_uid"],protocol["held_controller_uid"])
             for p in read(DEST / "pairs.json") if p["stage"]==stage}
    journal=ROOT / "journals" / EXPERIMENT
    paths=sorted(journal.glob(f"stage{stage}_shard*.jsonl"))
    if not paths:raise FileNotFoundError("No stage journals")
    stats=Counter()
    with connect() as db:
        for path in paths:
            lines=path.read_text().splitlines()
            source_uid=uid("src",{"path":str(path.resolve())})
            with transaction(db):
                db.execute("""INSERT OR IGNORE INTO source_file(source_uid,experiment_uid,path,sha256,
                    file_type,classification,rows_seen) VALUES(?,?,?,?,?,?,?)""",
                    (source_uid,EXPERIMENT,str(path.resolve()),old.digest(path),".jsonl","SEED_EXACT",len(lines)))
                for number,line in enumerate(lines,1):
                    envelope=json.loads(line)
                    assert envelope["schema"]=="ring_held_true_t0_v1"
                    row=envelope["record"]
                    eu=eta_identity(row["eta"])[0]
                    assert (row["state_uid"],eu,row["controller_uid"]) in allowed
                    assert 0<=row["future_index"]<16
                    sk=canonical({"future_index":row["future_index"]})
                    rid=uid("roll",{"state":row["state_uid"],"eta":eu,
                                    "controller":row["controller_uid"],"seed":sk})
                    core=tuple(int(row[k]) for k in ("success","deadlock","timeout","collision","numerical_failure"))
                    previous=db.execute("SELECT * FROM rollout WHERE rollout_uid=?",(rid,)).fetchone()
                    if previous:
                        if core!=tuple(previous[k] for k in ("success","deadlock","timeout","collision","numerical_failure")):
                            db.execute("UPDATE rollout SET conflict_quarantined=1 WHERE rollout_uid=?",(rid,))
                            stats["conflicts"]+=1
                            continue
                        stats["duplicates"]+=1
                    else:
                        db.execute("""INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,
                            continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,
                            episode_length,j_def,min_wall_distance,min_agent_distance,outcome,experiment_uid,
                            original_source_file,timestamp,compatibility_quality,raw_record_hash)
                            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (rid,row["state_uid"],eu,row["controller_uid"],sk,sk,*core,
                             row.get("episode_length"),row.get("J_def"),row.get("minimum_wall_clearance"),
                             row.get("minimum_agent_clearance"),row.get("outcome"),EXPERIMENT,
                             str(path),row.get("timestamp"),"EXACT_REUSE",_content_hash(row)))
                        stats["inserted"]+=1
                    db.execute("INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)",(rid,source_uid,number))
            stats["journals"]+=1
        db.execute("UPDATE experiment SET new_rollout_count=(SELECT COUNT(*) FROM rollout WHERE experiment_uid=?), end_time=CURRENT_TIMESTAMP WHERE experiment_uid=?",(EXPERIMENT,EXPERIMENT))
        db.commit()
    write(DEST / f"merge_audit_stage{stage}.json",dict(stats))
    print(json.dumps(dict(stats)))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("design","register","run","merge"))
    parser.add_argument("--stage",type=int,choices=(1,2),default=1)
    parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--shards",type=int,default=6)
    a=parser.parse_args()
    if a.action=="design":design()
    elif a.action=="register":register()
    elif a.action=="run":run(a.stage,a.shard,a.shards)
    else:merge(a.stage)
