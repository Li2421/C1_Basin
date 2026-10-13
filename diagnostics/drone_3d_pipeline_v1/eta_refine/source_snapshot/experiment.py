"""Exact cache identities and bounded single-scene pipeline development runs."""
from . import resources  # noqa: F401
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src.planner import preflight
from shared_rollout_db.src.rollout_db import canonical, connect, eta_identity, lookup_exact, transaction, uid
from . import pilot, pipeline

ROOT=Path("diagnostics/drone_3d_pipeline_v1")
ETAS=[[0.,0.,0.],[.5,0.,0.],[0.,.5,0.],[0.,-.25,0.],[-.5,0.,0.],[0.,0.,.5],[.5,.5,.5],[-.5,.5,.5]]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+"\n")


def sources():
    return {str(p):sha(p) for p in [Path(pilot.__file__),Path(pipeline.__file__),Path("new_benchmark_common/macflow.py"),
        Path("shared_control/basis_families.py"),
        Path("/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official/utils/networks.py"),
        Path("/home/zhihan/research/01_MACFlow_Baseline_Reproduction/MACFlow_Official/utils/flax_utils.py")]}


def states_for(stage):
    old=json.loads(Path("diagnostics/drone_3d_pilot_v3/protocol.json").read_text())
    scid=old["scenario_uid"]
    if stage=="smoke":
        return [dict(**s,split="development_seen_demonstration") for s in old["states"]]
    # No cached demonstration state is used as a held-out Q state.
    states=[]
    for index in range(24):
        seed=2026101300+index; rng=np.random.default_rng(seed)
        p=np.array([[-2.3,0.,.4],[2.3,0.,.4]])+rng.uniform(-1,1,(2,3))*[.08,.015,.015]
        physical=dict(positions=p.tolist(),velocities=np.zeros((2,3)).tolist(),goals=p[::-1].copy().tolist())
        split="train" if index<12 else ("dev" if index<18 else "test")
        states.append(dict(seed=seed,physical=physical,uid=uid("state",dict(scenario=scid,physical=physical)),split=split))
    return states


def prepare(folder,checkpoint,stage,placement,eta_values=None):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/"protocol.json").exists():
        print("Existing protocol retained."); return
    old=json.loads(Path("diagnostics/drone_3d_pilot_v3/protocol.json").read_text())
    scid=old["scenario_uid"]; config=asdict(pilot.Config()); states=states_for(stage)
    etas=eta_values if eta_values is not None else (ETAS if placement=="TT" else [ETAS[0],ETAS[2]])
    payload=dict(scenario_uid=scid,checkpoint=str(checkpoint.resolve()),flow_checkpoint_sha256=sha(checkpoint),
                 placement=placement,basis="orthoflow3_xyz_v1",basis_scale=3.303687238760696,source=sources(),
                 config=config,observation="p,v,g-p,other-p; no teacher state",conditioning="true_t0_eta_latched_v1",
                 rng="jax PRNGKey(future_index); fold_in(physical_step); common random numbers across eta",
                 success="all_goals_swept_safety_mutual_block_v1",initial_monitor_history=[],initial_step=0)
    cid=uid("ctl",payload); exp=uid("exp",dict(folder=str(folder.resolve()),controller=cid))
    cases=[]
    for state in states:
        for i,eta in enumerate(etas):
            eid=eta_identity(eta)[0]; seed_key=canonical(dict(future_index=0))
            rid=uid("roll",dict(state=state["uid"],eta=eid,controller=cid,seed=seed_key))
            cases.append(dict(state_uid=state["uid"],eta_uid=eid,eta=eta,eta_index=i,controller_uid=cid,seed_key=seed_key,
                              future_index=0,rollout_uid=rid,split=state["split"]))
    protocol=dict(stage=stage,placement=placement,experiment_uid=exp,scenario_uid=scid,states=states,
                  controller=dict(uid=cid,payload=payload),etas=etas,cases=cases,source=sources(),
                  max_cases=len(cases),max_cpu_seconds=600,scope="small N2 pipeline development; not B15 or controller transfer")
    dump(folder/"protocol.json",protocol)
    dump(folder/"planned_rollouts.json",dict(requests=[{k:c[k] for k in ("state_uid","eta_uid","controller_uid")}|dict(seed_keys=[c["seed_key"]]) for c in cases]))
    dump(folder/"history_audit.json",dict(
        prior="diagnostics/drone_3d_pilot_v3",reused="all three teacher NPZ; simulator/collision checker; existing baseline outcomes",
        flow_checkpoint=str(checkpoint.resolve()),flow_sha256=sha(checkpoint),
        no_old_equivalent="Existing pilot route witness/goal controller is not this learned Flow + latched eta + placement",
        remaining_gap="actual closed-loop outcomes of this checkpoint, eta and physical initial state",
        critic_encoding_policy="changing Q representation must reuse these labels",source=sources()))
    snapshot=folder/"source_snapshot"; snapshot.mkdir(exist_ok=True)
    for path in [Path(pilot.__file__),Path(pipeline.__file__),Path(__file__)]:shutil.copy2(path,snapshot/path.name)
    with connect() as db,transaction(db):
        db.execute("INSERT OR IGNORE INTO experiment(experiment_uid,name,path,protocol_hash,code_hash,start_time,metadata_json) VALUES(?,?,?,?,?,CURRENT_TIMESTAMP,?)",
                   (exp,f"Drone3D Flow {stage} {placement}",str(folder.resolve()),sha(folder/"protocol.json"),sha(__file__),canonical(dict(max_cases=len(cases),gpu=0))))
        db.execute("INSERT OR IGNORE INTO controller_config(controller_uid,scenario_uid,flow_checkpoint_sha256,orthoflow3_sha256,safety_config_hash,horizon,dt,success_semantics_version,conditioning_version,rng_semantics_version,config_json,compatibility_quality) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                   (cid,scid,sha(checkpoint),sha(pipeline.__file__),sha(pilot.__file__),str(config["max_steps"]),str(config["dt"]),
                    payload["success"],payload["conditioning"],payload["rng"],canonical(payload),"EXACT_PROFILE"))
        for state in states:
            db.execute("INSERT OR IGNORE INTO state(state_uid,scenario_uid,source_group,content_hash,physical_state_json,h0_json,goals_geometry_json,provenance_json,identity_quality) VALUES(?,?,?,?,?,?,?,?,?)",
                       (state["uid"],scid,state["split"],uid("physical",state["physical"]),canonical(state["physical"]),
                        canonical(dict(step=0,monitor_history=[])),canonical(dict(goals=state["physical"]["goals"])),canonical(dict(seed=state["seed"],protocol=str(folder.resolve()))),"CONTENT_EXACT"))
        for eta in etas:
            eid,v,hx=eta_identity(eta)
            db.execute("INSERT OR IGNORE INTO eta(eta_uid,eta1,eta2,eta3,normalized_eta_json,canonical_hex) VALUES(?,?,?,?,?,?)",(eid,*v,None,hx))
    receipt=preflight(folder/"planned_rollouts.json"); dump(folder/"cache_preflight.json",receipt)
    print(json.dumps(receipt["summary"]))


def run(folder,split="all",limit=None):
    protocol=json.loads((folder/"protocol.json").read_text())
    if protocol["source"]!=sources():raise ValueError("execution source changed; prepare a fresh protocol")
    cp=Path(protocol["controller"]["payload"]["checkpoint"])
    if sha(cp)!=protocol["controller"]["payload"]["flow_checkpoint_sha256"]:raise ValueError("checkpoint changed")
    receipt=preflight(folder/"planned_rollouts.json"); dump(folder/f"preflight_{split}.json",receipt)
    if receipt["summary"]["ambiguous"]:raise ValueError("ambiguous cache")
    states={s["uid"]:s for s in protocol["states"]}; runtime=None
    cases=[c for c in protocol["cases"] if split=="all" or c["split"]==split]
    if limit:cases=cases[:limit]
    raw=folder/"raw"; raw.mkdir(exist_ok=True); rows=[]; new=reused=0
    start=time.perf_counter(); cpu=time.process_time()
    for case in cases:
        path=raw/f"{case['rollout_uid']}.json"; trace_path=path.with_suffix(".npz"); state=states[case["state_uid"]]
        with connect(True) as db:
            conflict=db.execute("SELECT conflict_quarantined FROM rollout WHERE rollout_uid=?",(case["rollout_uid"],)).fetchone()
            if conflict and conflict[0]:raise ValueError("quarantined cache identity")
            hit=lookup_exact(db,case["state_uid"],case["eta_uid"],case["controller_uid"],[case["seed_key"]])
        if path.exists():
            row=json.loads(path.read_text())
            if any(row[k]!=case[k] for k in ("state_uid","controller_uid","eta_uid","rollout_uid")) or sha(trace_path)!=row["trace_sha256"]:
                raise ValueError("local evidence identity/hash mismatch")
            if hit["missing_seeds"] and not conflict:append_journal([row],protocol["experiment_uid"],"cache_recovery")
            reused+=1
        elif hit["status"]!="INCOMPATIBLE":
            # Reuse exact raw record from its source, preserving all trace provenance.
            source=Path(hit["records"][0]["original_source_file"])
            if not source.exists() and source.with_suffix(".merged").exists():source=source.with_suffix(".merged")
            row=next(json.loads(s) for s in source.read_text().splitlines() if json.loads(s).get("rollout_uid")==case["rollout_uid"])
            if sha(row["trace_path"])!=row["trace_sha256"]:raise ValueError("original trace unavailable")
            reused+=1
        else:
            if time.process_time()-cpu>protocol["max_cpu_seconds"]:raise RuntimeError("CPU budget reached; completed evidence is resumable")
            if runtime is None:runtime=pipeline.FlowRuntime(cp)
            t=time.perf_counter()
            trace,result=pipeline.rollout(state["physical"],runtime,case["eta"],protocol["placement"],case["future_index"])
            row=dict(**result,**case,cache_identity_schema="registered_exact_uid_v1",scenario_uid=protocol["scenario_uid"],
                     scenario="Drone3D_RaisedSideAlcove_N2",flow_checkpoint_sha256=sha(cp),
                     seed=state["seed"],initial_positions=state["physical"]["positions"],initial_velocities=state["physical"]["velocities"],goals=state["physical"]["goals"],
                     initial_state_sha256=uid("initial",dict(physical=state["physical"],config=asdict(pilot.Config()),step=0,monitor_history=[])),
                     timestamp=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-t,trace_path=str(trace_path.resolve()),
                     source_record=str(path.resolve()),source=sources())
            np.savez_compressed(trace_path,**trace,metadata_json=np.array(json.dumps(row)))
            row["trace_sha256"]=sha(trace_path); dump(path,row); append_journal([row],protocol["experiment_uid"],"cpu1"); new+=1
        rows.append(row)
        print(json.dumps(dict(state=state["seed"],eta=case["eta_index"],outcome=row["outcome"],steps=row["episode_length"],remaining=row["final_goal_distances"])),flush=True)
    subprocess.run([sys.executable,"-m","shared_rollout_db.src.merger","--experiment",protocol["experiment_uid"]],check=True)
    summary=dict(split=split,cases=len(rows),new_rollouts=new,reused_rollouts=reused,outcomes=dict(Counter(r["outcome"] for r in rows)),
                 wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
                 by_eta={str(i):dict(Counter(r["outcome"] for r in rows if r["eta_index"]==i)) for i in range(len(protocol["etas"]))},
                 gpu=False,all_collision_free=all(not r["collision"] for r in rows))
    dest=folder/f"summary_{split}_{'new' if new else 'reuse'}.json"
    dump(dest,summary); dump(folder/"cache_postflight.json",preflight(folder/"planned_rollouts.json")); print(json.dumps(summary,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("command",choices=["prepare","run"])
    p.add_argument("--root",type=Path,default=ROOT/"smoke_tt"); p.add_argument("--checkpoint",type=Path,default=ROOT/"flow/best.pkl")
    p.add_argument("--stage",choices=["smoke","qdata"],default="smoke"); p.add_argument("--placement",choices=["TT","FF"],default="TT")
    p.add_argument("--split",default="all"); p.add_argument("--limit",type=int); p.add_argument("--etas",type=json.loads)
    a=p.parse_args()
    if a.command=="prepare":prepare(a.root,a.checkpoint,a.stage,a.placement,a.etas)
    else:run(a.root,a.split,a.limit)
