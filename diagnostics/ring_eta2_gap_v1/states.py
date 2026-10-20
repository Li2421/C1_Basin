"""Outcome-blind disjoint TRAIN-new and TEST state samples."""
import collections
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from ring_exchange.environment import LocalFrameConfig,sample_instance
from shared_rollout_db.src.rollout_db import uid,canonical,connect

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")

def sig(p):
    if not isinstance(p,dict):return None
    parts=[]
    for names in (("positions","initial_positions"),("velocities","initial_velocities"),("goals",)):
        raw=next((p[k] for k in names if k in p),None)
        if raw is None:return None
        a=np.asarray(raw,np.float64).copy();a[a==0]=0
        parts.append((a.shape,a.tobytes().hex()))
    return hashlib.sha256(canonical(dict(arr=parts,timestep=float(p.get("timestep",p.get("step",0))))).encode()).hexdigest()

def main():
    reg=json.loads((HERE/"PREREGISTRATION.receipt.json").read_text())
    assert sha(HERE/"PREREGISTRATION.md")==reg["sha256"]
    old=json.loads((ROOT/"diagnostics/controller_universal_eta_schedule_v1/protocol.json").read_text())
    scene=old["environments"]["ring_exchange"]
    scenario_uid=scene["scenario_uid"]
    cfg=LocalFrameConfig(**scene["config"]["config"])
    existing_uid=set();existing_content=set();existing_sig=set();unresolved=collections.Counter()
    with connect(True) as db:
        for row in db.execute("SELECT state_uid,content_hash,physical_state_json,identity_quality FROM state"):
            existing_uid.add(row["state_uid"])
            if row["content_hash"]:existing_content.add(row["content_hash"])
            try:p=json.loads(row["physical_state_json"]) if row["physical_state_json"] else None
            except (TypeError,ValueError):p=None
            h=sig(p)
            if h:existing_sig.add(h)
            else:unresolved[row["identity_quality"]]+=1
    groups={};attempts=[];own_sig=set()
    for split,start,target in [('TRAIN-new',2026181000,32),('TEST',2026182000,48)]:
        states=[];seed=start
        while len(states)<target:
            ins=sample_instance(seed,"development",cfg)
            physical=dict(positions=ins.positions.tolist(),velocities=ins.velocities.tolist(),goals=ins.goals.tolist(),timestep=0)
            alias=f"round18_eta2_{split}_{len(states)}"
            content=uid("content",dict(physical=physical,reset="empty monitor true-t0"))
            state_uid=uid("state",dict(scenario=scenario_uid,content=content))
            h=sig(physical)
            collision=state_uid in existing_uid or content in existing_content or h in existing_sig or h in own_sig
            attempts.append(dict(split=split,seed=seed,state_uid=state_uid,physical_signature=h,collision=collision))
            if not collision:
                states.append(dict(uid=alias,state_uid=state_uid,content_hash=content,scenario="ring_exchange",scenario_uid=scenario_uid,
                                   rng_namespace=int(hashlib.sha256(alias.encode()).hexdigest()[:8],16),physical=physical,
                                   family="round18_eta2_gap",study_split=split,study_index=len(states),
                                   metadata=dict(draw_seed=seed,outcome_blind=True)))
                own_sig.add(h)
            seed+=1
            assert seed<start+1000,"Too many collisions; stop instead of weakening uniqueness rule"
        groups[split]=states
        write(HERE/f'{split}_STATES.json',states)
    write(HERE/"STATE_DEDUP_AUDIT.json",dict(status="PASS",utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         counts={k:len(v) for k,v in groups.items()},attempts=attempts,collisions=sum(x["collision"] for x in attempts),db_state_uid_count=len(existing_uid),
         db_content_count=len(existing_content),db_physical_signatures=len(existing_sig),unresolved_db_physical_by_quality=dict(unresolved),
         scenario_uid=scenario_uid,source_protocol_sha256=sha(ROOT/"diagnostics/controller_universal_eta_schedule_v1/protocol.json"),
         train_states_sha256=sha(HERE/'TRAIN-new_STATES.json'),test_states_sha256=sha(HERE/'TEST_STATES.json'),new_success_labels_generated=False))
    print(json.dumps(dict(counts={k:len(v) for k,v in groups.items()},attempts=len(attempts),db_rows=len(existing_uid))))

if __name__=="__main__":main()
