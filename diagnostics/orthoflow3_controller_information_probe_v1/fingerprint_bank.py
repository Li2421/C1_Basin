"""24D Flow/safety response signature on eight outcome-blind source states.

The bank is fixed using state UID only and is shared by every controller in a
scene. This diagnostic tests whether known controllers can be identified from
physical function responses. No task rollout or success label is used.
"""
from __future__ import annotations

import json

import numpy as np

from diagnostics.orthoflow3_controller_intervention_generalization_v1.rich_context import RichRuntime
from .probe import OUT, SRC, SCENES, read, write


def run(scene):
    if scene=="toy_giveway": raise ValueError("Toy heldout panel is already B15 saturated; bank diagnostic uses DB/Four/Ring")
    import jax
    rows=read(SRC/"rich_probe_rows.json")
    source=read(SRC/"../orthoflow3_loso_partial_count_v1/states.json")
    known=read(SRC/"training_rows.json")
    uid_to_index={r["state_uid"]:r["state_index"] for r in known if r["scene"]==scene}
    train_uids=sorted({r["state_uid"] for r in rows if r["scene"]==scene and r["split"]=="train"})
    bank=train_uids[:8]
    p1=read(SRC/"balanced_expansion/protocol.json")["profiles"][scene]["alternate_path"]
    p2=read(SRC/"second_variant/protocol.json")["profiles"][scene]["alternate_path"]
    basert=RichRuntime(scene)
    runtimes=[basert,RichRuntime(scene,p1),RichRuntime(scene,p2)]
    features=[]
    for ci,rt in enumerate(runtimes):
        f=[]
        for bi,uid in enumerate(bank):
            physical=source[uid_to_index[uid]]["physical"]
            env=basert.core.reset(physical)
            values=[]
            for root in (2026100417,2026100499):
                key=jax.random.fold_in(jax.random.PRNGKey(root),bi)
                raw=np.asarray(rt.base_flow(env,key) if rt.alt_flow is None else rt.alt_flow(env,key),float)
                safe=np.asarray(rt.core.project(env,raw),float)
                goal=np.asarray(env.goals-env.positions,float)
                goal/=np.maximum(np.linalg.norm(goal,axis=1,keepdims=True),1e-12)
                speed=basert.core.cfg.max_speed
                along=float(np.mean(np.sum(raw*goal,axis=1))/speed)
                lateral=float(np.mean(raw[:,1]*goal[:,0]-raw[:,0]*goal[:,1])/speed)
                removal=float(np.mean(np.linalg.norm(safe-raw,axis=1))/speed)
                values.append([along,lateral,removal])
            f.extend(np.asarray(values).mean(0).tolist())
        assert len(f)==24 and np.isfinite(f).all()
        features.append(f)
    write(OUT/f"fingerprint_bank_{scene}.json",{"scene":scene,"bank_state_uids":bank,
           "source_TRAIN_only":True,"controller_variants":["base","first","second"],
           "features":["Flow_goal_projection","Flow_lateral_projection","safety_removal"],
           "matched_probe_roots":[2026100417,2026100499],"observed_task_outcomes_read":False,
           "full_rollout":False,"new_task_continuation":0,"vectors":features})
    print(json.dumps({"scene":scene,"bank_states":len(bank),"controller_vector_distances":
                      [[float(np.linalg.norm(np.asarray(a)-b)) for b in features] for a in features]}))


if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument("--scene",choices=SCENES[1:],required=True)
    args=ap.parse_args();run(args.scene)
