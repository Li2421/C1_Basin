"""Freeze H20/eta-only scores on a fresh true-t0 held-controller panel."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

from diagnostics.orthoflow3_controller_intervention_generalization_v1 import h20_features as h20
from diagnostics.orthoflow3_controller_intervention_generalization_v1.intervention import OLD
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from new_benchmark_common import basin_dataset_v1 as bd
from shared_rollout_db.src.rollout_db import connect

from .held_true_t0_benchmark import DEST, FLOW
from .probe import OUT, Critic, read, write


def physical():
    if (DEST / "context_physical.json").exists(): raise FileExistsError("physical context frozen")
    states = read(DEST / "states.json")
    old_states = read(OLD / "states.json")
    template = next(x["physical"] for x in old_states if x["scenario"] == "ring_exchange")
    base = h20.rc.RichRuntime("ring_exchange")
    output = []
    for row in states:
        p = copy.deepcopy(template)
        p.update(positions=row["physical"]["positions"],
                 velocities=row["physical"]["velocities"],
                 goals=row["physical"]["goals"],
                 remaining_fraction=1.0,
                 remaining_seconds=p["max_seconds"],
                 flow_committed=False)
        env = base.core.reset(p)
        key = jax.random.fold_in(jax.random.PRNGKey(bd.CONDITIONING_FLOW_ROOT),
                                 bd.state_token(row["uid"]))
        p["flow"] = np.asarray(base.base_flow(env, key), float).tolist()
        assert all(np.isfinite(x).all() for x in rep.entities(p).values())
        output.append({"state_uid": row["uid"], "physical": p})
    # The parser path must reproduce the exact old Ring encoder tensors.
    idx = next(i for i,x in enumerate(old_states) if x["scenario"]=="ring_exchange")
    old_x = dict(np.load(OLD / "entities.npz"))
    replay = rep.entities(old_states[idx]["physical"])
    for key in ("agents", "pairs", "globals", "agent_mask"):
        np.testing.assert_allclose(replay[key], old_x[key][idx], atol=1e-6, rtol=1e-6)
    write(DEST / "context_physical.json", output)
    print(json.dumps({"states":len(output),"old_parser_replay": "PASS"}))


def features(shard, shards):
    target=DEST / f"context_shard{shard}of{shards}.json"
    if target.exists(): raise FileExistsError("H20 feature shard already exists")
    h20.rc.OUT=DEST
    held=h20.rc.RichRuntime("ring_exchange",FLOW)
    base=h20.rc.RichRuntime("ring_exchange")
    pairs=read(DEST / "pairs.json")
    physical=read(DEST / "context_physical.json")
    results=[]
    for i,p in enumerate(pairs):
        if i%shards!=shard:continue
        state={"state_uid":p["state_uid"],"physical":physical[p["state_index"]]["physical"]}
        assert physical[p["state_index"]]["state_uid"]==p["state_uid"]
        a=h20.rc.cached(held,state,p["eta"])
        b=h20.rc.cached(base,state,p["eta"])
        results.append({"pair_index":i,"held_valid":a["valid"],"base_valid":b["valid"],
                        "held":a["features"]["mean"] if a["valid"] else None,
                        "wrong_base":b["features"]["mean"] if b["valid"] else None,
                        "held_error":a.get("error"),"base_error":b.get("error")})
        if len(results)%16==0:print(json.dumps({"shard":shard,"done":len(results)}),flush=True)
    write(target,results)
    print(json.dumps({"shard":shard,"complete":len(results),
                      "valid":sum(r["held_valid"] and r["base_valid"] for r in results)}),flush=True)


def freeze(shards):
    target=DEST / "frozen_predictions.json"
    if target.exists():raise FileExistsError("predictions already frozen")
    protocol=read(DEST / "protocol.json")
    with connect(True) as db:
        count=db.execute("SELECT COUNT(*) FROM rollout WHERE controller_uid=?",
                         (protocol["held_controller_uid"],)).fetchone()[0]
    assert count==0,"No held-controller Q16 outcome may exist before score freeze"
    pairs=read(DEST / "pairs.json")
    states=read(DEST / "context_physical.json")
    context=sorted((r for sh in range(shards)
                    for r in read(DEST / f"context_shard{sh}of{shards}.json")),
                   key=lambda r:r["pair_index"])
    assert len(context)==len(pairs)==768 and all(r["pair_index"]==i and r["held_valid"] and r["base_valid"]
                                             for i,r in enumerate(context))
    held=np.asarray([r["held"] for r in context],np.float32)
    wrong=np.asarray([r["wrong_base"] for r in context],np.float32)
    entities=rep.batch([rep.entities(s["physical"]) for s in states])
    # Keep the same padded obstacle axis as the original train-time tensor.
    old_x=dict(np.load(OLD / "entities.npz"))
    target_obstacles=old_x["obstacles"].shape[2]
    if entities["obstacles"].shape[2]<target_obstacles:
        pad=target_obstacles-entities["obstacles"].shape[2]
        entities["obstacles"]=np.pad(entities["obstacles"],((0,0),(0,0),(0,pad),(0,0)))
        entities["obstacle_mask"]=np.pad(entities["obstacle_mask"],((0,0),(0,pad)))
    si=np.asarray([p["state_index"] for p in pairs],int)
    x={k:jnp.asarray(v[si]) for k,v in entities.items()}
    predictions=[]
    for kind in ("eta_only","physical_context"):
        model=Critic(kind!="eta_only",kind!="eta_only",False)
        for seed in (17,23,41):
            folder=OUT / "models/ring_exchange" / kind / f"seed{seed}"
            norm=read(folder / "normalization.json")
            eta=(np.asarray([p["eta"] for p in pairs],np.float32)
                 -np.asarray(norm["eta_center"],np.float32))/np.asarray(norm["eta_scale"],np.float32)
            c0=np.asarray(norm["context_center"],np.float32)
            cs=np.asarray(norm["context_scale"],np.float32)
            first={k:v[:1] for k,v in x.items()}
            template=model.init(jax.random.PRNGKey(seed),first,jnp.zeros((1,3)),
                                jnp.zeros((1,24)),jnp.zeros((1,3)))
            params=serialization.from_bytes(template,(folder / "checkpoint.msgpack").read_bytes())
            for variant,raw in (("correct",held),("wrong_base",wrong)) if kind!="eta_only" else (("none",held),):
                c=(raw-c0)/cs
                z=[]
                for lo in range(0,len(pairs),128):
                    hi=min(len(pairs),lo+128)
                    chunk={k:v[lo:hi] for k,v in x.items()}
                    logits=np.asarray(model.apply(params,chunk,jnp.asarray(eta[lo:hi]),
                                                  jnp.asarray(c[lo:hi]),jnp.zeros((hi-lo,3))))
                    z.extend(np.asarray(jax.nn.sigmoid(logits),float).tolist())
                predictions.append({"kind":kind,"seed":seed,"context":variant,
                                    "checkpoint_sha256":hashlib.sha256((folder / "checkpoint.msgpack").read_bytes()).hexdigest(),
                                    "probabilities":z})
    write(target,{"schema":"held_true_t0_scores_frozen_before_Q16_v1",
                  "held_outcomes_present_at_score_time":False,
                  "pairs":[{"state_uid":p["state_uid"],"eta_uid":p["eta_uid"],"stage":p["stage"]} for p in pairs],
                  "models":predictions,
                  "physical_representation_replay": "PASS",
                  "context_correct_wrong_median_normalized_L2":float(np.median(np.linalg.norm(held-wrong,axis=1)))})
    print(json.dumps({"pairs":len(pairs),"model_conditions":len(predictions)}))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("physical","features","freeze"))
    parser.add_argument("--shard",type=int,default=0)
    parser.add_argument("--shards",type=int,default=8)
    args=parser.parse_args()
    if args.action=="physical":physical()
    elif args.action=="features":features(args.shard,args.shards)
    else:freeze(args.shards)
