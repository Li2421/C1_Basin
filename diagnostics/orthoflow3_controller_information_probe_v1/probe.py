"""Source-family held-out diagnostic: is controller information learnable?

All variants see the same matched eta/trial evidence. Controller identity is
an intentionally privileged *diagnostic upper bound*, never a zero-shot input.
No target TEST, generator, or rollout DB write is involved.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from itertools import combinations
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
import flax.linen as nn

from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "diagnostics/orthoflow3_controller_intervention_generalization_v1"
OUT = Path(__file__).resolve().parent
SCENES = ("toy_giveway", "double_bottleneck", "four_way_intersection", "ring_exchange")
KINDS = ("eta_only", "controller_eta", "physical_context", "physical_context_plus_id")
SEEDS = (17, 23, 41)
CONTROLLERS = ("base", "alt", "second")
KEYS = ("agents", "pairs", "obstacles", "globals", "agent_mask", "obstacle_mask")


def read(path): return json.loads(Path(path).read_text())
def write(path, obj):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


class Critic(nn.Module):
    use_state: bool
    use_context: bool
    use_id: bool

    @nn.compact
    def __call__(self, x, eta, context, controller_id):
        # Identical architecture/parameter initialization in each variant.
        h = rep.Encoder(name="physical_encoder")(x)
        if not self.use_state: h = jnp.zeros_like(h)
        e = nn.silu(nn.Dense(32, name="eta_encoder")(eta))
        c = nn.silu(nn.Dense(32, name="context_encoder")(context if self.use_context else jnp.zeros_like(context)))
        cid = nn.silu(nn.Dense(16, name="id_encoder")(controller_id if self.use_id else jnp.zeros_like(controller_id)))
        z = nn.silu(nn.Dense(128, name="trunk1")(jnp.concatenate((h, e, c, cid), -1)))
        z = nn.silu(nn.Dense(64, name="trunk2")(z))
        return nn.Dense(1, name="out")(z)[..., 0]


def load_scene(scene, include_crossmatrix=False):
    d = dict(np.load(SRC / "second_variant/triplet_dataset_h20.npz"))
    rows = read(SRC / "rich_probe_rows.json")
    ids = np.asarray([i for i,r in enumerate(rows) if r["scene"] == scene], int)
    base_rows=[rows[i] for i in ids]
    base_split=np.asarray([r["split"] for r in base_rows])
    assert len({r["state_uid"] for r in np.asarray(base_rows,dtype=object)[base_split=="train"]} &
               {r["state_uid"] for r in np.asarray(base_rows,dtype=object)[base_split=="validation"]}) == 0
    base_eta = d["eta"][ids]
    base_tr = base_split == "train"
    center = base_eta[base_tr].mean(0); scale = np.maximum(base_eta[base_tr].std(0), .1)
    base_context = np.stack([d[f"{c}_c"][ids] for c in CONTROLLERS], axis=0)
    # Use observed source TRAIN controller contexts only. Toy's third condition
    # has no outcomes and is excluded from normalization.
    base_valid = np.stack([(d[f"{c}_s"][ids]+d[f"{c}_f"][ids]) > 0 for c in CONTROLLERS])
    observed_context = base_context[:, base_tr, :][base_valid[:, base_tr]]
    ccenter = observed_context.mean(0); cscale = np.maximum(observed_context.std(0), .05)
    split=base_split
    raw_eta=base_eta
    scene_context=base_context
    successes = np.stack([d[f"{c}_s"][ids] for c in CONTROLLERS]).astype(np.float32)
    failures = np.stack([d[f"{c}_f"][ids] for c in CONTROLLERS]).astype(np.float32)
    scene_rows=base_rows
    state_index=d["state_index"][ids]
    if include_crossmatrix:
        assert scene=="ring_exchange"
        extra=read(OUT/"crossmatrix_ring_v2/pair_counts.json")["new_pairs_only"]
        train_uids={r["state_uid"] for r in np.asarray(base_rows,dtype=object)[base_tr]}
        val_uids={r["state_uid"] for r in np.asarray(base_rows,dtype=object)[base_split=="validation"]}
        assert all(r["state_uid"] in train_uids and r["state_uid"] not in val_uids for r in extra)
        scene_rows=base_rows + [{"scene":scene,"split":"train","state_uid":r["state_uid"],
                                 "eta_uid":r["eta_uid"]} for r in extra]
        split=np.concatenate((base_split,np.repeat("train",len(extra))))
        raw_eta=np.concatenate((base_eta,np.asarray([r["eta"] for r in extra],np.float32)))
        state_index=np.concatenate((state_index,np.asarray([r["state_index"] for r in extra],int)))
        scene_context=np.concatenate((base_context,np.zeros((3,len(extra),24),np.float32)),axis=1)
        successes=np.concatenate((successes,np.asarray([[r[f"controller{c}_success"] for r in extra]
                                                         for c in range(3)],np.float32)),axis=1)
        failures=np.concatenate((failures,np.asarray([[r[f"controller{c}_failure"] for r in extra]
                                                      for c in range(3)],np.float32)),axis=1)
    eta=(raw_eta-center)/scale
    context=(scene_context-ccenter)/cscale
    valid=(successes+failures)>0
    # Every pair appears under each compatible controller, with observed trials
    # only. An invalid/numerical condition contributes zero trials.
    x = {k:d[k] for k in KEYS}
    return {"d":d,"rows":scene_rows,"ids":ids,"split":split,"eta":eta.astype(np.float32),
            "context":context.astype(np.float32),"success":successes,"failure":failures,"valid":valid,
            "x":x,"state_index":state_index,"eta_center":center,"eta_scale":scale,
            "context_center":ccenter,"context_scale":cscale}


def observed_nll(logits, s, f):
    return float((s*np.logaddexp(0,-logits)+f*np.logaddexp(0,logits)).sum()/max(1,(s+f).sum()))


def score_selection(z, scene, data, split):
    r = data["rows"]; ix=np.flatnonzero(data["split"]==split)
    by_state={}
    for i in ix:by_state.setdefault(r[i]["state_uid"],[]).append(i)
    counts = []
    for controller in range(3):
        if scene=="toy_giveway" and controller==2: continue
        available=correct=unknown=0; lower=[];upper=[]
        for state, pairs in by_state.items():
            trial = data["success"][controller,pairs]+data["failure"][controller,pairs]
            lo = data["success"][controller,pairs]/16
            hi = (16-data["failure"][controller,pairs])/16
            # Fifteen observed successes already certify B15 even if the
            # sixteenth seed was never run; two failures certify non-B15.
            robust = data["success"][controller,pairs] >= 15
            if not robust.any(): continue
            available+=1
            chosen=pairs[int(np.argmax(z[controller,pairs]))]
            jj=pairs.index(chosen)
            correct += int(bool(robust[jj]))
            unknown += int(not robust[jj] and data["failure"][controller,chosen]<2 and trial[jj]<16)
            lower.append(float(lo[jj]));upper.append(float(hi[jj]))
        counts.append({"controller":controller,"available_B15_states":available,"selected_B15_states":correct,
                       "selected_unknown_states":unknown,
                       "mean_selected_Q16_lower":float(np.mean(lower)) if lower else None,
                       "mean_selected_Q16_upper":float(np.mean(upper)) if upper else None})
    return counts


def reversal_audit(z, data):
    # Conservative interval comparison. Identical eta candidates under each
    # controller; no Q16 imputation of partial counts.
    ix=np.flatnonzero(data["split"]=="validation")
    rows=data["rows"]
    by_state={}
    for i in ix:by_state.setdefault(rows[i]["state_uid"],[]).append(i)
    total=correct=sign_change=0
    for pairs in by_state.values():
        for a,b in combinations(pairs,2):
            true=[]
            for c in range(3):
                sa,fa=data["success"][c,a],data["failure"][c,a]
                sb,fb=data["success"][c,b],data["failure"][c,b]
                lo=(sa-(16-fb))/16; hi=((16-fa)-sb)/16
                true.append(1 if lo>0 else -1 if hi<0 else 0)
            for c0,c1 in combinations(range(3),2):
                if true[c0]*true[c1] >= 0:continue
                total+=1
                p0=z[c0,a]-z[c0,b];p1=z[c1,a]-z[c1,b]
                correct+=int(np.sign(p0)==true[c0] and np.sign(p1)==true[c1])
                sign_change+=int(p0*p1<0)
    return {"certified_reversal_cases":total,"both_rankings_correct":correct,"predicted_sign_reversal":sign_change}


def train(scene, kind, seed, heldout_controller=None, crossmatrix=False):
    suffix=f"_heldout_controller{heldout_controller}" if heldout_controller is not None else ""
    if crossmatrix:suffix+="_crossmatrix"
    dest=OUT/"models"/scene/(kind+suffix)/f"seed{seed}"
    if (dest/"summary.json").exists() and read(dest/"summary.json").get("b15_rule") == "observed_successes_at_least_15":return
    data=load_scene(scene,include_crossmatrix=crossmatrix)
    if kind.startswith("fingerprint_"):
        bank=read(OUT/f"fingerprint_bank_{scene}.json")
        fingerprint=np.asarray(bank["vectors"],np.float32)
        source_fingerprint=np.delete(fingerprint,heldout_controller,axis=0) if heldout_controller is not None else fingerprint
        center=source_fingerprint.mean(0)
        scale=np.maximum(source_fingerprint.std(0),.05)
        data["context"]=np.broadcast_to(((fingerprint-center)/scale)[:,None,:],
                                           (3,len(data["eta"]),24)).copy()
        data["context_center"],data["context_scale"]=center,scale
    variant={"eta_only":(False,False,False),"controller_eta":(False,False,True),
             "physical_context":(True,True,False),"physical_context_plus_id":(True,True,True),
             "fingerprint_eta":(False,True,False),"fingerprint_full":(True,True,False)}[kind]
    model=Critic(*variant)
    x=data["x"];si=data["state_index"];eta=data["eta"]
    cx=data["context"];s=data["success"];f=data["failure"]
    ctl=np.broadcast_to(np.eye(3,dtype=np.float32)[:,None,:],(3,len(eta),3))
    template=model.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,24)),jnp.zeros((1,3)))
    params=template
    optimizer=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(8e-4,weight_decay=1e-4))
    state=optimizer.init(params)

    @jax.jit
    def step(p,os,xx,ee,cc,jj,ss,ff):
        def loss(pp):
            z=model.apply(pp,xx,ee,cc,jj)
            return jnp.sum(ss*jax.nn.softplus(-z)+ff*jax.nn.softplus(z))/jnp.maximum(1,jnp.sum(ss+ff))
        value,grads=jax.value_and_grad(loss)(p)
        upd,os=optimizer.update(grads,os,p)
        return optax.apply_updates(p,upd),os,value

    predict=jax.jit(lambda p,xx,ee,cc,jj:model.apply(p,xx,ee,cc,jj))
    def all_scores(p,subset):
        values=np.zeros((3,len(eta)),np.float32)
        for c in range(3):
            active=subset & data["valid"][c]
            ix=np.flatnonzero(active)
            if len(ix):values[c,ix]=np.asarray(predict(p,gather(x,si[ix]),jnp.asarray(eta[ix]),jnp.asarray(cx[c,ix]),jnp.asarray(ctl[c,ix])))
        return values

    tr=np.flatnonzero(data["split"]=="train")
    va=data["split"]=="validation"
    allowed=np.asarray([c for c in range(3) if c!=heldout_controller],int)
    rng=np.random.default_rng(seed)
    best=(float("inf"),None,0)
    stale=0; history=[]
    for iteration in range(1,1501):
        draw=rng.choice(tr,32)
        ci=np.repeat(allowed,len(draw)); pi=np.tile(draw,len(allowed))
        params,state,value=step(params,state,gather(x,si[pi]),jnp.asarray(eta[pi]),
                                jnp.asarray(cx[ci,pi]),jnp.asarray(ctl[ci,pi]),
                                jnp.asarray(s[ci,pi]),jnp.asarray(f[ci,pi]))
        if iteration%100:continue
        zz=all_scores(params,va)
        metric=observed_nll(zz[allowed][:,va],s[allowed][:,va],f[allowed][:,va])
        history.append({"step":iteration,"train":float(value),"validation_NLL":metric})
        if metric<best[0]-1e-5:
            best=(metric,serialization.to_bytes(params),iteration);stale=0
        else:stale+=1
        if stale>=7 and iteration>=700:break
    params=serialization.from_bytes(template,best[1]);z=all_scores(params,va)
    results={"scene":scene,"kind":kind,"seed":seed,"best_step":best[2],
             "val_observed_NLL":best[0],"train_states":len({data['rows'][i]['state_uid'] for i in tr}),
             "validation_states":len({data['rows'][i]['state_uid'] for i in np.flatnonzero(va)}),
             "validation_selection":score_selection(z,scene,data,"validation"),
             "validation_reversals":reversal_audit(z,data),
             "b15_rule":"observed_successes_at_least_15",
             "controller_context_used":variant[1],"state_used":variant[0],"controller_identity_used":variant[2],
             "heldout_controller":heldout_controller,"crossmatrix_added":crossmatrix,
             "target_TEST_used":False,"new_rollout":0}
    results["context_source"]="fixed_source_TRAIN_controller_response_bank" if kind.startswith("fingerprint_") else "H20_local_response"
    dest.mkdir(parents=True,exist_ok=True)
    (dest/"checkpoint.msgpack").write_bytes(best[1])
    write(dest/"summary.json",results)
    write(dest/"normalization.json",{"eta_center":data['eta_center'].tolist(),"eta_scale":data['eta_scale'].tolist(),
                                      "context_center":data['context_center'].tolist(),"context_scale":data['context_scale'].tolist()})
    write(dest/"history.json",history)
    np.savez_compressed(dest/"validation_predictions.npz",logits=z,eta=data['eta'],
                        state_index=data['state_index'],split=data['split'])
    print(json.dumps({"scene":scene,"kind":kind,"seed":seed,"val_NLL":best[0],
                      "selection":results['validation_selection'],"reversal":results['validation_reversals']}),flush=True)


def summarize():
    out=[]
    for scene in SCENES:
        for kind in KINDS:
            for seed in SEEDS:
                result=read(OUT/"models"/scene/kind/f"seed{seed}/summary.json")
                for row in result["validation_selection"]:
                    out.append({"scene":scene,"kind":kind,"seed":seed,
                                "val_NLL":result["val_observed_NLL"],**row,
                                **result["validation_reversals"]})
    with (OUT/"source_family_validation.csv").open("w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    write(OUT/"protocol.json",{"data":"existing triplet_dataset_h20.npz, observed success/failure counts only",
        "source_family_split":"inherited 12 TRAIN and 4 independent VAL states per scene",
        "variants":KINDS,"seeds":SEEDS,
        "loss":"observed continuation Bernoulli likelihood; no unrun-seed Q imputation",
        "checkpoint":"same independent family VAL NLL in each variant",
        "controller_identity":"within-scene privileged diagnostic upper bound; never deployed zero-shot",
        "new_rollout":0,"target_TEST_used":False,"generator_changed":False})
    print(json.dumps({"finished_runs":len(SCENES)*len(KINDS)*len(SEEDS),"csv":"source_family_validation.csv"}))


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("action",choices=("train","summarize"))
    ap.add_argument("--scene",choices=SCENES);ap.add_argument("--kind",choices=KINDS+("fingerprint_eta","fingerprint_full"))
    ap.add_argument("--seed",type=int,choices=SEEDS)
    ap.add_argument("--heldout-controller",type=int,choices=(2,))
    ap.add_argument("--crossmatrix",action="store_true");args=ap.parse_args()
    if args.action=="train":
        assert args.scene and args.kind and args.seed
        if args.heldout_controller is not None and args.kind not in ("fingerprint_eta","fingerprint_full"):
            ap.error("heldout-controller is reserved for fingerprint diagnostics")
        if args.crossmatrix and args.scene!="ring_exchange":
            ap.error("crossmatrix applies only to Ring")
        train(args.scene,args.kind,args.seed,args.heldout_controller,args.crossmatrix)
    else:summarize()
