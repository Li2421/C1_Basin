"""Fixed-controller Q(s, eta), trained only from exact real rollout records.

Reuses the repository StateCritic with 24 physical inputs and 3 eta inputs.
The TEST prediction file must be frozen before any TEST outcomes are opened.
One future seed per pair is an engineering check, not B15 certification.
"""
from . import resources  # noqa: F401
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from diagnostics.physical_state_critic_formal_v1.train_ring_ff import StateCritic
from .experiment import dump, sha
from .pipeline import observation
from shared_rollout_db.src.rollout_db import connect, lookup_exact


def features(physical):
    return observation(physical["positions"], physical["velocities"], physical["goals"]).reshape(24)


def records(folder, protocol, split):
    rows = []
    for c in protocol["cases"]:
        if c["split"] != split: continue
        path = folder/"raw"/f"{c['rollout_uid']}.json"
        r = json.loads(path.read_text())
        for k in ("controller_uid", "state_uid", "eta_uid", "rollout_uid"):
            if r[k] != c[k]: raise ValueError(f"cache identity mismatch: {k}")
        if sha(r["trace_path"]) != r["trace_sha256"]: raise ValueError("changed trace")
        if r["numerical_failure"]: raise ValueError("numerical failures are unknown, not negative Q labels")
        rows.append(r)
    return rows


def fit(folder):
    out=folder/"critic"; out.mkdir(exist_ok=True)
    if (out/"fit.json").exists():
        info=json.loads((out/"fit.json").read_text())
        if sha(out/"checkpoint.msgpack")!=info["checkpoint_sha256"] or sha(folder/"protocol.json")!=info["protocol_sha256"]:
            raise ValueError("existing fit provenance mismatch")
        print("Frozen critic retained; no retraining."); freeze_predictions(folder); return
    p=json.loads((folder/"protocol.json").read_text())
    train=records(folder,p,"train"); dev=records(folder,p,"dev")
    states={s["uid"]:s for s in p["states"]}
    groups={name:{s["uid"] for s in p["states"] if s["split"]==name} for name in ("train","dev","test")}
    if any(groups[a]&groups[b] for a,b in (("train","dev"),("train","test"),("dev","test"))):
        raise ValueError("physical state leakage")
    history=dict(source_protocol=str((folder/"protocol.json").resolve()),controller_uid=p["controller"]["uid"],
                 reused_rollouts=len(train)+len(dev),new_rollouts=0,
                 old_critic="2-D StateCritic architecture reusable; old weights and 2-D labels incompatible with this 3-D controller",
                 test_labels_opened=False,training_outcomes=dict(Counter(r["outcome"] for r in train)),
                 source_records=[dict(path=r["source_record"],sha256=sha(r["source_record"])) for r in train+dev])
    dump(out/"history_audit.json",history)
    if len({bool(r["success"]) for r in train}) < 2:
        dump(out/"gate.json",dict(status="insufficient_success_contrast",trained=False,**history))
        raise ValueError("training success labels need both classes; no artificial Q labels")
    if not dev or not groups["test"]: raise ValueError("requires disjoint train/dev/test states")
    def pack(rows):
        return (np.stack([features(states[r["state_uid"]]["physical"]) for r in rows]),
                np.array([r["eta"] for r in rows],np.float32),np.array([r["success"] for r in rows],np.float32))
    x,e,y=pack(train); vx,ve,vy=pack(dev)
    mean=x.mean(0); scale=np.maximum(x.std(0),.1)
    x=(x-mean)/scale;vx=(vx-mean)/scale
    np.savez_compressed(out/"train_dev.npz",state=x,eta=e,success=y,dev_state=vx,dev_eta=ve,dev_success=vy,mean=mean,scale=scale)
    net=StateCritic(); params=net.init(jax.random.PRNGKey(2026),jnp.array(x[:1]),jnp.array(e[:1]))
    optimizer=optax.adam(1e-3); optim=optimizer.init(params)
    xx,ee,yy=map(jnp.asarray,(x,e,y))
    @jax.jit
    def block(params,optim):
        def one(carry,_):
            pp,oo=carry
            def loss(z):return optax.sigmoid_binary_cross_entropy(net.apply(z,xx,ee),yy).mean()
            value,grads=jax.value_and_grad(loss)(pp)
            updates,oo=optimizer.update(grads,oo,pp)
            return (optax.apply_updates(pp,updates),oo),value
        return jax.lax.scan(one,(params,optim),None,length=200)[0]
    predict=jax.jit(lambda w,s,t:net.apply(w,s,t))
    started=time.perf_counter(); best=None; best_bytes=None; log=[]
    ne=len(p["etas"])
    for step in range(200,1201,200):
        params,optim=block(params,optim)
        z=np.asarray(predict(params,jnp.asarray(vx),jnp.asarray(ve))).reshape(-1,ne)
        labels=vy.reshape(-1,ne); choice=z.argmax(1)
        regret=float(np.mean(labels.max(1)-labels[np.arange(len(choice)),choice]))
        nll=float(np.mean(np.logaddexp(0,z)-labels*z))
        key=(regret,nll,step);log.append(dict(step=step,selection_regret=regret,nll=nll))
        if best is None or key<best:best=key;best_bytes=serialization.to_bytes(params)
    cp=out/"checkpoint.msgpack";cp.write_bytes(best_bytes)
    np.savez_compressed(out/"normalization.npz",mean=mean,scale=scale)
    prior=np.array([np.mean([r["success"] for r in train if r["eta_index"]==i]) for i in range(ne)])
    dump(out/"fit.json",dict(controller_uid=p["controller"]["uid"],protocol_sha256=sha(folder/"protocol.json"),
         checkpoint_sha256=sha(cp),script_sha256=sha(__file__),train_pairs=len(train),dev_pairs=len(dev),
         train_states=len(groups["train"]),dev_states=len(groups["dev"]),steps=1200,best_step=best[2],
         train_best_fixed_eta=int(prior.argmax()),train_eta_success=prior.tolist(),log=log,
         gpu=False,wall_seconds=time.perf_counter()-started,test_labels_opened=False,
         scope="fixed controller; physical state and eta only; one future realization; no controller transfer or B15"))
    freeze_predictions(folder)


def freeze_predictions(folder):
    out=folder/"critic"; dest=out/"test_predictions.json"
    if dest.exists(): print("TEST predictions already frozen."); return
    # A future attempt cannot obtain a clean test by deleting an old prediction file.
    p=json.loads((folder/"protocol.json").read_text())
    if any((folder/"raw"/f"{c['rollout_uid']}.json").exists() for c in p["cases"] if c["split"]=="test"):
        raise ValueError("TEST outcomes already exist; predictions cannot be labeled fresh")
    # An independent Q encoding may reuse historical TEST identities. That is
    # cached evaluation, never a newly unseen experiment even in a new folder.
    with connect(True) as db:
        historical_test=sum(lookup_exact(db,c["state_uid"],c["eta_uid"],c["controller_uid"],[c["seed_key"]])["status"]!="INCOMPATIBLE"
                            for c in p["cases"] if c["split"]=="test")
    info=json.loads((out/"fit.json").read_text()); cp=out/"checkpoint.msgpack"
    if info["controller_uid"]!=p["controller"]["uid"] or sha(cp)!=info["checkpoint_sha256"]:
        raise ValueError("critic controller/checkpoint mismatch")
    net=StateCritic(); template=net.init(jax.random.PRNGKey(2026),jnp.zeros((1,24)),jnp.zeros((1,3)))
    params=serialization.from_bytes(template,cp.read_bytes()); norm=np.load(out/"normalization.npz")
    etas=np.asarray(p["etas"],np.float32); rows=[]
    for state in p["states"]:
        if state["split"]!="test": continue
        x=np.tile((features(state["physical"])-norm["mean"])/norm["scale"],(len(etas),1))
        z=np.asarray(net.apply(params,jnp.asarray(x),jnp.asarray(etas)))
        rows.append(dict(state_uid=state["uid"],eta_index=int(z.argmax()),eta=etas[z.argmax()].tolist(),logits=z.tolist()))
    dump(dest,dict(controller_uid=p["controller"]["uid"],checkpoint_sha256=sha(cp),
                   test_labels_opened=False,historical_test_pairs=historical_test,
                   fresh_test_evidence=historical_test==0,rows=rows,train_best_fixed_eta=info["train_best_fixed_eta"]))
    print(json.dumps(dict(critic=str(cp),frozen_test_states=len(rows))))


def evaluate(folder):
    out=folder/"critic"; predictions=out/"test_predictions.json"
    pred=json.loads(predictions.read_text()); p=json.loads((folder/"protocol.json").read_text())
    if pred["controller_uid"]!=p["controller"]["uid"]: raise ValueError("controller mismatch")
    rows=records(folder,p,"test"); lookup={(r["state_uid"],r["eta_index"]):r for r in rows}
    chosen=[lookup[r["state_uid"],r["eta_index"]] for r in pred["rows"]]
    zero=p["etas"].index([0,0,0]); fixed=pred["train_best_fixed_eta"]
    outcomes={}
    for name,group in [("Q_selected",chosen),("eta_zero",[lookup[s["state_uid"],zero] for s in pred["rows"]]),
                       ("train_best_fixed",[lookup[s["state_uid"],fixed] for s in pred["rows"]])]:
        outcomes[name]=dict(successes=sum(r["success"] for r in group),count=len(group),
                            outcomes=dict(Counter(r["outcome"] for r in group)))
    oracle=sum(any(lookup[s["state_uid"],i]["success"] for i in range(len(p["etas"]))) for s in pred["rows"])
    fixed_counts=[sum(lookup[s["state_uid"],i]["success"] for s in pred["rows"]) for i in range(len(p["etas"]))]
    result=dict(predictions_sha256=sha(predictions),heldout_states=len(pred["rows"]),test_pairs=len(rows),
                selectors=outcomes,oracle_successes=oracle,new_rollouts=0,
                all_continuous_collision_free=all(not r["collision"] for r in rows),
                selected_records=[r["source_record"] for r in chosen],
                minimum_wall_clearance=min(r["minimum_wall_clearance"] for r in chosen),
                minimum_pair_ellipsoid_gap=min(r["minimum_pair_ellipsoid_gap"] for r in chosen),
                q_benefit_over_fixed=outcomes["Q_selected"]["successes"]-outcomes["train_best_fixed"]["successes"],
                all_fixed_eta_successes=fixed_counts,
                choices=[s["eta_index"] for s in pred["rows"]],
                state_dependent_selection_benefit_demonstrated=False,
                fixed_baseline_note="train-best tie uses first index; another tied fixed eta gets the same TEST success as Q; do not claim adaptive benefit",
                limitation="six fresh initial states, one Flow future seed, fixed geometry/controller; not B15 or transfer evidence")
    dump(out/"evaluation.json",result); print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["fit","predict","evaluate"])
    parser.add_argument("--root",type=Path,default=Path("diagnostics/drone_3d_pipeline_v1/q_panel"))
    a=parser.parse_args(); {"fit":fit,"predict":freeze_predictions,"evaluate":evaluate}[a.command](a.root)
