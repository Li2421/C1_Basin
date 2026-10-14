"""Fit A3 on 92 historical controllers only; never imports Round17 success labels."""
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE","false")
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):os.environ.setdefault(k,"1")
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
R16=ROOT/"diagnostics/ring_controller_critic_v1"
sys.path.insert(0,str(R16))
import fit as oldfit
jax.config.update("jax_enable_x64",False)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n")

def cohorts():
    src=oldfit.source_data()
    yield dict(tag="R15",names=src["names"],states=src["states"],physical=src["physical"],r5=src["r5"],
               y=src["y"],n=src["n"],arms=oldfit.ARMS,available=np.ones((56,8),bool),
               label_source=ROOT/"diagnostics/ring_controller_population_v1/dataset.jsonl",
               feature_source=ROOT/"diagnostics/ring_controller_population_v1/R5_FEATURES.npz")
    for tag,(folder,featfile,key) in oldfit.EXT.items():
        with np.load(folder/"LABELS.npz",allow_pickle=False) as z:
            names=list(z["controller_names"] if "controller_names" in z.files else z["controllers"])
            arms=list(z["arms"])
            success=z["success"].astype(np.float32);unknown=z["unknown"].astype(bool)
            available=z["available"].astype(bool) if "available" in z.files else np.ones((len(names),len(arms)),bool)
        if featfile:
            with np.load(folder/featfile,allow_pickle=False) as z:
                feature_names=list(z["controller_names"]);r5=z[key].astype(np.float32)
        else:
            feature_source=ROOT/"diagnostics/controller_behavior_repr_v1/FEATURES.npz"
            with np.load(feature_source,allow_pickle=False) as z:
                feature_names=list(z["controller_names"]);r5=z[key].astype(np.float32)
        r5=r5[[feature_names.index(n) for n in names]]
        states=json.loads((folder/"FRESH_STATES.json").read_text())
        y=np.transpose((success*(~unknown)).sum(-1),(0,2,1)).astype(np.float32)
        n=np.transpose((~unknown).sum(-1),(0,2,1)).astype(np.float32)
        yield dict(tag=tag,names=names,states=states,physical=oldfit.physical23(states).astype(np.float32),r5=r5,
                   y=y,n=n,arms=arms,available=available,label_source=folder/"LABELS.npz",
                   feature_source=folder/featfile if featfile else feature_source)

def transform(cohorts):
    raw=np.concatenate([np.repeat(c["physical"][None],len(c["names"]),axis=0).reshape(-1,23) for c in cohorts])
    sm=raw.mean(0);ss=np.where(raw.std(0)>1e-6,raw.std(0),1)
    r5=np.concatenate([c["r5"] for c in cohorts]);r5=np.delete(r5,np.s_[105:110],axis=1)
    keep=np.isfinite(r5).any(0);r5=r5[:,keep]
    mu=np.nanmean(np.where(np.isfinite(r5),r5,np.nan),axis=0)
    fixed=np.where(np.isfinite(r5),r5,mu);scale=np.where(fixed.std(0)>1e-6,fixed.std(0),1)
    _,_,vt=np.linalg.svd((fixed-mu)/scale,full_matrices=False)
    pc=vt[:2].copy()
    for v in pc:
        if v[np.argmax(np.abs(v))]<0:v*=-1
    return dict(state_mean=sm.astype(np.float32),state_scale=ss.astype(np.float32),r5_keep=keep,
                r5_mean=mu.astype(np.float32),r5_scale=scale.astype(np.float32),pca=pc.astype(np.float32))

def main():
    reg=json.loads((HERE/"PREREGISTRATION.receipt.json").read_text())
    assert sha(HERE/"PREREGISTRATION.md")==reg["sha256"]
    assert not (HERE/"NEW_LABELS_STARTED.receipt.json").exists()
    c=list(cohorts())
    assert [len(x["names"]) for x in c]==[56,14,14,8]
    assert len({n for x in c for n in x["names"]})==92
    tr=transform(c)
    out=HERE/"A3_ALL";out.mkdir(exist_ok=True)
    np.savez_compressed(out/"transform.npz",**tr)
    xs=[];ys=[];ns=[];counts={};inputs={}
    for group in c:
        x=oldfit.features(group["physical"],group["r5"],None,tr,"A3")
        count=0
        for j,arm in enumerate(group["arms"]):
            if arm not in oldfit.ARMS:continue
            ai=oldfit.ARMS.index(arm)
            mask=group["available"][:,j,None] & (group["n"][:,:,j]>0)
            xx=x[:,:,ai][mask];yy=group["y"][:,:,j][mask];nn=group["n"][:,:,j][mask]
            xs.append(xx);ys.append(yy);ns.append(nn);count+=len(xx)
        counts[group["tag"]]=dict(controllers=len(group["names"]),valid_cells=count,known_seed_labels=int(np.sum(group["n"])))
        inputs[group["tag"]]=dict(labels=sha(group["label_source"]),features=sha(group["feature_source"]))
    X=np.concatenate(xs).astype(np.float32);Y=np.concatenate(ys).astype(np.float32);N=np.concatenate(ns).astype(np.float32)
    assert X.shape[1]==33 and len(X)==sum(z["valid_cells"] for z in counts.values())
    net=oldfit.net_for("A3")
    opt=optax.chain(optax.clip_by_global_norm(5),optax.adamw(.0008,weight_decay=.0001))
    x=jnp.asarray(X);y=jnp.asarray(Y);n=jnp.asarray(N)
    @jax.jit
    def run(params,state,ids):
        def step(carry,idx):
            p,st=carry
            def loss(pp):
                logits=net.apply(pp,x[idx])
                return jnp.mean((y[idx]*jax.nn.softplus(-logits)+(n[idx]-y[idx])*jax.nn.softplus(logits))/4)
            grads=jax.grad(loss)(p)
            updates,st=opt.update(grads,st,p)
            return (optax.apply_updates(p,updates),st),None
        return jax.lax.scan(step,(params,state),ids)[0]
    results=[]
    for seed in oldfit.SEEDS:
        rng=np.random.default_rng(seed)
        ids=rng.integers(len(X),size=(1000,128),dtype=np.int32)
        p=net.init(jax.random.PRNGKey(seed),jnp.zeros((1,33),jnp.float32));state=opt.init(p)
        p,_=run(p,state,jnp.asarray(ids))
        ck=out/f"A3_all_seed{seed}_step1000.msgpack";ck.write_bytes(serialization.to_bytes(p))
        logits=np.asarray(net.apply(p,x));nll=float(np.sum(Y*np.logaddexp(0,-logits)+(N-Y)*np.logaddexp(0,logits))/np.sum(N))
        results.append(dict(seed=seed,checkpoint=str(ck.relative_to(HERE)),checkpoint_sha256=sha(ck),source_train_NLL=nll))
        print(json.dumps(results[-1]),flush=True)
    write(out/"TRAINING_AUDIT.json",dict(status="FITTED_BEFORE_NEW_SUCCESS_LABELS",utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
          controllers=92,groups=counts,input_sha256=inputs,transform_sha256=sha(out/"transform.npz"),
          model="Round16 A3 64/64/16",selected_step=1000,lr=.0008,weight_decay=.0001,batch_size=128,
          fit_seeds=oldfit.SEEDS,known_labels_exclude_unknown=True,unavailable_arms_excluded=True,
          checkpoint_results=results,new_success_labels_read=False))

if __name__=="__main__":main()
