"""Post-hoc Ring eta-only adaptation control on identical TRAIN families."""
from __future__ import annotations

import argparse
import json

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from scipy.special import expit

from diagnostics.orthoflow3_controller_context_loso_v1.evaluate import TARGET
from .fewshot_ring import DEST, SIZES, SEEDS, ranked_families
from .train import OUT, OLD, read, write


class EtaOnly(nn.Module):
    @nn.compact
    def __call__(self, eta):
        x = nn.silu(nn.Dense(64)(eta))
        x = nn.silu(nn.Dense(64)(x))
        return nn.Dense(1)(x)[..., 0]


def train(size, seed):
    dest = DEST / "eta_only" / f"n{size}" / f"seed{seed}"
    if (dest / "training.json").exists():return
    d = dict(np.load(OUT / "training_dataset.npz"))
    rows = read(OUT / "training_rows.json")
    family = set(ranked_families(rows)[:size])
    tr = np.asarray([i for i,r in enumerate(rows) if r["scene"] == "ring_exchange" and
                     r["role"] == "canonical_historical" and r["split"] == "train" and r["family"] in family], int)
    va = np.asarray([i for i,r in enumerate(rows) if r["scene"] == "ring_exchange" and
                     r["role"] == "canonical_historical" and r["split"] == "validation"], int)
    assert len(tr) and len(va)
    # Use the same source-fold eta normalization as the transferred model.
    norm = read(OUT / "models/ring/contrast_full" / f"seed{seed}" / "normalization.json")
    eta = (d["eta"] - np.asarray(norm["eta_center"], np.float32)) / np.asarray(norm["eta_radius"], np.float32)
    model = EtaOnly()
    params = model.init(jax.random.PRNGKey(seed), jnp.zeros((1,3)))
    optimizer = optax.chain(optax.clip_by_global_norm(5.), optax.adamw(6e-4, weight_decay=1e-4))
    state = optimizer.init(params)
    @jax.jit
    def step(p,o,xx,ss,ff):
        def loss(pp):
            z=model.apply(pp,xx)
            return jnp.mean((ss*jax.nn.softplus(-z)+ff*jax.nn.softplus(z))/16)
        value,grad=jax.value_and_grad(loss)(p)
        updates,o=optimizer.update(grad,o,p)
        return optax.apply_updates(p,updates),o,value
    predict=jax.jit(lambda p,x:model.apply(p,x))
    rng=np.random.default_rng(seed)
    best=(float("inf"),None,0);stale=0
    for it in range(1,1801):
        ix=rng.choice(tr,256)
        params,state,_=step(params,state,jnp.asarray(eta[ix]),jnp.asarray(d["s"][ix]),jnp.asarray(d["f"][ix]))
        if it%50:continue
        z=np.asarray(predict(params,jnp.asarray(eta[va])))
        nll=float((d["s"][va]*np.logaddexp(0,-z)+d["f"][va]*np.logaddexp(0,z)).sum() /
                  (d["s"][va]+d["f"][va]).sum())
        if nll<best[0]-1e-5:best=(nll,serialization.to_bytes(params),it);stale=0
        else:stale+=1
        if stale>=8 and it>=400:break
    dest.mkdir(parents=True,exist_ok=True)
    (dest/"checkpoint.msgpack").write_bytes(best[1])
    write(dest/"training.json",{"size_train_source_families":size,"seed":seed,"train_pairs":len(tr),
                                "best_step":best[2],"ring_val_NLL":best[0],
                                "classification":"POST_HOC_TARGET_SUPERVISED_ETA_ONLY_NOT_ZERO_SHOT",
                                "frozen_K16_TEST_not_used_for_selection":True})
    print(json.dumps({"size":size,"seed":seed,"val_NLL":best[0]}))


def evaluate():
    manifest=read(TARGET/"ring"/"manifest.json")
    truth=read(OLD/"cached_truth.json")["ring"]
    assert [m["state_uid"] for m in manifest]==[t["state_uid"] for t in truth]
    eta=np.asarray([m["eta"] for m in manifest],np.float32).reshape(-1,3)
    robust=np.asarray([[v is True for v in t["robust"]] for t in truth],bool)
    lower=np.asarray([t["lower"] for t in truth])
    output=[]
    for size in SIZES:
        for seed in SEEDS:
            folder=DEST/"eta_only"/f"n{size}"/f"seed{seed}"
            norm=read(OUT/"models/ring/contrast_full"/f"seed{seed}"/"normalization.json")
            ee=(eta-np.asarray(norm["eta_center"],np.float32))/np.asarray(norm["eta_radius"],np.float32)
            model=EtaOnly();empty=model.init(jax.random.PRNGKey(seed),jnp.zeros((1,3)))
            params=serialization.from_bytes(empty,(folder/"checkpoint.msgpack").read_bytes())
            z=np.asarray(model.apply(params,jnp.asarray(ee))).reshape(len(manifest),16)
            choice=z.argmax(1);ii=np.arange(len(manifest))
            output.append({"train_families":size,"seed":seed,"B15":int(robust[ii,choice].sum()),
                           "oracle_B15":int(robust.any(1).sum()),
                           "mean_selected_Q_lower":float(lower[ii,choice].mean()),
                           "mean_selected_probability":float(expit(z[ii,choice]).mean()),
                           "severe_FP":int(((expit(z[ii,choice])>.9)&(lower[ii,choice]<=.5)).sum()),
                           "POST_HOC_NOT_ZERO_SHOT":True})
    write(DEST/"eta_only/results.json",{"classification":"POST_HOC_TARGET_SUPERVISED_ETA_ONLY_NOT_ZERO_SHOT",
                                        "results":output})
    print(json.dumps(output))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=("train","evaluate"))
    parser.add_argument("--size",type=int,choices=SIZES);parser.add_argument("--seed",type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.action=="train":train(args.size,args.seed)
    else:evaluate()
