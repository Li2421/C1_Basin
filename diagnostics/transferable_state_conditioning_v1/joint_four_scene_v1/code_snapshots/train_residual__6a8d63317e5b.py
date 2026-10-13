"""Test source-only eta-prior plus physical-state residual ranking hypothesis.

The eta-only prior is trained on exactly the same source labels and frozen.
Only physical entities and η enter the residual network. Pairwise loss uses
within-state source TRAIN labels; checkpoint selection uses source VAL only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import EtaOnly as WideEtaOnly
from diagnostics.physical_state_critic_formal_v1.train_unified import select_key
from diagnostics.transferable_state_conditioning_v1.assemble_source import HERE, PANELS, read, sha
from diagnostics.transferable_state_conditioning_v1.train_baselines import SEEDS, NEAR_TIE, metrics

HERE = Path(os.environ.get("TRANSFER_Q_WORK_ROOT", str(HERE)))
STEPS = tuple(range(200, int(os.environ.get("TRANSFER_Q_MAX_STEPS", "4000")) + 1, 200))
PAIR_WEIGHT = .25
RESIDUAL_L2 = .01


class StateEtaResidual(nn.Module):
    @nn.compact
    def __call__(self, x, eta):
        """x: B physical states; eta: B×K×3; returns B×K residual logits."""
        h = rep.Encoder(name="physical_encoder")(x)
        state = nn.tanh(nn.Dense(64, name="state_projection")(h))[:,None,:]
        e = nn.silu(nn.Dense(32, name="eta_encoder")(eta))
        eta_h = nn.tanh(nn.Dense(64, name="eta_projection")(e))
        interaction = state * eta_h
        z = nn.silu(nn.Dense(64, name="interaction_hidden")(interaction))
        return nn.Dense(1, name="residual_out")(z)[...,0]


def selected_prior(panel):
    runs = [read(HERE / "fits" / panel / "eta_only_wide" / f"seed{s}" / "complete.json") for s in SEEDS]
    return min(runs, key=lambda r: (*select_key(r["selected"], NEAR_TIE), r["seed"]))


def prior_logits(panel, eta):
    run = selected_prior(panel)
    ck = HERE / "fits" / panel / "eta_only_wide" / f"seed{run['seed']}" / "checkpoint.msgpack"
    assert sha(ck) == run["checkpoint_sha256"]
    net = WideEtaOnly()
    dummy = {"agents":jnp.zeros((1,1,15)),"pairs":jnp.zeros((1,1,1,9)),
             "obstacles":jnp.zeros((1,1,1,13)),"globals":jnp.zeros((1,15)),
             "agent_mask":jnp.ones((1,1)),"obstacle_mask":jnp.ones((1,1))}
    template = net.init(jax.random.PRNGKey(run["seed"]), dummy, jnp.zeros((1,3)))
    params = serialization.from_bytes(template, ck.read_bytes())
    flat = np.asarray(eta, np.float32).reshape(-1,3)
    scores = []
    for a in range(0,len(flat),1024):
        part=jnp.asarray(flat[a:a+1024])
        scores.append(np.asarray(net.apply(params,dummy,part)))
    out=np.concatenate(scores).reshape(eta.shape[:2])
    assert np.isfinite(out).all()
    return out,run


def fit(panel,seed):
    assert panel in PANELS and seed in SEEDS
    data_path = HERE / "source_data" / panel / "trainval.npz"
    manifest = read(HERE / "source_data" / panel / "manifest.json")
    assert sha(data_path)==manifest["trainval_sha256"] and not manifest["test_labels_opened"]
    folder=HERE/"fits"/panel/"state_residual"/f"seed{seed}"
    if (folder/"complete.json").exists():
        done=read(folder/"complete.json")
        assert done["trainval_sha256"]==sha(data_path)
        assert done["checkpoint_sha256"]==sha(folder/"checkpoint.msgpack")
        return done
    folder.mkdir(parents=True,exist_ok=True)
    d=np.load(data_path)
    train=np.flatnonzero(d["split"]=="train")
    val=np.flatnonzero(d["split"]=="validation")
    eta=jnp.asarray(d["eta"],jnp.float32)
    success=jnp.asarray(d["success"],jnp.float32)
    failure=jnp.asarray(d["failure"],jnp.float32)
    numerical=np.asarray(d["numerical"])
    scenes=np.asarray(d["scene"])
    x={key[7:]:jnp.asarray(d[key]) for key in d.files if key.startswith("entity_")}
    prior_np,prior_run=prior_logits(panel,d["eta"])
    base=jnp.asarray(prior_np,jnp.float32)
    net=StateEtaResidual()
    def gather(ii):return {k:v[ii] for k,v in x.items()}
    params=net.init(jax.random.PRNGKey(seed),gather(jnp.array([0])),eta[:1])
    optimizer=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0006,weight_decay=.0002))
    opt_state=optimizer.init(params)

    @jax.jit
    def block(weights,ostate,draws):
        def step(carry,ii):
            p,o=carry
            def loss(pp):
                residual=net.apply(pp,gather(ii),eta[ii])
                z=base[ii]+residual
                k=success[ii];f=failure[ii]
                bce=jnp.sum(k*jax.nn.softplus(-z)+f*jax.nn.softplus(z))/jnp.maximum(jnp.sum(k+f),1.)
                empirical=k/jnp.maximum(k+f,1.)
                diff=empirical[:,:,None]-empirical[:,None,:]
                weight=jnp.abs(diff)*(jnp.abs(diff)>=.5)
                logit=z[:,:,None]-z[:,None,:]
                pair=jnp.sum(weight*jax.nn.softplus(-jnp.sign(diff)*logit))/jnp.maximum(jnp.sum(weight),1.)
                return bce+PAIR_WEIGHT*pair+RESIDUAL_L2*jnp.mean(residual**2)
            value,grad=jax.value_and_grad(loss)(p)
            updates,o=optimizer.update(grad,o,p)
            return (optax.apply_updates(p,updates),o),value
        return jax.lax.scan(step,(weights,ostate),draws)

    @jax.jit
    def predict(weights,ii):
        return base[ii]+net.apply(weights,gather(ii),eta[ii])

    rng=np.random.default_rng(seed)
    pools={sc:train[scenes[train]==sc] for sc in sorted(set(scenes[train]))}
    assert all(len(pool)>=32 for pool in pools.values())
    best=None;best_bytes=None;history=[];prior_step=0;started=time.monotonic()
    for step in STEPS:
        draws=[]
        for k,pool in enumerate(pools.values()):
            amount=16//len(pools)+(k<16%len(pools))
            draws.append(pool[rng.integers(len(pool),size=(step-prior_step,amount))])
        sample=np.concatenate(draws,axis=1)
        (params,opt_state),_=block(params,opt_state,jnp.asarray(sample))
        val_logits=[]
        for a in range(0,len(val),64):
            val_logits.append(np.asarray(predict(params,jnp.asarray(val[a:a+64]))))
        z=np.concatenate(val_logits)
        score=metrics(z,np.asarray(success)[val],np.asarray(failure)[val],numerical[val],scenes[val])
        row={"step":step,**score}
        history.append(row)
        if best is None or select_key(row,NEAR_TIE)<select_key(best,NEAR_TIE):
            best=row;best_bytes=serialization.to_bytes(params)
        prior_step=step
        if step%1000==0:
            print(json.dumps({"panel":panel,"seed":seed,"step":step,
                              "val_regret":row["selection_regret"],"best_step":best["step"],
                              "elapsed_seconds":time.monotonic()-started}),flush=True)
    (folder/"checkpoint.msgpack").write_bytes(best_bytes)
    (folder/"history.json").write_text(json.dumps(history,indent=2)+"\n")
    done={"panel":panel,"kind":"state_residual","seed":seed,
          "optimization_steps_max":max(STEPS),
          "selected":best,"trainval_sha256":sha(data_path),
          "checkpoint_sha256":sha(folder/"checkpoint.msgpack"),
          "eta_prior_seed":prior_run["seed"],
          "eta_prior_checkpoint_sha256":prior_run["checkpoint_sha256"],
          "pairwise_weight":PAIR_WEIGHT,"residual_l2":RESIDUAL_L2,
          "target_labels_used":False,"elapsed_seconds":time.monotonic()-started,
          "backend":jax.default_backend()}
    (folder/"complete.json").write_text(json.dumps(done,indent=2)+"\n")
    return done


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--panel",choices=tuple(PANELS),required=True)
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    a=p.parse_args()
    print(json.dumps(fit(a.panel,a.seed)),flush=True)
