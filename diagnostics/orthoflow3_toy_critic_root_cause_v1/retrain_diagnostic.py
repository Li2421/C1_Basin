#!/usr/bin/env python3
"""Small preregistered pure-NLL data/capacity audit. No rollouts or TEST tuning."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import flax.linen as nn
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.special import expit
jax.config.update('jax_enable_x64', True)

ROOT=Path('/home/zhihan/research/Basin_C1')
D=ROOT/'diagnostics'
OUT=Path(__file__).resolve().parent
W=D/'orthoflow3_nll_weighting_ablation_v1'
R=D/'orthoflow3_ranking_aware_critic_v1'
U=D/'orthoflow3_toy_critic_uncertainty_local_v1'
K=D/'orthoflow3_mode_free_k_sweep_latency_v1'
spec=importlib.util.spec_from_file_location('weighting_audit_lib',W/'run_experiment.py')
lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
rank=lib.lib

class Critic(nn.Module):
    size: str
    @nn.compact
    def __call__(self,h,eta):
        dims={'small':(64,32,16,64,32),'current':(128,64,32,128,64),'larger':(192,96,48,192,96)}[self.size]
        z=nn.silu(nn.Dense(dims[0],name='state1')(h))
        z=nn.silu(nn.Dense(dims[1],name='state2')(z))
        e=nn.silu(nn.Dense(dims[2],name='eta1')(eta))
        x=jnp.concatenate([z,e],axis=-1)
        x=nn.silu(nn.Dense(dims[3],name='c1')(x))
        x=nn.silu(nn.Dense(dims[4],name='c2')(x))
        return nn.Dense(1,name='out')(x)[...,0]

def pick(rows,frac,source):
    if frac==1:return np.arange(len(rows))
    by={}
    for i,r in enumerate(rows):by.setdefault(r['state_uid'],[]).append(i)
    chosen=[]
    for sid,ix in sorted(by.items()):
        ix.sort(key=lambda i:hashlib.sha256(f"critic_root_cause_v1|{source}|{sid}|{rows[i]['eta_uid']}".encode()).hexdigest())
        chosen+=ix[:max(1,int(round(frac*len(ix))))]
    return np.array(sorted(chosen))

def train(config,struct,wide,val):
    frac,size,seed=config
    six=pick(struct['rows'],frac,'structured')
    wix=pick(wide['rows'],frac,'wide')
    model=Critic(size=size)
    par=model.init(jax.random.PRNGKey(seed),jnp.zeros((1,214)),jnp.zeros((1,3)))
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(1e-3,weight_decay=1e-4))
    state=opt.init(par)
    @jax.jit
    def step(p,s,h,e,y):
        def loss(x):return jnp.mean(optax.sigmoid_binary_cross_entropy(model.apply(x,h,e),y))
        value,g=jax.value_and_grad(loss)(p)
        update,s=opt.update(g,s,p)
        return optax.apply_updates(p,update),s,value
    rng=np.random.default_rng(seed)
    best=(float('inf'),None,0);stale=0;frozen=False;trajectory=[]
    for it in range(1,4001):
        si=six[rng.integers(0,len(six),64)]
        wi=wix[rng.integers(0,len(wix),64)]
        hh=np.concatenate([struct['h'][si],wide['h'][wi]])
        ee=np.concatenate([struct['eta'][si],wide['eta'][wi]])
        yy=np.concatenate([struct['y'][si],wide['y'][wi]])
        par,state,_=step(par,state,jnp.asarray(hh),jnp.asarray(ee),jnp.asarray(yy))
        if it%50:continue
        logits=np.asarray(model.apply(par,jnp.asarray(val['h']),jnp.asarray(val['eta'])))
        loss=float(np.mean(np.logaddexp(0,logits)-val['y']*logits))
        trajectory.append({'fraction':frac,'size':size,'seed':seed,'step':it,'val_pair_equal_nll':loss})
        if not frozen:
            if loss<best[0]-1e-6:best=(loss,jax.tree_util.tree_map(np.asarray,par),it);stale=0
            else:stale+=1
            if stale>=20:frozen=True
    return model,best,trajectory,len(six),len(wix)

def main():
    OUT.mkdir(exist_ok=True)
    sr,wr,_,vr,_=lib.prepare()
    struct,wide,val=map(rank.normalize,(sr,wr,vr))
    configs=[(frac,'current',seed) for frac in (.25,.5,1.) for seed in (17,23,41)]
    configs += [(1.,size,seed) for size in ('small','larger') for seed in (17,23,41)]
    meta=[];trajectories=[];frozen=[]
    old={}
    if (OUT/'retraining_diagnostic.csv').exists():
        for row in csv.DictReader((OUT/'retraining_diagnostic.csv').open()):
            old[(float(row['fraction']),row['size'],int(row['seed']))]=row
    for cfg in configs:
        frac,size,seed=cfg
        if cfg in old and Path(old[cfg]['checkpoint']).exists():
            row={k:(int(v) if k in ('seed','structured_pairs','wide_pairs','val_selected_step') else
                     float(v) if k in ('fraction','val_nll') else v)
                 for k,v in old[cfg].items() if k not in ('hard_selected_B15','hard_mean_selected_Q16','hard_bad13_rescued','hard_all_proposal_MAE')}
            model=Critic(size=size)
            template=model.init(jax.random.PRNGKey(seed),jnp.zeros((1,214)),jnp.zeros((1,3)))
            selected=serialization.from_bytes(template,Path(row['checkpoint']).read_bytes())
        else:
            model,best,traj,ns,nw=train(cfg,struct,wide,val)
            dest=OUT/'controlled_models'/f'data{frac:g}_{size}_seed{seed}'
            dest.mkdir(parents=True,exist_ok=True)
            (dest/'checkpoint.msgpack').write_bytes(serialization.to_bytes(best[1]))
            row={'fraction':frac,'size':size,'seed':seed,'structured_pairs':ns,'wide_pairs':nw,
                 'val_selected_step':best[2],'val_nll':best[0], 'checkpoint':str(dest/'checkpoint.msgpack')}
            trajectories+=traj;selected=best[1]
        meta.append(row);frozen.append((row,model,selected))
    # Test outcomes are opened only after every configuration has been fixed by VAL.
    rr=list(csv.DictReader((U/'frozen_k16_proposal_predictions.csv').open()))
    rr.sort(key=lambda r:(int(r['episode_index']),int(r['proposal_index'])))
    q=np.asarray([float(r['Q16_lower']) for r in rr]).reshape(200,16)
    ep=np.asarray([[float(r[f'eta{i}']) for i in (1,2,3)] for r in rr],np.float64)
    h=np.load(K/'cohort_features.npz')['h_raw'].astype(np.float32)
    man=json.loads((D/'orthoflow3_continuous_basin_critic_v1/dataset_manifest.json').read_text())
    sn=man['state_normalization']['Toy'];en=man['eta_normalization']
    hh=np.repeat((h-np.asarray(sn['mean'],np.float32))/np.asarray(sn['std'],np.float32),16,axis=0)
    ez=(ep-np.asarray(en['center'],np.float32))/np.asarray(en['scale'],np.float32)
    bad=[70,73,77,95,102,103,106,115,151,152,162,176,177]
    for row,model,par in frozen:
        logits=[]
        for start in range(0,len(ez),512):
            logits.extend(np.asarray(model.apply(par,jnp.asarray(hh[start:start+512]),jnp.asarray(ez[start:start+512]))).tolist())
        p=expit(np.asarray(logits)).reshape(200,16)
        picked=np.argmax(p,axis=1)
        row.update({'hard_selected_B15':int((q[np.arange(200),picked]>=15/16).sum()),
                    'hard_mean_selected_Q16':float(q[np.arange(200),picked].mean()),
                    'hard_bad13_rescued':int((q[bad,picked[bad]]>=15/16).sum()),
                    'hard_all_proposal_MAE':float(np.mean(abs(p-q)))})
    fields=list(meta[0]);
    with (OUT/'retraining_diagnostic.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(meta)
    if trajectories:
        with (OUT/'retraining_trajectories.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(trajectories[0]));w.writeheader();w.writerows(trajectories)
    (OUT/'retraining_protocol.json').write_text(json.dumps({'new_rollout':0,'selection':'VAL pair-equal NLL only','configs':configs,
      'train_steps':4000,'batch':'64 structured + 64 wide','other_hyperparameters':'same as frozen W1','TEST_opened_after_all_checkpoints_frozen':True},indent=2)+'\n')

if __name__=='__main__':main()
