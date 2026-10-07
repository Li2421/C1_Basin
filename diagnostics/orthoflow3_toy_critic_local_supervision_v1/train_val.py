"""Same pretrained pure-NLL critic, with versus without cached local labels.

This script reads only TRAIN and VAL outcomes. It freezes every checkpoint before
the TEST rollout phase starts. Five parent families are deliberately shared:
this is a local learnability intervention, not a new-family generalization test.
"""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from scipy.special import expit
from shared_rollout_db.src.rollout_db import connect,lookup_exact,canonical
jax.config.update('jax_enable_x64',True)

PRIOR=ROOT/'diagnostics/orthoflow3_toy_critic_local_state_probe_v1'
OLD=ROOT/'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1'
NORM=ROOT/'diagnostics/orthoflow3_continuous_basin_critic_v1/dataset_manifest.json'
W1=ROOT/'diagnostics/orthoflow3_nll_weighting_ablation_v1'
spec=importlib.util.spec_from_file_location('frozen_w1_lib',W1/'run_experiment.py')
lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
model=lib.SingleCritic()
SEEDS=(17,23,41)
STEPS=(0,25,50,100,200,400,800)
EXTRA_BATCH=32

def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def csvrows(p):
    with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def csvwrite(p,rs):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def normalization():
    m=read(NORM);sn=m['state_normalization']['Toy'];en=m['eta_normalization']
    return (np.asarray(sn['mean'],np.float32),np.asarray(sn['std'],np.float32),
            np.asarray(en['center'],np.float32),np.asarray(en['scale'],np.float32))
def from_rows(rows):
    hm,hs,em,es=normalization()
    return {'h':(np.asarray([r['h_raw'] for r in rows],np.float32)-hm)/hs,
            'eta':(np.asarray([r['eta'] for r in rows],np.float32)-em)/es,
            'y':np.asarray([r['empirical_q'] for r in rows],np.float32),'rows':rows}
def db_rows(states,mapping):
    states={int(s['episode_index']):s for s in states}
    out=[];seeds=[canonical({'future_index':i}) for i in range(16)]
    with connect(True) as con:
        for r in mapping:
            state=states[int(r['episode_index'])]
            evidence=lookup_exact(con,*(r[k] for k in ('state_uid','eta_uid','controller_uid')),seeds)
            assert evidence['status']=='EXACT_REUSE' and len(evidence['records'])==16
            rr=evidence['records']
            assert all(x['compatibility_quality']=='EXACT_REUSE' and not x['numerical_failure'] and not x['conflict_quarantined'] for x in rr)
            out.append({'state_uid':r['state_uid'],'eta_uid':r['eta_uid'],'parent_episode_index':state['parent_episode_index'],
                'source_group':state['source_group'],'offset_m':state['physical_offset_m'],
                'kind':r['kind'],'h_raw':state['h_raw'],'eta':state['eta'][r['kind']],
                'n_trials':16,'success_count':int(sum(x['success'] for x in rr)),
                'empirical_q':sum(x['success'] for x in rr)/16})
    return out
def local_train_rows():
    frozen=read(PRIOR/'frozen_proposals.json')
    refmap=csvrows(PRIOR/'reference_cache_keys.csv')
    localmap=csvrows(PRIOR/'candidate_cache_keys.csv')
    refstates=frozen['references']
    h=np.load(OLD/'cohort_features.npz')['h_raw']
    refstates=[dict(s,h_raw=h[s['episode_index']].astype(float).tolist(),physical_offset_m=0.0,
                    parent_episode_index=s['episode_index']) for s in refstates]
    rs=db_rows(refstates,refmap)+db_rows(frozen['states'],localmap)
    assert len(rs)==30 and len({(r['state_uid'],r['eta_uid']) for r in rs})==30
    assert sorted({r['offset_m'] for r in rs})==[-.01,0.0,.01]
    return rs
def val_rows():
    assert read(OUT/'val/cache_postflight.json')['summary']['exact_reusable']==320
    frozen=read(OUT/'val/frozen_proposals.json')
    return db_rows(frozen['states'],csvrows(OUT/'val/candidate_cache_keys.csv'))
def metrics(logits,rows):
    q=np.asarray([r['empirical_q'] for r in rows]);p=expit(np.asarray(logits))
    nll=float(np.mean(np.logaddexp(0,np.asarray(logits))-q*np.asarray(logits)))
    ordered=[]
    for i in range(0,len(rows),2):
        pair=rows[i:i+2];assert len(pair)==2 and pair[0]['state_uid']==pair[1]['state_uid']
        sel=i+int(np.argmax(logits[i:i+2]));good=i+int(np.argmax(q[i:i+2]))
        ordered.append((q[sel]>=15/16,q[good]>=15/16,q[sel],q[good],int(sel==good)))
    return {'nll':nll,'mae':float(np.mean(np.abs(p-q))),
            'B15_selected':int(sum(x[0] for x in ordered)),
            'B15_oracle':int(sum(x[1] for x in ordered)),
            'selected_mean_Q16':float(np.mean([x[2] for x in ordered])),
            'oracle_mean_Q16':float(np.mean([x[3] for x in ordered])),
            'correct_two_eta_orderings':int(sum(x[4] for x in ordered))}
def predict(params,data):
    return np.asarray(model.apply(params,jnp.asarray(data['h']),jnp.asarray(data['eta'])),np.float64)
def train_variant(seed,variant,structured,wide,local,val):
    template=model.init(jax.random.PRNGKey(seed),jnp.zeros((1,214)),jnp.zeros((1,3)))
    cp=W1/'models/secondary_combined/W1'/f'seed{seed}'/'checkpoint.msgpack'
    params=serialization.from_bytes(template,cp.read_bytes())
    optimizer=optax.chain(optax.clip_by_global_norm(5.0),optax.adamw(1e-3,weight_decay=1e-4))
    optstate=optimizer.init(params)
    @jax.jit
    def step(p,s,h,e,y):
        def loss(x):return jnp.mean(optax.sigmoid_binary_cross_entropy(model.apply(x,h,e),y))
        value,grad=jax.value_and_grad(loss)(p)
        updates,s=optimizer.update(grad,s,p)
        return optax.apply_updates(p,updates),s,value
    rng=np.random.default_rng(seed)
    trajectories=[];best=None;maxstep=max(STEPS)
    for it in range(maxstep+1):
        if it in STEPS:
            val_logits=predict(params,val);valm=metrics(val_logits,val['rows'])
            local_logits=predict(params,local);localm=metrics(local_logits,local['rows'])
            row={'variant':variant,'seed':seed,'step':it,**{f'val_{k}':v for k,v in valm.items()},
                 **{f'local_train_{k}':v for k,v in localm.items()}}
            trajectories.append(row)
            if best is None or valm['nll']<best['val_nll']-1e-8:
                best={'step':it,'val_nll':valm['nll'],'params':jax.tree_util.tree_map(np.asarray,params),
                      'val_metrics':valm}
        if it==maxstep:break
        si=rng.integers(0,len(structured['y']),64)
        wi=rng.integers(0,len(wide['y']),64)
        if variant=='local':extra=local;ei=rng.integers(0,len(local['y']),EXTRA_BATCH)
        else:
            extra=structured
            ei=rng.integers(0,len(structured['y']),EXTRA_BATCH)
        bh=np.concatenate([structured['h'][si],wide['h'][wi],extra['h'][ei]])
        be=np.concatenate([structured['eta'][si],wide['eta'][wi],extra['eta'][ei]])
        by=np.concatenate([structured['y'][si],wide['y'][wi],extra['y'][ei]])
        params,optstate,_=step(params,optstate,jnp.asarray(bh),jnp.asarray(be),jnp.asarray(by))
    return best,trajectories,{'original_checkpoint':str(cp),'original_checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest()}
def main():
    if (OUT/'selected_checkpoints.json').exists():raise RuntimeError('Checkpoint selection already frozen')
    sr,wr,_,_,_=lib.prepare()
    structured,wide=map(lib.lib.normalize,(sr,wr))
    localrows=local_train_rows();valrows=val_rows()
    local=from_rows(localrows);val=from_rows(valrows)
    assert len(structured['y'])==1920 and len(wide['y'])==956
    assert len(localrows)==30 and len(valrows)==20
    assert not ({(r['state_uid'],r['eta_uid']) for r in localrows} & {(r['state_uid'],r['eta_uid']) for r in valrows})
    assert all(r['source_group'] in {z['source_group'] for z in localrows} for r in valrows)
    # No TEST file/database query exists anywhere in this training script.
    alltraj=[];selected=[]
    for variant in ('control','local'):
        for seed in SEEDS:
            best,traj,source=train_variant(seed,variant,structured,wide,local,val)
            dest=OUT/'models'/variant/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
            cp=dest/'checkpoint.msgpack';cp.write_bytes(serialization.to_bytes(best['params']))
            selected.append({'variant':variant,'seed':seed,'checkpoint':str(cp),
                'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),
                'val_selected_step':best['step'],'val_pair_equal_nll':best['val_nll'],
                'val_metrics':best['val_metrics'],**source})
            alltraj.extend(traj)
    csvwrite(OUT/'training_trajectory.csv',alltraj)
    write(OUT/'selected_checkpoints.json',{'selection_based_on':'VAL pair-equal soft-label NLL only',
          'seeds':list(SEEDS),'variants':['control','local'],'steps':list(STEPS),
          'original_structured_pairs':len(sr),'original_wide_pairs':len(wr),
          'local_train_pairs':len(localrows),'val_pairs':len(valrows),
          'TEST_outcomes_seen_before_checkpoint_freeze':False,
          'selected':selected})
    print(json.dumps({'selected':[(x['variant'],x['seed'],x['val_selected_step'],round(x['val_pair_equal_nll'],4)) for x in selected]}))
if __name__=='__main__':main()
