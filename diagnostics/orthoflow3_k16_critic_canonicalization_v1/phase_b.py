"""Derived coordinates and generator-first learning on fixed train/dev evidence."""
from __future__ import annotations
import argparse,json,shutil,os
_requested_backend=os.environ.get('JAX_PLATFORMS','cpu')
_requested_devices=os.environ.get('CUDA_VISIBLE_DEVICES','')
from collections import defaultdict
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import jax,jax.numpy as jnp,optax
from flax import serialization
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import train_phase_a as ta
from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import canonical_conditioning as cc
# An archived rollout module imported by phase_a forces CPU at import time.
# Respect the scheduler's explicit training device before JAX initializes.
os.environ['JAX_PLATFORMS']=_requested_backend
os.environ['CUDA_VISIBLE_DEVICES']=_requested_devices
base=a.learn
OUT=a.OUT/'phase_b'
DATA=a.ROOT/'datasets/orthoflow3_basin_dataset_v2_canonical'


def dump(name,x):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')


def transformed(row):
    cond=a.decode(row['conditioning']);phy=a.decode(row['structured_state']);env=a.decode(row['environment_descriptor'])
    h,meta=cc.encode(row['scenario'],cond['flat'],phy,env)
    return {**row,'conditioning':json.dumps({**cond,'flat':h.tolist(),'schema':cc.VERSION,
                    'source_conditioning_hash':a.digest(cond),'canonical_transform':meta}),
            'source_state_uid':row['state_uid'],'conditioning_version':cc.VERSION}


def build():
    if (DATA/'manifest.json').exists():return a.load(DATA/'manifest.json')
    rows=[transformed(r) for r in a.states()]
    extra=[transformed(r) for r in a.load(a.OUT/'initial_expansion/states.json')]
    DATA.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows),DATA/'states.parquet')
    shutil.copyfile(a.DATA/'eta_labels.parquet',DATA/'eta_labels.parquet')
    allrows=rows+extra
    keys=sorted({k for r in allrows if r['split']=='train' for k in base.scalar_environment_fields(a.decode(r['environment_descriptor']))})
    env=np.asarray([[base.scalar_environment_fields(a.decode(r['environment_descriptor'])).get(k,0.) for k in keys] for r in allrows if r['split']=='train'])
    cm,cs=env.mean(0),env.std(0);cs[cs<1e-6]=1.
    norm={'environment_keys':keys,'context_mean':cm.tolist(),'context_std':cs.tolist(),
          'scenario_onehot':False,'fit':'train_only','scenarios':{}}
    for sc in a.SCENARIOS:
        hh=np.asarray([a.decode(r['conditioning'])['flat'] for r in allrows if r['scenario']==sc and r['split']=='train'])
        hm,hs=cc.shared_slot_stats(hh,sc)
        norm['scenarios'][sc]={'h_mean':hm.tolist(),'h_std':hs.tolist()}
    dump('normalization.json',norm);dump('initial_expansion_canonical.json',extra)
    manifest={'version':cc.VERSION,'source':str(a.DATA),'state_count':len(rows),'source_states_sha256':a.sha(a.DATA/'states.parquet'),
        'labels_sha256':a.sha(DATA/'eta_labels.parquet'),'source_labels_sha256':a.sha(a.DATA/'eta_labels.parquet'),
        'state_ids_unchanged':True,'labels_byte_identical':True,'new_rollouts_for_reencoding':0,
        'state_file_sha256':a.sha(DATA/'states.parquet'),'normalization':str(OUT/'normalization.json'),
        'normalization_sha256':a.sha(OUT/'normalization.json'),'encoder_sha256':a.sha(Path(cc.__file__)),
        'additional_critic_states':len(extra),'accepted_phase_A':'PHASE_A_CRITIC_PARTIAL',
        'phase_A_numerical_uncertainty_preserved':True}
    (DATA/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    dump('protocol.json',{'generator_architecture_loss_budget':'unchanged train_evaluate.train_generator',
        'critic_architecture_loss_budget':'unchanged phase-A empirical-Q BCE,80% aligned/20% generic',
        'critic_evidence':'same v2 and Phase-A train/dev Q evidence; no confirmation labels',
        'proposal_convention':'mean+16 stochastic; finite ranking; no mean-only selector',
        'dev_evaluation_states':'12 validation states/scenario,lowest SHA256(phase-b-dev|state_uid)',
        'active_rotation_limitation':'frozen Four MACFlow is not rotation-equivariant; transform actual Flow suffix,do not impose invariant output',
        'context':'physical scalar descriptors with global train-only stats; native dimension adapters retained',
        'normalization':'shared feature stats across equivalent agent slots; Double unchanged native layout'})
    return manifest


def rows():return pq.read_table(DATA/'states.parquet').to_pylist()


def inputs(row,norm=None):
    if norm is None:norm=a.load(OUT/'normalization.json')
    n=norm['scenarios'][row['scenario']];h=np.asarray(a.decode(row['conditioning'])['flat'],np.float32)
    h=(h-np.asarray(n['h_mean'],np.float32))/np.asarray(n['h_std'],np.float32)
    e=base.scalar_environment_fields(a.decode(row['environment_descriptor']))
    c=(np.asarray([e.get(k,0.) for k in norm['environment_keys']],np.float32)-np.asarray(norm['context_mean'],np.float32))/np.asarray(norm['context_std'],np.float32)
    return h,c


def prepare_generator():
    data={};norm=a.load(OUT/'normalization.json')
    for sc in a.SCENARIOS:
        ss=[r for r in rows() if r['scenario']==sc];hc=[inputs(r,norm) for r in ss]
        data[sc]={'states':ss,'index':{r['state_uid']:i for i,r in enumerate(ss)},'h':np.asarray([x[0] for x in hc]),
                  'c':np.asarray([x[1] for x in hc]),'labels':defaultdict(list),'positive':defaultdict(list),'negative':defaultdict(list)}
    for r in pq.read_table(DATA/'eta_labels.parquet').to_pylist():
        e=np.asarray(a.decode(r['eta_raw']),np.float32)
        if not (np.all(e>=base.LOW-1e-8) and np.all(e<=base.HIGH+1e-8)):continue
        if r['robust_15of16'] is None:continue
        item={**r,'eta':e}
        data[r['scenario']]['labels'][r['state_uid']].append(item)
        data[r['scenario']]['positive' if r['robust_15of16'] else 'negative'][r['state_uid']].append(item)
    return data


def train_generator():
    base.OUT=OUT;base.K=16
    _,_,selected=base.train_generator(prepare_generator())
    dump('generator_frozen.json',{'selected':selected,'frozen_before_critic_training':True,'conditioning_sha256':a.sha(DATA/'states.parquet')})
    return selected


def prepare_critic():
    data,ignored=ta.prepare()
    by={r['state_uid']:r for r in rows()+a.load(OUT/'initial_expansion_canonical.json')}
    for sc,d in data.items():
        ss=[by[r['state_uid']] for r in d['states']];hc=[inputs(r) for r in ss]
        d.update(states=ss,h=np.asarray([x[0] for x in hc]),c=np.asarray([x[1] for x in hc]))
    return data,ignored


def train_critic(seed):
    assert (OUT/'generator_frozen.json').exists()
    data,ignored=prepare_critic();model=base.Critic();dims={s:data[s]['h'].shape[1] for s in a.SCENARIOS};cdim=data[a.SCENARIOS[0]]['c'].shape[1]
    params=base.merge_initialized(model,dims,cdim,critic=True)
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(1e-3,weight_decay=1e-4));st=opt.init(params)
    @jax.jit
    def step(p,st,*args):
        def loss(pp):
            losses=[]
            for k,sc in enumerate(a.SCENARIOS):
                h,c,e,q,w=args[5*k:5*k+5];z=model.apply(pp,h,c,e,method=getattr(model,base.METHOD[sc]))
                losses.append(jnp.sum(w*optax.sigmoid_binary_cross_entropy(z,q))/jnp.sum(w))
            return jnp.mean(jnp.stack(losses))
        value,g=jax.value_and_grad(loss)(p);up,st=opt.update(g,st,p)
        return optax.apply_updates(p,up),st,value
    rng={s:np.random.default_rng(base.stable_int('critic',seed,s)) for s in a.SCENARIOS}
    best=(float('inf'),None,0);stale=0;history=[]
    for it in range(1,4001):
        args=[]
        for sc in a.SCENARIOS:args.extend(ta.batch(data,sc,'train',rng[sc]))
        params,st,loss=step(params,st,*args)
        if it%100:continue
        val=base.critic_val_loss(model,params,data);history.append({'step':it,'train':float(loss),'validation':val})
        print(json.dumps({'seed':seed,**history[-1]}),flush=True)
        if val<best[0]-1e-5:best=(val,serialization.to_bytes(params),it);stale=0
        else:stale+=1
        if stale>=10 and it>=1500:break
    path=OUT/f'critic/seed{seed}/checkpoint.msgpack';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(best[1])
    dump(f'critic/seed{seed}/history.json',history)
    selected=serialization.from_bytes(params,best[1]);dump(f'critic/seed{seed}/metrics.json',ta.metrics(model,selected,data))
    result={'seed':seed,'best_step':best[2],'steps':it,'validation_loss':best[0],'checkpoint':str(path),'sha256':a.sha(path),
        'ignored_targets':ignored,'exposure_per_scenario':96*it,'normalization_sha256':a.sha(OUT/'normalization.json')}
    dump(f'critic/seed{seed}/training.json',result);return result


def freeze_critic():
    rr=[a.load(OUT/f'critic/seed{s}/training.json') for s in base.SEEDS]
    selected=min(rr,key=lambda r:(r['validation_loss'],r['seed']))
    dump('critic_frozen.json',{'selected':selected,'selected_on':'validation_BCE_only','no_confirmation_labels':True})
    return selected


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','generator','critic','freeze_critic']);p.add_argument('--seed',type=int,default=17);x=p.parse_args()
    print(json.dumps({'build':build,'generator':train_generator,'critic':lambda:train_critic(x.seed),'freeze_critic':freeze_critic}[x.stage](),indent=2))
