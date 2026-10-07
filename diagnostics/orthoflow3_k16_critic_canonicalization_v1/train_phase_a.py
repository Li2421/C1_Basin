"""Canonical BCE critic, v2 plus frozen-generator proposal-aligned evidence."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import json

import flax.serialization as serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow.parquet as pq
from scipy.stats import spearmanr

from diagnostics.orthoflow3_k16_critic_canonicalization_v1 import phase_a as a
from diagnostics.orthoflow3_generator_critic_v1 import train_evaluate as base


def state_rows():
    return a.states()+a.load(a.OUT/'initial_expansion/states.json')


def all_evidence():
    paths=[a.OUT/'phase_a/proposal_Q_evidence.json',a.OUT/'initial_expansion/phase_a/proposal_Q_evidence.json']
    bundles=[a.load(p) for p in paths]
    if not all(b['complete'] for b in bundles):raise RuntimeError('Aligned proposal evidence incomplete')
    return [r for b in bundles for r in b['rows']]


def all_proposals():
    return a.load(a.OUT/'phase_a/proposals.json')['states']+a.load(a.OUT/'initial_expansion/phase_a/proposals.json')['states']


def prepare():
    norm=a.load(a.frozen.NORMALIZATION)
    prepared={}
    for sc in a.SCENARIOS:
        ss=[r for r in state_rows() if r['scenario']==sc]
        hc=[a.inputs(r,norm) for r in ss]
        prepared[sc]={'states':ss,'index':{r['state_uid']:i for i,r in enumerate(ss)},
                      'h':np.asarray([r[0] for r in hc]),'c':np.asarray([r[1] for r in hc]),
                      'labels':defaultdict(list),'aligned':defaultdict(list),'generic':defaultdict(list),
                      'state_exposure':Counter(),'source_exposure':Counter()}
    evidence=all_evidence()
    if any(r['collisions'] for r in evidence):
        raise RuntimeError('Valid collision evidence requires a safety audit before learning')
    ignored=Counter()
    for row in pq.read_table(a.DATA/'eta_labels.parquet').to_pylist():
        # Do not treat early-stopped fractions as unbiased continuous Q targets.
        if row['seed_count']<16 or row['numerical_failure_count']:
            ignored['generic_partial_or_numerical']+=1;continue
        item={**row,'eta':np.asarray(a.decode(row['eta_raw']),np.float32),
              'q':row['success_count']/row['seed_count'],'weight':min(row['seed_count'],16),'source':'v2'}
        prepared[row['scenario']]['generic'][row['state_uid']].append(item)
    for row in evidence:
        if row['proposal_index'] is None:continue
        if row['Q16'] is None:
            ignored['aligned_numerical_uncertified']+=1;continue
        item={**row,'eta':np.asarray(row['eta'],np.float32),'q':row['Q16'],'weight':16,
              'robust_15of16':row['robust'],'source':'frozen_generator_proposal_Q16'}
        d=prepared[row['scenario']];d['aligned'][row['state_uid']].append(item)
        d['labels'][row['state_uid']].append(item)
    for sc,d in prepared.items():
        for r in d['states']:
            assert d['aligned'][r['state_uid']], (sc,r['state_uid'],'no certified Q')
    return prepared,dict(ignored)


def batch(data,sc,split,rng,n=96):
    d=data[sc];ids=base.split_state_ids(data,sc,split);chosen=rng.choice(ids,n)
    h=[];c=[];e=[];q=[];w=[]
    for sid in chosen:
        pool=d['aligned'][sid] if rng.random()<.8 else d['generic'][sid]
        if not pool:pool=d['aligned'][sid]
        row=pool[int(rng.integers(len(pool)))];i=d['index'][sid]
        d['state_exposure'][sid]+=1;d['source_exposure'][row['source']]+=1
        h.append(d['h'][i]);c.append(d['c'][i]);e.append((row['eta']-base.CENTER)/base.RADIUS);q.append(row['q']);w.append(row['weight'])
    return tuple(jnp.asarray(x,jnp.float32) for x in (h,c,e,q,w))


def metrics(model,params,data):
    proposals=all_proposals()
    evidence=all_evidence()
    by=defaultdict(dict)
    for e in evidence:by[e['state_uid']][e['proposal_index']]=e
    result=[]
    original_ids={x['state_uid'] for x in a.states()}
    for row in proposals:
        if row['split']!='validation':continue
        sc=row['scenario'];sid=row['state_uid'];d=data[sc];i=d['index'][sid]
        logits=np.asarray(model.apply(params,jnp.asarray(d['h'][i:i+1].repeat(17,0)),jnp.asarray(d['c'][i:i+1].repeat(17,0)),
                       jnp.asarray((np.asarray(row['etas'])-base.CENTER)/base.RADIUS),method=getattr(model,base.METHOD[sc])))
        scores=1/(1+np.exp(-logits));pick=int(np.argmax(logits));ev=by[sid];selected=ev[pick]
        exact=all(ev[j]['Q16'] is not None for j in range(17))
        lo=max(ev[j]['Q_lower'] for j in range(17));hi=max(ev[j]['Q_upper'] for j in range(17))
        robust_any=(True if any(ev[j]['robust'] is True for j in range(17)) else
                    None if any(ev[j]['robust'] is None for j in range(17)) else False)
        pairs=correct=0
        for j in range(17):
            for k in range(j):
                if ev[j]['Q16'] is None or ev[k]['Q16'] is None:continue
                diff=ev[j]['Q16']-ev[k]['Q16']
                if abs(diff)<1/16:continue
                pairs+=1;correct+=(scores[j]-scores[k])*diff>0
        b0=ev[None]['robust'];cur=selected['robust'];old=ev[row['old_index']]['robust']
        result.append({'scenario':sc,'state_uid':sid,'timestep':d['states'][i]['timestep'],
                       'population':'v2' if sid in original_ids else 'initial_expansion',
                       'oracle_robust':robust_any,'old_robust':old,'selected_robust':cur,
                       'selected_index':pick,'old_index':row['old_index'],'scores':scores.tolist(),'B0_robust':b0,
                       'rescue':b0 is False and cur is True,'break':b0 is True and cur is False,
                       'miss':robust_any is True and cur is False,'unresolved':cur is None,
                       'exploitation':bool(scores[pick]>=.9375 and selected['Q_upper']<.5),
                       'old_exploitation':bool(row['old_scores'][row['old_index']]>=.9375 and ev[row['old_index']]['Q_upper']<.5),
                       'regret_lower':max(0.,lo-selected['Q_upper']),'regret_upper':hi-selected['Q_lower'],
                       'oracle_q_exact':lo if exact else None,'selected_Q16':selected['Q16'],
                       'top1_Q_tie_aware':selected['Q16']==lo if exact else None,
                       'pairwise_correct':int(correct),'pairwise_pairs':pairs})
    summary={}
    for sc in a.SCENARIOS:
        rr=[r for r in result if r['scenario']==sc]
        summary[sc]={'states':len(rr),**{k:sum(r[k] is True for r in rr) for k in ['oracle_robust','old_robust','selected_robust','rescue','break','miss','unresolved','exploitation','old_exploitation','top1_Q_tie_aware']},
                     'old_misses':sum(r['oracle_robust'] is True and r['old_robust'] is False for r in rr),
                     'old_unresolved':sum(r['old_robust'] is None for r in rr),
                     'oracle_unresolved':sum(r['oracle_robust'] is None for r in rr),
                     'B0_robust':sum(r['B0_robust'] is True for r in rr),
                     'B0_nonrobust':sum(r['B0_robust'] is False for r in rr),
                     'B0_unresolved':sum(r['B0_robust'] is None for r in rr),
                     'old_rescue':sum(r['B0_robust'] is False and r['old_robust'] is True for r in rr),
                     'old_break':sum(r['B0_robust'] is True and r['old_robust'] is False for r in rr),
                     'mean_regret_lower':float(np.mean([r['regret_lower'] for r in rr])),
                     'mean_regret_upper':float(np.mean([r['regret_upper'] for r in rr])),
                     'pairwise_accuracy':sum(r['pairwise_correct'] for r in rr)/max(1,sum(r['pairwise_pairs'] for r in rr))}
    return {'summary':summary,'states':result}


def train(seed):
    data,ignored=prepare();model=base.Critic();dims={s:data[s]['h'].shape[1] for s in a.SCENARIOS};cdim=data[a.SCENARIOS[0]]['c'].shape[1]
    config={**a.load(a.OUT/'protocol.json')['phase_a'],
            'normalization_sha256':a.sha(a.frozen.NORMALIZATION),'generator_sha256':a.sha(a.frozen.GENERATOR_CKPT),
            'v2_states_sha256':a.sha(a.DATA/'states.parquet'),'v2_eta_labels_sha256':a.sha(a.DATA/'eta_labels.parquet'),
            'aligned_evidence_hashes':[a.sha(a.OUT/p) for p in ['phase_a/proposal_Q_evidence.json','initial_expansion/phase_a/proposal_Q_evidence.json']],
            'generic_critic_targets':'only full>=16valid-seed v2 empirical Q; no early-rejected fraction as exactQ',
            'numerical_targets':'exclude uncertified Q tuples, retain separately in evidence',
            'frozen_v1_labels_consumed':0,'training_state_ids':{s:base.split_state_ids(data,s,'train') for s in a.SCENARIOS},
            'validation_state_ids':{s:base.split_state_ids(data,s,'validation') for s in a.SCENARIOS}}
    a.dump('phase_a/critic/config.json',config)
    params=base.merge_initialized(model,dims,cdim,critic=True)
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(1e-3,weight_decay=1e-4));st=opt.init(params)
    @jax.jit
    def step(p,st,*args):
        def objective(pp):
            losses=[]
            for k,sc in enumerate(a.SCENARIOS):
                h,c,e,q,w=args[5*k:5*k+5]
                logits=model.apply(pp,h,c,e,method=getattr(model,base.METHOD[sc]))
                losses.append(jnp.sum(w*optax.sigmoid_binary_cross_entropy(logits,q))/jnp.sum(w))
            return jnp.mean(jnp.stack(losses))
        loss,grad=jax.value_and_grad(objective)(p);up,st=opt.update(grad,st,p)
        return optax.apply_updates(p,up),st,loss
    rngs={s:np.random.default_rng(base.stable_int('critic',seed,s)) for s in a.SCENARIOS}
    history=[];best=(float('inf'),None,0);stale=0
    for iteration in range(1,4001):
        args=[]
        for sc in a.SCENARIOS:args.extend(batch(data,sc,'train',rngs[sc]))
        params,st,loss=step(params,st,*args)
        if iteration%100:continue
        val=base.critic_val_loss(model,params,data)
        history.append({'step':iteration,'train':float(loss),'validation_proposal_BCE':val})
        print(json.dumps({'seed':seed,**history[-1]}),flush=True)
        if val<best[0]-1e-5:best=(val,serialization.to_bytes(params),iteration);stale=0
        else:stale+=1
        if stale>=10 and iteration>=1500:break
    path=a.OUT/f'phase_a/critic/seed{seed}/checkpoint.msgpack';path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(best[1])
    a.dump(f'phase_a/critic/seed{seed}/history.json',history)
    params=serialization.from_bytes(params,best[1]);m=metrics(model,params,data)
    a.dump(f'phase_a/critic/seed{seed}/metrics.json',m)
    result={'seed':seed,'best_step':best[2],'steps':iteration,'validation_loss':best[0],'checkpoint':str(path),'sha256':a.sha(path),
            'sampling_exposure_per_scenario':96*iteration,'ignored_targets':ignored,
            'state_exposure':{sc:dict(data[sc]['state_exposure']) for sc in a.SCENARIOS},
            'source_exposure':{sc:dict(data[sc]['source_exposure']) for sc in a.SCENARIOS}}
    a.dump(f'phase_a/critic/seed{seed}/training.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True,choices=base.SEEDS);args=p.parse_args()
    print(json.dumps(train(args.seed),indent=2))
