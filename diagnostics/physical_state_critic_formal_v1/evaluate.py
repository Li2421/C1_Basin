"""State-level paired B15/mean-Q statistics and state-conditioning diagnostics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

HERE=Path(__file__).resolve().parent
PANELS={
 'within_toy_TT':('matched','toy_give_way','TT'),
 'within_toy_FF':('matched','toy_give_way','FF'),
 'within_ring_TT':('matched','ring_exchange','TT'),
 'within_ring_FF':('matched','ring_exchange','FF'),
 'within_db_FF':('loso','double_bottleneck',None),
 'within_four_FF':('loso','four_way_intersection',None),
 'loso_heldout_toy':('loso','toy_give_way',None),
 'loso_heldout_db':('loso','double_bottleneck',None),
 'loso_heldout_four':('loso','four_way_intersection',None),
 'loso_heldout_ring':('loso','ring_exchange',None),
}
RNG_SEED=20261008


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def bootstrap(diff,seed=RNG_SEED):
    diff=np.asarray(diff,float)
    n=len(diff)
    if n==0:return dict(n=0,mean=None,ci95=None)
    rng=np.random.default_rng(seed)
    estimate=[]
    for _ in range(10000):
        estimate.append(float(np.mean(diff[rng.integers(n,size=n)])))
    return dict(n=n,mean=float(np.mean(diff)),ci95=np.quantile(estimate,[.025,.975]).tolist(),
                positive_states=int(np.sum(diff>0)),negative_states=int(np.sum(diff<0)),
                tied_states=int(np.sum(diff==0)))


def entropy(indices,levels=16):
    c=np.bincount(indices,minlength=levels)
    p=c[c>0]/len(indices)
    return float(-np.sum(p*np.log2(p)))


def ranking(logits,lower,upper):
    a,b=np.triu_indices(16,1)
    true=np.where(lower[:,a]-upper[:,b]>=.25,1,
                  np.where(lower[:,b]-upper[:,a]>=.25,-1,0))
    pred=np.sign(logits[:,a]-logits[:,b])
    valid=true!=0
    correct=(pred==true).astype(float)+.5*(pred==0)
    per=np.divide(np.sum(correct*valid,axis=1),np.sum(valid,axis=1),
                  out=np.full(len(true),np.nan),where=np.sum(valid,axis=1)>0)
    return dict(accuracy=float(np.mean(correct[valid])) if valid.any() else None,
                decisive_cases=int(valid.sum()),
                per_state=[None if not np.isfinite(v) else float(v) for v in per])


def state_effect(logits,lower,numerical):
    # Exact historical definition is meaningful only for a shared fixed eta bank.
    pred=expit(logits)
    true=np.where(numerical==0,lower,np.nan)
    pc=pred-np.nanmean(pred,axis=0,keepdims=True)
    tc=true-np.nanmean(true,axis=0,keepdims=True)
    valid=np.isfinite(tc)
    if valid.sum()<=10 or np.std(pc[valid])==0 or np.std(tc[valid])==0:
        return None
    value=float(np.corrcoef(pc[valid],tc[valid])[0,1])
    return value if np.isfinite(value) else None


def reversal(logits_eta,logits_state,lower,upper):
    n=lower.shape[0]
    hits_eta=np.zeros(n,float);hits_state=np.zeros(n,float);total=np.zeros(n,int)
    all_pairs=[]
    for a in range(16):
        for b in range(a+1,16):
            pos=np.flatnonzero(lower[:,a]-upper[:,b]>=.25)
            neg=np.flatnonzero(lower[:,b]-upper[:,a]>=.25)
            if len(pos)==0 or len(neg)==0:continue
            all_pairs.append(dict(a=a,b=b,A_states=len(pos),B_states=len(neg)))
            for i in pos:
                total[i]+=1
                hits_eta[i]+=int(logits_eta[i,a]>logits_eta[i,b])+.5*int(logits_eta[i,a]==logits_eta[i,b])
                hits_state[i]+=int(logits_state[i,a]>logits_state[i,b])+.5*int(logits_state[i,a]==logits_state[i,b])
            for i in neg:
                total[i]+=1
                hits_eta[i]+=int(logits_eta[i,b]>logits_eta[i,a])+.5*int(logits_eta[i,a]==logits_eta[i,b])
                hits_state[i]+=int(logits_state[i,b]>logits_state[i,a])+.5*int(logits_state[i,a]==logits_state[i,b])
    included=total>0
    if not included.any():return dict(strong_pairs=0,states=0,eta_only_accuracy=None,state_accuracy=None)
    # State is the resampling unit; pair multiplicity stays inside each state.
    eta_rate=np.divide(hits_eta[included],total[included])
    state_rate=np.divide(hits_state[included],total[included])
    return dict(strong_pairs=len(all_pairs),states=int(included.sum()),
                pair_details=all_pairs,
                eta_only_accuracy=bootstrap(eta_rate),state_accuracy=bootstrap(state_rate),
                state_minus_eta=bootstrap(state_rate-eta_rate))


def main(panel):
    mode,scene,chain=PANELS[panel]
    outcome=(HERE/'matched'/scene/chain/'test'/'outcomes.npz' if mode=='matched'
             else HERE/'loso'/scene/'test'/'outcomes.npz')
    prediction=(HERE/'ring_ff_test_predictions.npz' if panel=='within_ring_FF'
                else HERE/'fits'/panel/'test_predictions.npz')
    d=np.load(outcome);p=np.load(prediction)
    assert d['values'].shape==(128,17,16,5) and d['seen'].all(),f'incomplete test {panel}'
    assert list(d['state_uids'])==list(p['state_uids'])
    eta=d['eta'] if mode=='loso' else np.broadcast_to(d['eta'],(128,17,3))
    np.testing.assert_allclose(eta[:,:16],p['eta'],rtol=0,atol=1e-7)
    value=d['values'];success=np.sum((value[...,0]==1)&(value[...,4]==0),axis=2)
    numerical=np.sum(value[...,4]==1,axis=2)
    collisions=np.sum(value[...,3]==1,axis=2)
    lower=success/16;upper=(success+numerical)/16
    definite_success=success>=15
    definite_failure=success+numerical<15
    unresolved=~(definite_success|definite_failure)
    choices={'Safety':np.full(128,16,int),
             'Q_eta':np.asarray(p['eta_only_choice'],int),
             'Q_state':np.asarray(p['state_choice'],int),
             'Oracle':np.argmax(lower[:,:16],axis=1)}
    assert all(np.all((c>=0)&(c<17)) for c in choices.values())
    methods={}
    raw=[]
    for name,c in choices.items():
        ii=np.arange(128)
        qlo=lower[ii,c];qhi=upper[ii,c]
        b15=definite_success[ii,c]
        amb=unresolved[ii,c]
        col=collisions[ii,c]
        regret_lo=np.maximum(0,np.max(lower[:,:16],axis=1)-qhi)
        regret_hi=np.maximum(0,np.max(upper[:,:16],axis=1)-qlo)
        methods[name]=dict(states=128,B15_confirmed=int(b15.sum()),
                           B15_possible=int((b15|amb).sum()),B15_unresolved=int(amb.sum()),
                           mean_success_lower=float(qlo.mean()),mean_success_upper=float(qhi.mean()),
                           oracle_regret_lower=float(regret_lo.mean()),
                           oracle_regret_upper=float(regret_hi.mean()),
                           collisions=int(col.sum()),collision_rate=float(col.sum()/(128*16)),
                           selected_eta_index_counts=np.bincount(c,minlength=17).tolist(),
                           selected_index_entropy_bits=entropy(c,17))
        for i in range(128):
            raw.append(dict(panel=panel,method=name,state_uid=str(d['state_uids'][i]),
                            selected_index=int(c[i]),selected_eta=eta[i,c[i]].tolist(),
                            B15_confirmed=bool(b15[i]),B15_unresolved=bool(amb[i]),
                            success_count=int(success[i,c[i]]),numerical_count=int(numerical[i,c[i]]),
                            collision_count=int(col[i]),q_lower=float(qlo[i]),q_upper=float(qhi[i]),
                            oracle_regret_lower=float(regret_lo[i]),oracle_regret_upper=float(regret_hi[i])))
    comparisons={}
    for other in ('Q_eta','Safety'):
        a=choices['Q_state'];b=choices[other];ii=np.arange(128)
        comparisons[f'Q_state_minus_{other}']=dict(
            B15_confirmed=bootstrap(definite_success[ii,a].astype(float)-definite_success[ii,b].astype(float)),
            mean_success_lower=bootstrap(lower[ii,a]-lower[ii,b]),
            mean_success_upper=bootstrap(upper[ii,a]-upper[ii,b]),
            unresolved_pair_states=int(np.sum(unresolved[ii,a]|unresolved[ii,b])))
    ratios={}
    for base in ('Safety','Q_eta'):
        denom=methods['Oracle']['mean_success_lower']-methods[base]['mean_success_lower']
        numerator=methods['Q_state']['mean_success_lower']-methods[base]['mean_success_lower']
        ratios[f'from_{base}']=float(numerator/denom) if denom>0 else None
    qeta=np.asarray(p['eta_only_logits'],float);qstate=np.asarray(p['state_logits'],float)
    local_lower,local_upper=lower[:,:16],upper[:,:16]
    diagnostics=dict(local_pairwise_eta_only=ranking(qeta,local_lower,local_upper),
                     local_pairwise_state=ranking(qstate,local_lower,local_upper),
                     state_effect_correlation=(state_effect(qstate,local_lower,numerical[:,:16])
                                               if mode=='matched' else None),
                     strong_reversal=(reversal(qeta,qstate,local_lower,local_upper)
                                      if mode=='matched' else None),
                     selection_entropy_bits=methods['Q_state']['selected_index_entropy_bits'],
                     selection_index_interpretation=('same fixed eta across states' if mode=='matched' else
                                                      'proposal slot only: each state has different eta values'))
    # Per-state eta choices and difficult cases remain inspectable, even when aggregate wins.
    failures=[]
    for i in np.argsort(np.max(lower[:,:16],axis=1)-lower[np.arange(128),choices['Q_state']])[::-1][:12]:
        failures.append(dict(state_uid=str(d['state_uids'][i]),
                             state_choice=int(choices['Q_state'][i]),
                             eta_choice=int(choices['Q_eta'][i]),
                             oracle_choice=int(choices['Oracle'][i]),
                             q_state=float(lower[i,choices['Q_state'][i]]),
                             q_eta=float(lower[i,choices['Q_eta'][i]]),
                             q_oracle=float(np.max(lower[i,:16])),
                             collision_state=int(collisions[i,choices['Q_state'][i]]),
                             numerical_state=int(numerical[i,choices['Q_state'][i]])))
    result=dict(panel=panel,scene=scene,placement=chain or 'FF',
                generalization='within_scene' if panel.startswith('within_') else 'cross_scene_LOSO',
                test_states=128,test_outcome_sha256=sha(outcome),
                prediction_sha256=sha(prediction),methods=methods,
                paired_comparisons=comparisons,oracle_recovery=ratios,
                diagnostics=diagnostics,failure_cases=failures,
                numerical_unknown_total=int(numerical.sum()))
    dest=HERE/'evaluation'/panel
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (dest/'selected_state_rows.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(raw[0]));writer.writeheader()
        for r in raw:writer.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()})
    print(json.dumps(dict(panel=panel,methods={k:(v['B15_confirmed'],v['B15_unresolved'])
                                                for k,v in methods.items()},
                          state_minus_eta=comparisons['Q_state_minus_Q_eta']['B15_confirmed'])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',choices=tuple(PANELS),required=True)
    a=p.parse_args();main(a.panel)
