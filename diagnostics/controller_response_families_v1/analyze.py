"""Independent-seed estimates of Q information erased by nominal fingerprints."""
import itertools,json
from pathlib import Path
import numpy as np
from shared_rollout_db.src.rollout_db import connect
from .study import HERE,read,write
from .runtime import FAMILIES

def variants(success,unknown):
    ix=np.flatnonzero(unknown.ravel());base=success.ravel().astype(float)
    out=np.broadcast_to(base,(2**len(ix),len(base))).copy()
    for i,bits in enumerate(itertools.product((0.,1.),repeat=len(ix))):out[i,ix]=bits
    return out.reshape((-1,)+success.shape)

def project(a,kind):
    x=a-a.mean(-2,keepdims=True)
    if kind=='controller_eta':x=x-x.mean(-1,keepdims=True)
    return x

def run():
    p=read(HERE/'protocol.json');cases=read(HERE/'cases.json');records=[]
    with connect(True) as db:
        for r in cases:
            x=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone();assert x is not None
            records.append(dict(r,**{k:x[k] for k in ['success','numerical_failure','collision','deadlock','timeout']}))
    write(HERE/'all_seed_results.json',records)
    summary=[];boots={};arrays={}
    for scene in sorted(p['controllers']):
        ids=sorted(s['state_uid'] for s in p['states'] if s['scenario']==scene);n=len(ids)
        a=np.zeros((n,5,4,2),float);u=np.zeros_like(a,dtype=bool);collisions=0
        for r in records:
            if r['scene']!=scene or r['controller_name']=='flow':continue
            idx=(ids.index(r['state_uid']),FAMILIES.index(r['controller_name']),p['eta_indices'].index(r['eta_index']),r['seed'])
            a[idx]=r['success'];u[idx]=r['numerical_failure'];collisions+=r['collision']
        boot=np.random.default_rng(2026101103).integers(n,size=(20000,n))
        energies={}
        for kind in ('controller','controller_eta'):
            half=[project(a[...,k],kind) for k in range(2)]
            estimate=np.mean(half[0]*half[1],axis=(1,2));lo=[];hi=[]
            for i in range(n):
                aa=project(variants(a[i,...,0],u[i,...,0]),kind).reshape(-1,20)
                bb=project(variants(a[i,...,1],u[i,...,1]),kind).reshape(-1,20)
                product=aa@bb.T/20
                lo.append(float(product.min()));hi.append(float(product.max()))
            lo=np.asarray(lo);hi=np.asarray(hi);dist=estimate[boot].mean(1)
            lows=lo[boot].mean(1);highs=hi[boot].mean(1)
            energies[kind]=dict(verified_success_cross_half_energy=float(estimate.mean()),
                CI95=np.quantile(dist,[.025,.975]).tolist(),
                censor_lower=float(lo.mean()),censor_upper=float(hi.mean()),
                censor_robust_CI95=[float(np.quantile(lows,.025)),float(np.quantile(highs,.975))])
            boots[scene+'_'+kind]=dist;boots[scene+'_'+kind+'_lower']=lows;boots[scene+'_'+kind+'_upper']=highs
        arrays[scene+'_success']=a;arrays[scene+'_unknown']=u;arrays[scene+'_state_uid']=np.asarray(ids)
        familyrows=[]
        for ci,c in enumerate(FAMILIES):
            familyrows.append(dict(family=c,success=int(a[:,ci].sum()),attempts=int(a[:,ci].size),unknown=int(u[:,ci].sum()),
                per_eta_verified_success=a[:,ci].mean((0,2)).tolist()))
        summary.append(dict(scene=scene,physical_states=n,collision_count=collisions,families=familyrows,erased_information=energies))
    np.savez_compressed(HERE/'root_alias_bootstrap.npz',**boots)
    np.savez_compressed(HERE/'root_alias_labels.npz',**arrays)
    write(HERE/'root_alias_result.json',dict(summary=summary,new_rollouts=640,exact_reused_rollouts=128,
        current_input='physical state + nominal14 + contact18 + eta; identical across the five endpoint-matched controllers at fixed state and eta',
        estimand='Average conditional controller variance in true success probability. Independent two-seed cross-products remove Bernoulli variance bias. It is the minimum squared probability error of a controller-blind predictor under uniform five-controller weighting.',
        eta_interaction='Double-center across controller and eta before independent-half product. Positive value rules out only a controller-specific success offset.',
        bounds='Exact enumeration of all binary completions of numerical outcomes per physical-state panel; state bootstrap confidence intervals are conditional on the five frozen controller families and four frozen etas.',
        limitation='Mechanism falsification, not final generalization;Q2 does not certify B15, and8states give limited precision. Old same-scene training family need not exhibit the exact same alias.'))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':run()
