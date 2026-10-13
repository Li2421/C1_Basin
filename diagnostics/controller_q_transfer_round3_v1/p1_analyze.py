"""P1 matched-controller Q and independent-half selection audit."""
import json
import sys
from pathlib import Path

import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
from shared_rollout_db.src.rollout_db import connect


def read(name):return json.loads((HERE/name).read_text())
def write(name,v):(HERE/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def choose(q, prior=None, native_first=False):
    """Argmax with preregistered tie order; no outcome-dependent bank ordering."""
    q=np.asarray(q)
    if prior is None:prior=np.zeros(16)
    if native_first:
        order=sorted(range(16),key=lambda j:(j!=13,j))
    else:
        order=sorted(range(16),key=lambda j:(-float(prior[j]),j))
    rank=np.empty(16,int);rank[order]=np.arange(16)
    bump=(16-rank)*1e-10
    return np.argmax(q+bump,axis=-1)


def results(Y,U):
    """C,S,E,4 half-crossfit arrays. Unknown remains separate upper bound."""
    C,S,E,_=Y.shape
    fixed=np.zeros((C,2),int);state=np.zeros((C,2,S),int)
    fixed_success=np.zeros((C,2,S));fixed_unknown=np.zeros_like(fixed_success)
    state_success=np.zeros_like(fixed_success);state_unknown=np.zeros_like(fixed_success)
    global_regret=np.zeros((C,C,2,S));global_lower=np.zeros_like(global_regret)
    state_regret=np.zeros_like(global_regret);state_lower=np.zeros_like(global_regret)
    q=np.full((C,S,E),np.nan)
    for c in range(C):
        q[c]=np.divide(Y[c].sum(-1),np.maximum(4-U[c].sum(-1),1))
    for h,(discover,evaluate) in enumerate((((0,1),(2,3)),((2,3),(0,1)))):
        d=Y[:,:,:,discover].mean(-1)
        ev=Y[:,:,:,evaluate].mean(-1)
        un=U[:,:,:,evaluate].mean(-1)
        for c in range(C):
            prior=d[c].mean(0)
            fixed[c,h]=choose(prior,native_first=True)
            state[c,h]=choose(d[c],prior)
            fixed_success[c,h]=ev[c,:,fixed[c,h]]
            fixed_unknown[c,h]=un[c,:,fixed[c,h]]
            state_success[c,h]=ev[c,np.arange(S),state[c,h]]
            state_unknown[c,h]=un[c,np.arange(S),state[c,h]]
        for a in range(C):
            for b in range(C):
                source_fixed=ev[b,:,fixed[a,h]]
                source_fixed_unknown=un[b,:,fixed[a,h]]
                source_state=ev[b,np.arange(S),state[a,h]]
                source_state_unknown=un[b,np.arange(S),state[a,h]]
                global_regret[a,b,h]=fixed_success[b,h]-source_fixed
                global_lower[a,b,h]=fixed_success[b,h]-source_fixed-source_fixed_unknown
                state_regret[a,b,h]=state_success[b,h]-source_state
                state_lower[a,b,h]=state_success[b,h]-source_state-source_state_unknown
    return locals()


def centered(x,axes):
    z=x
    for ax in axes:z=z-z.mean(axis=ax,keepdims=True)
    return z


def main():
    assert read('P1_POSTFLIGHT.json')['missing']==0
    p=read('protocol.json');names=p['controller_order'];C=len(names);S=len(p['states']);Y=np.zeros((C,S,16,4),float);U=np.zeros_like(Y)
    cases=read('P1_CASES.json')
    with connect(True) as db:
        for r in cases:
            z=db.execute('SELECT success,numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(r['rollout_uid'],)).fetchone()
            assert z is not None and not z['conflict_quarantined']
            i=names.index(r['controller_name']);s=r['state_index'];j=r['eta_index'];k=r['seed']
            Y[i,s,j,k]=z['success'];U[i,s,j,k]=z['numerical_failure']
    np.savez_compressed(HERE/'P1_LABELS.npz',Y=Y,U=U,controller_names=np.array(names),state_uids=np.array([s['state_uid'] for s in p['states']]))
    m=results(Y,U)
    fs=m['fixed_success'].mean((1,2));su=m['state_success'].mean((1,2));
    eligible=np.flatnonzero((fs>=.4)&(fs<=.95))
    pairs=[(a,b) for a in eligible for b in eligible if a!=b]
    rng=np.random.default_rng(2026101206);draws=rng.integers(S,size=(5000,S))
    rows=[]
    for c,name in enumerate(names):
        diff=(m['state_success'][c]-m['fixed_success'][c]).mean(0)
        boot=diff[draws].mean(1)
        rows.append(dict(name=name,fixed_eta_halves=m['fixed'][c].tolist(),fixed_success=float(fs[c]),fixed_unknown=float(m['fixed_unknown'][c].mean()),
                         state_oracle_success=float(su[c]),state_oracle_gain=float(su[c]-fs[c]),
                         state_gain_ci95=np.quantile(boot,[.025,.975]).tolist(),
                         unknown_count=int(U[c].sum()),hindsight_best_fixed_eta=int(Y[c].mean((0,2)).argmax()),
                         hindsight_best_fixed_success=float(Y[c].mean((0,2)).max()),
                         viable_gate=bool(.4<=fs[c]<=.95)))
    cross=[]
    for a in range(C):
        for b in range(C):
            if a==b:continue
            cross.append(dict(source=names[a],target=names[b],eligible_pair=bool(a in eligible and b in eligible),
                              global_regret=float(m['global_regret'][a,b].mean()),global_conservative_lower=float(m['global_lower'][a,b].mean()),
                              state_regret=float(m['state_regret'][a,b].mean()),state_conservative_lower=float(m['state_lower'][a,b].mean())))
    gate={}
    for field in ('global','state'):
        if pairs:
            arr=np.array([m[field+'_regret'][a,b].mean(0) for a,b in pairs]);low=np.array([m[field+'_lower'][a,b].mean(0) for a,b in pairs])
            med=np.median(arr.mean(1));mlow=np.median(low.mean(1))
            boot=np.median(low[:,draws].mean(-1),axis=0)
            ci=np.quantile(boot,[.0125,.9875])
            gate[field]=dict(median_regret=float(med),conservative_unknown_median=float(mlow),simultaneous_ci95_conservative=ci.tolist(),qualified=bool(med>=.05 and ci[0]>0))
        else:gate[field]=dict(median_regret=None,conservative_unknown_median=None,simultaneous_ci95_conservative=None,qualified=False)
    # Exploratory provenance sensitivity: 12 of 14 have exact archived
    # training-manifest hashes; two use a frame-matched fallback manifest.
    exact={r['name'] for r in read('P1_MANIFEST_CROSSCHECK.json')['rows'] if r['exact_training_manifest']}
    exact_eligible=np.array([c for c in eligible if names[c] in exact],int)
    exact_pairs=[(a,b) for a in exact_eligible for b in exact_eligible if a!=b]
    provenance_sensitivity=dict(eligible_names=[names[c] for c in exact_eligible],eligible_count=len(exact_eligible),modes={})
    for field in ('global','state'):
        if exact_pairs:
            arr=np.array([m[field+'_regret'][a,b].mean(0) for a,b in exact_pairs])
            low=np.array([m[field+'_lower'][a,b].mean(0) for a,b in exact_pairs])
            boot=np.median(low[:,draws].mean(-1),axis=0)
            provenance_sensitivity['modes'][field]=dict(median=float(np.median(arr.mean(1))),
                simultaneous_ci95_conservative=np.quantile(boot,[.0125,.9875]).tolist())
    # Split-half cross-products estimate interaction energies without the positive noise bias of squared Q4 means.
    qhalf=np.divide(Y.reshape(C,S,16,2,2).sum(-1),np.maximum(2-U.reshape(C,S,16,2,2).sum(-1),1))
    qh0,qh1=qhalf[...,0],qhalf[...,1]
    ce0=centered(qh0.mean(1),(0,1));ce1=centered(qh1.mean(1),(0,1))
    sce0=centered(qh0,(0,1,2));sce1=centered(qh1,(0,1,2))
    energy=dict(controller_eta=float(np.mean(ce0*ce1)),state_controller_eta=float(np.mean(sce0*sce1)),
                estimator='independent half cross-product using observed successes / valid counts; all-unknown half-cells set to lower endpoint 0 for this auxiliary statistic only. Censor-aware energy bounds require a separate audit; gate uses explicit unknown bounds.')
    final=dict(status='COMPLETED DEVELOPMENT',controllers=C,states=S,eta=16,seeds=4,rows=rows,
               cross_controller_regret=cross,eligible_names=[names[i] for i in eligible],gate=dict(
                 count_eligible=len(eligible),count_rule_pass=len(eligible)>=8,regrets=gate,
                 p2_qualified=bool(len(eligible)>=8 and any(v['qualified'] for v in gate.values()))),
               energy=energy,provenance_sensitivity=provenance_sensitivity,unknown_total=int(U.sum()),new_rollouts=read('P1_COST.json')['genuinely_missing'],
               cached_rollouts=read('P1_COST.json')['cached_valid']+read('P1_COST.json')['cached_unknown'],
               limitation='Previously opened 24 development states; Q4 crossfit oracle and chance headroom do not prove deployable selection transfer.')
    write('P1_RESULTS.json',final)
    print(json.dumps({'eligible':final['eligible_names'],'gate':final['gate'],'rows':[{k:r[k] for k in ('name','fixed_success','state_oracle_gain')} for r in rows]},indent=2))


if __name__=='__main__':main()
