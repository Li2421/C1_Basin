"""Paired bounded-domain volume and cubical topology analysis."""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label,generate_binary_structure

HERE=Path(__file__).resolve().parent


def gf2_rank(columns):
    pivots={}
    for col in columns:
        v=col
        while v:
            lead=v.bit_length()-1
            if lead not in pivots:
                pivots[lead]=v
                break
            v ^= pivots[lead]
    return len(pivots)


def betti_from_voxels(mask):
    """Homology over F2 of the closed cubical complex formed by true voxels."""
    cells=[set() for _ in range(4)]
    for ix in zip(*np.where(mask)):
        stack=[(tuple(ix),(0,1,2))]
        while stack:
            base,axes=stack.pop();d=len(axes)
            key=(base,axes)
            if key in cells[d]:continue
            cells[d].add(key)
            for a in axes:
                rem=tuple(x for x in axes if x!=a)
                stack.append((base,rem))
                b=list(base);b[a]+=1
                stack.append((tuple(b),rem))
    seq=[sorted(c) for c in cells]
    ids=[{c:i for i,c in enumerate(s)} for s in seq]
    ranks=[0]
    for d in (1,2,3):
        columns=[]
        for base,axes in seq[d]:
            v=0
            for a in axes:
                rem=tuple(x for x in axes if x!=a)
                v ^= 1 << ids[d-1][(base,rem)]
                b=list(base);b[a]+=1
                v ^= 1 << ids[d-1][(tuple(b),rem)]
            columns.append(v)
        ranks.append(gf2_rank(columns))
    n=[len(s) for s in seq]
    beta=[n[0]-ranks[1],n[1]-ranks[1]-ranks[2],n[2]-ranks[2]-ranks[3],n[3]-ranks[3]]
    if any(x<0 for x in beta):raise AssertionError((n,ranks,beta))
    return dict(beta=beta,cells=n,boundary_ranks=ranks,euler=sum((-1)**d*n[d] for d in range(4)))


def main():
    protocol=json.loads((HERE/'protocol.json').read_text())
    bykey={}
    missing=[];bad=[]
    for c in protocol['cases']:
        stem=f"{c['uid']}_{c['kind']}_{c['index']:03d}_{c['chain']}"
        p=HERE/'panel_results'/(stem+'.json')
        if not p.exists():missing.append(stem);continue
        r=json.loads(p.read_text())
        if r['termination']=='numerical_failure' or r['rho'] is None:bad.append(stem)
        bykey[(c['uid'],c['kind'],c['index'],c['chain'])]=r
    if missing:raise RuntimeError(f'panel incomplete missing={len(missing)}')
    states=protocol['states'];N=64
    box=protocol['eta_box'];extent=float(np.prod(np.asarray(box['upper'])-box['lower']))
    summaries=[]
    for st in states:
        uid=st['uid'];pair=[]
        unknown_tt=unknown_ff=0;delta_lower=delta_upper=0
        for i in range(N):
            ra=bykey[(uid,'sobol64',i,'TT')];rb=bykey[(uid,'sobol64',i,'FF')]
            a,b=ra['success'],rb['success']
            ua=ra['termination']=='numerical_failure';ub=rb['termination']=='numerical_failure'
            unknown_tt+=ua;unknown_ff+=ub
            delta_lower+=int(bool(b))-int(bool(a) or ua)
            delta_upper+=int(bool(b) or ub)-int(bool(a))
            pair.append((bool(a),bool(b)))
        counts=dict(tt_only=sum(a and not b for a,b in pair),ff_only=sum(b and not a for a,b in pair),
                    both=sum(a and b for a,b in pair),neither=sum(not a and not b for a,b in pair))
        summaries.append(dict(uid=uid,scene=st['scenario'],counts=counts,
            TT_fraction=(counts['tt_only']+counts['both'])/N,FF_fraction=(counts['ff_only']+counts['both'])/N,
            gain_fraction=counts['ff_only']/N,loss_fraction=counts['tt_only']/N,
            delta_fraction=(counts['ff_only']-counts['tt_only'])/N,
            box_volume=extent,delta_volume=extent*(counts['ff_only']-counts['tt_only'])/N,
            numerical_failures=dict(TT=unknown_tt,FF=unknown_ff),
            delta_fraction_solver_uncertainty_interval=[delta_lower/N,delta_upper/N]))
    topology=[]
    for uid in protocol['grid_states']:
        scene=next(s['scenario'] for s in states if s['uid']==uid)
        for chain in ('TT','FF'):
            rec=[bykey[(uid,'grid5',i,chain)] for i in range(125)]
            values=np.array([r['rho'] if r['rho'] is not None else -np.inf for r in rec]).reshape(5,5,5)
            success=np.array([bool(r['success']) for r in rec]).reshape(5,5,5)
            unknown=np.array([r['termination']=='numerical_failure' for r in rec]).reshape(5,5,5)
            for res,sl in ((3,np.ix_([0,2,4],[0,2,4],[0,2,4])),(5,np.s_[:,:,:])):
                a=values[sl];m0=success[sl];unknown0=unknown[sl]
                for threshold in (-.02,-.005,0.,.005,.02):
                    m=m0 if threshold==0 else a>threshold
                    hom=betti_from_voxels(m)
                    _,b0_6=label(m,generate_binary_structure(3,1))
                    _,b0_26=label(m,generate_binary_structure(3,3))
                    edge=np.zeros_like(m,dtype=bool)
                    for ax in range(3):
                        lower=[slice(None)]*3;lower[ax]=0;edge[tuple(lower)]=True
                        upper=[slice(None)]*3;upper[ax]=-1;edge[tuple(upper)]=True
                    topology.append(dict(uid=uid,scene=scene,chain=chain,resolution=res,threshold=threshold,
                        success_voxels=int(m.sum()),numerical_failure_voxels=int(unknown0.sum()),
                        success_voxels_touching_box=int((m&edge).sum()),
                        component_counts_6_26=[int(b0_6),int(b0_26)],**hom))
                # For the actual binary evaluator, audit every assignment of
                # solver failures when the number is small enough.
                missing_idx=np.argwhere(unknown0)
                variants=[]
                if len(missing_idx)<=10:
                    for bits in range(1<<len(missing_idx)):
                        m=m0.copy()
                        for k,ix in enumerate(missing_idx):m[tuple(ix)]=bool((bits>>k)&1)
                        variants.append(tuple(betti_from_voxels(m)['beta']))
                topology.append(dict(uid=uid,scene=scene,chain=chain,resolution=res,threshold='numeric_failure_sensitivity',
                    unknown_voxels=int(len(missing_idx)),exhaustive=len(missing_idx)<=10,
                    possible_betti=sorted({x for x in variants}) if variants else []))
    rng=np.random.default_rng(20261008)
    stats={}
    for scene in ('ring_exchange','toy_give_way','both'):
        ss=[x for x in summaries if scene=='both' or x['scene']==scene]
        arr=np.asarray([x['delta_fraction'] for x in ss]);g=np.asarray([x['gain_fraction'] for x in ss]);l=np.asarray([x['loss_fraction'] for x in ss])
        boot=np.mean(arr[rng.integers(0,len(arr),size=(20000,len(arr)))],axis=1)
        stats[scene]=dict(states=len(ss),mean_delta_fraction=float(np.mean(arr)),median_delta_fraction=float(np.median(arr)),
            paired_state_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),ff_greater_fraction=float(np.mean(arr>0)),
            mean_gain_fraction=float(np.mean(g)),mean_loss_fraction=float(np.mean(l)),
            mean_delta_volume=float(extent*np.mean(arr)))
    result=dict(box=box,box_volume=extent,sampling='64 paired scrambled Sobol points per state, seed 20261008',
        states=summaries,scenario_stats=stats,topology=topology,
        numerical_failure_count=len(bad),numerical_failure_paths=bad,
        topology_warning='3^3/5^3 is only screening; nonmatching features need refinement before topological claims')
    (HERE/'panel_analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    xs=np.arange(len(summaries));delta=np.array([s['delta_fraction'] for s in summaries])
    axes[0].bar(xs,delta,color=['#1f77b4' if s['scene']=='ring_exchange' else '#ff7f0e' for s in summaries]);axes[0].axhline(0,color='black',lw=.8)
    axes[0].set_xticks(xs,[s['uid'].split('_')[-1] for s in summaries],rotation=65);axes[0].set_ylabel('FF − TT bounded volume fraction')
    gain=np.array([s['gain_fraction'] for s in summaries]);loss=np.array([s['loss_fraction'] for s in summaries])
    axes[1].scatter(loss,gain,c=['#1f77b4' if s['scene']=='ring_exchange' else '#ff7f0e' for s in summaries]);axes[1].plot([0,max(gain.max(),loss.max())],[0,max(gain.max(),loss.max())],'k--',lw=.8)
    axes[1].set_xlabel('TT-only fraction (loss)');axes[1].set_ylabel('FF-only fraction (gain)')
    fig.tight_layout();(HERE/'figures').mkdir(exist_ok=True);fig.savefig(HERE/'figures/volume_gain_loss.pdf');fig.savefig(HERE/'figures/volume_gain_loss.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(scenario_stats=stats,topology_zero=[x for x in topology if x['threshold']==0]),indent=2))


if __name__=='__main__':main()
