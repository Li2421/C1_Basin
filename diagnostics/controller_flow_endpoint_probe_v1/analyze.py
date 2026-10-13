"""Independent-half signal audit; bounds retain every numerical unknown."""
import itertools
import numpy as np
from .study import HERE,read,write,FAMILIES
from shared_rollout_db.src.rollout_db import connect

AXES={'state':0,'controller':1,'eta':2}
GROUPS=[('controller',),('eta',),('state',),('controller','eta'),('state','eta'),('state','controller'),('state','controller','eta')]

def centered(a,axes):
    for axis in axes:a=a-a.mean(axis,keepdims=True)
    return a

def interval_center(lo,hi,axes):
    for axis in axes:
        n=lo.shape[axis]
        lo,hi=(1-1/n)*lo-(hi.sum(axis,keepdims=True)-hi)/n,(1-1/n)*hi-(lo.sum(axis,keepdims=True)-lo)/n
    return lo,hi

def energy(lo,hi,axes):
    # Independent seed halves. Interval propagation is conservative when
    # multiple centerings reuse the same unknown variable.
    al,ah=interval_center(lo[...,0],hi[...,0],axes);bl,bh=interval_center(lo[...,1],hi[...,1],axes)
    corners=np.stack((al*bl,al*bh,ah*bl,ah*bh))
    mid=(lo+hi)/2;a=centered(mid[...,0],axes);b=centered(mid[...,1],axes)
    dims=tuple(range(a.ndim-3,a.ndim))
    return (a*b).mean(dims),corners.min(0).mean(dims),corners.max(0).mean(dims)

def run():
    p=read(HERE/'protocol.json');cases=read(HERE/'cases.json');scenes=sorted(p['controllers']);refs=('goal','flow')
    eta_idx={j:k for k,j in enumerate(p['eta_indices'])};shape=(4,8,2,5,4,2)
    success=np.zeros(shape,np.int8);unknown=np.zeros(shape,np.int8);collision=np.zeros(shape,np.int8);length=np.zeros(shape,np.int32);seen=np.zeros(shape,bool)
    terms=np.empty(shape,'<U24');uids=np.empty(shape,'<U80')
    with connect(True) as db:
        for c in cases:
            row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone();assert row is not None,c['rollout_uid'];assert not row['conflict_quarantined']
            prefix,family=c['controller_name'].split('_',1)
            ix=(scenes.index(c['scene']),c['state_index'],refs.index(prefix),FAMILIES.index(family),eta_idx[c['eta_index']],c['seed'])
            assert not seen[ix];seen[ix]=True
            success[ix]=row['success'];unknown[ix]=row['numerical_failure'];collision[ix]=row['collision'];length[ix]=row['episode_length'];terms[ix]=row['outcome'];uids[ix]=c['rollout_uid']
    assert seen.all()
    np.savez_compressed(HERE/'raw_labels.npz',success=success,unknown=unknown,collision=collision,episode_length=length,
        termination=terms,rollout_uid=uids,scene=np.array(scenes),reference=np.array(refs),controller=np.array(FAMILIES),eta=np.asarray(p['eta'])[p['eta_indices']])
    rng=np.random.default_rng(2026101212);boot=rng.integers(8,size=(20000,8));rows=[];effects=[];arrays={}
    for si,sc in enumerate(scenes):
        for ri,ref in enumerate(refs):
            lo=success[si,:,ri].astype(float);hi=lo+unknown[si,:,ri];mid=(lo+hi)/2
            counts={name:int(np.sum(terms[si,:,ri]==name)) for name in np.unique(terms[si,:,ri])}
            # Lower-bound discovery selects deterministically. Evaluation uses
            # independent seed1, with explicit worst/best unknown completions.
            selected=lo[...,0].argmax(-1);fixed=lo[...,0].mean(0).argmax(-1)
            a=np.take_along_axis(lo[...,1],selected[...,None],-1)[...,0];b=np.take_along_axis(hi[...,1],selected[...,None],-1)[...,0]
            fixed=np.broadcast_to(fixed,(8,5));f0=np.take_along_axis(lo[...,1],fixed[...,None],-1)[...,0];f1=np.take_along_axis(hi[...,1],fixed[...,None],-1)[...,0]
            equal=selected==fixed;dl=np.where(equal,0,a-f1).mean(-1);dh=np.where(equal,0,b-f0).mean(-1)
            # Qualifying reversal pairs are defined only on discovery seed0.
            # Confirm both directions on independent seed1; no best-pair pick.
            pairs=[]
            for ci in range(5):
                for j,k in itertools.combinations(range(4),2):
                    positive=lo[:,ci,j,0]-hi[:,ci,k,0]>=1
                    negative=lo[:,ci,k,0]-hi[:,ci,j,0]>=1
                    if positive.sum()<2 or negative.sum()<2:continue
                    da=lo[positive,ci,j,1]-hi[positive,ci,k,1]
                    db=lo[negative,ci,k,1]-hi[negative,ci,j,1]
                    pairs.append(dict(controller=FAMILIES[ci],eta_pair=[p['eta_indices'][j],p['eta_indices'][k]],
                        positive_states=int(positive.sum()),negative_states=int(negative.sum()),evaluation_positive_lower=float(da.mean()),
                        evaluation_negative_lower=float(db.mean()),both_directions_confirmed=bool(da.mean()>=.5 and db.mean()>=.5)))
            row=dict(scene=sc,reference=ref,N=8,controllers=5,candidates=4,rollouts=int(lo.size),success=int(lo.sum()),
                unknown=int(unknown[si,:,ri].sum()),collision=int(collision[si,:,ri].sum()),termination=counts,
                success_bounds=[float(lo.mean()),float(hi.mean())],
                crossfit_selection_gain_bounds=[float(dl.mean()),float(dh.mean())],
                crossfit_selection_gain_censor_CI95=[float(np.quantile(dl[boot].mean(1),.025)),float(np.quantile(dh[boot].mean(1),.975))],
                qualifying_reversal_pairs=len(pairs),independently_confirmed_reversal_pairs=sum(v['both_directions_confirmed'] for v in pairs),reversal_cases=pairs)
            rows.append(row)
            for group in GROUPS:
                axes=[AXES[k] for k in group];point,lower,upper=energy(lo,hi,axes);bs=[];bl=[];bh=[]
                for start in range(0,len(boot),1000):
                    ii=boot[start:start+1000];a,b,c=energy(lo[ii],hi[ii],[k+1 for k in axes]);bs.append(a);bl.append(b);bh.append(c)
                bs,bl,bh=np.concatenate(bs),np.concatenate(bl),np.concatenate(bh)
                name=sc+'_'+ref+'_'+'x'.join(group);arrays[name]=bs;arrays[name+'_lower']=bl;arrays[name+'_upper']=bh
                effects.append(dict(scene=sc,reference=ref,centered_over=list(group),
                    independent_half_energy_midpoint=float(point),identification_bounds=[float(lower),float(upper)],
                    censor_CI95=[float(np.quantile(bl,.025)),float(np.quantile(bh,.975))]))
        delta_lo=(success[si,:,1]-success[si,:,0]-unknown[si,:,0]).mean((1,2,3))
        delta_hi=(success[si,:,1]+unknown[si,:,1]-success[si,:,0]).mean((1,2,3))
        rows.append(dict(scene=sc,reference='paired_flow_minus_goal',N=8,success_difference_bounds=[float(delta_lo.mean()),float(delta_hi.mean())],
            censor_CI95=[float(np.quantile(delta_lo[boot].mean(1),.025)),float(np.quantile(delta_hi[boot].mean(1),.975))]))
    np.savez_compressed(HERE/'signal_bootstrap.npz',**arrays)
    write(HERE/'data_quality_results.json',dict(rows=rows,signal=effects,
        interpretation='DEVELOPMENT data-quality test, not fitted-model validation. Cross-half covariance estimates reproducible variation; negative estimates reflect noise. Midpoints of censored labels are only a displayed sensitivity point; decisions use the conservative identification/CI bounds.',
        centering='Demean the stated factors while retaining unmentioned factors. For example state+eta removes per-controller global eta and per-state global success: it measures state-dependent eta structure, including three-way interaction.',
        limitations='Only8TRAINstates/scene,4prefixedeta,2rolloutseeds. All candidates/poses fixed before new outcomes. Reversal analysis is low-power and does not replace later K16/independent robust confirmation.',
        new_outcomes=1280,reused_outcomes=1280,collision_total=int(collision.sum())))
    for r in rows:print({k:v for k,v in r.items() if k in ('scene','reference','success','unknown','success_difference_bounds','censor_CI95','crossfit_selection_gain_bounds','independently_confirmed_reversal_pairs')})
    for r in effects:
        if r['centered_over']==['state','eta']:print('state_eta_signal',r)

if __name__=='__main__':run()
