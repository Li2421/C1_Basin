"""Evaluate frozen fitted-state predictions on genuinely new rollout seeds."""
import csv
import numpy as np
from .study import HERE,ROOT,read,write,sha


def run():
    from shared_rollout_db.src.rollout_db import connect
    from diagnostics.controller_ff_factorial_repair_v1.review import (
        select,selection_difference,mse_gain,intervals,derangements,reversal,corr)
    from diagnostics.controller_q_diversity_v1.evaluate import donor_bootstrap
    from diagnostics.controller_flow_endpoint_probe_v1.analyze import centered,interval_center,energy
    p=read(HERE/'protocol.json');models=read(HERE/'models_frozen.json')
    assert sha(HERE/'models_frozen.json')==p['model_manifest_sha256']
    for z in models:
        assert sha(z['checkpoint'])==z['checkpoint_sha256'] and sha(z['predictions'])==z['prediction_sha256']
    states=sorted(s['state_uid'] for s in p['states']);cs=sorted(p['controllers']['ring_exchange']);eta=p['eta_indices']
    n,nc,k=len(states),len(cs),len(eta);shape=(n,nc,k,4)
    raw={key:np.zeros(shape,np.int8) for key in ('success','unknown','collision','deadlock','timeout','observed')}
    raw['rollout_uid']=np.full(shape,'','<U80');raw['termination']=np.full(shape,'','<U24')
    with connect(True) as db:
        for c in read(HERE/'cases.json'):
            r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            assert r is not None and not r['conflict_quarantined']
            assert all(r[key]==c[key] for key in ('state_uid','eta_uid','controller_uid','seed_key'))
            ix=(states.index(c['state_uid']),cs.index(c['controller_name']),eta.index(c['eta_index']),c['seed'])
            assert raw['observed'][ix]==0;raw['observed'][ix]=1
            for key in ('success','collision','deadlock','timeout'):raw[key][ix]=r[key]
            raw['unknown'][ix]=r['numerical_failure'];raw['rollout_uid'][ix]=r['rollout_uid'];raw['termination'][ix]=r['outcome']
    assert raw['observed'].all();assert not (raw['success']&raw['unknown']).any()
    np.savez_compressed(HERE/'raw_labels.npz',**raw,state_uid=np.array(states),controller=np.array(cs),eta=np.asarray(p['eta'])[eta])
    d=dict(np.load(ROOT/'controller_cached_union_v1/dataset.npz'))
    ri={(str(s),str(c)):i for i,(s,c) in enumerate(zip(d['state_uid'],d['controller']))}
    ids=np.array([ri[s,c] for s in states for c in cs]);pp={};seedpred={}
    for arm in dict.fromkeys(m['arm'] for m in models):
        aa=np.array([np.load(z['predictions'])['probability'][ids][:,eta].reshape(n,nc,k) for z in models if z['arm']==arm])
        pp[arm]=aa.mean(0);seedpred[arm]=aa
    pp['old_Q2_lookup_midpoint']=(raw['success'][...,:2]+.5*raw['unknown'][...,:2]).mean(-1)
    pp['old_controller_eta_prior']=np.broadcast_to(pp['old_Q2_lookup_midpoint'].mean(0,keepdims=True),(n,nc,k)).copy()
    lo=raw['success'][...,2:].astype(float);hi=lo+raw['unknown'][...,2:]
    qlo,qhi=lo.mean(-1),hi.mean(-1);mid=(qlo+qhi)/2
    boot=np.random.default_rng(2026102402).integers(n,size=(20000,n));arr={'bootstrap_indices':boot};saved={'state_uid':np.array(states),'controllers':np.array(cs),'permutations':derangements(n)}
    rows=[];mechanism=[];failures=[]
    def pack(l,u,al,au):
        return dict(identification_bounds=[float(l),float(u)],censor_CI95=[float(np.quantile(al,.025)),float(np.quantile(au,.975))])
    def projected(z,axes):return centered(z,axes) if axes else z
    def error_bounds(q,l,h,axes):
        x=projected(q,axes)
        al,ah=interval_center(l[...,0],h[...,0],axes) if axes else (l[...,0],h[...,0])
        bl,bh=interval_center(l[...,1],h[...,1],axes) if axes else (l[...,1],h[...,1])
        vals=np.stack([(x-a)*(x-b) for a in (al,ah) for b in (bl,bh)])
        dims=tuple(range(x.ndim-3,x.ndim));return vals.min(0).mean(dims),vals.max(0).mean(dims)
    def skill_bounds(q,l,h,axes):
        x=projected(q,axes);a=2*x*l.mean(-1)-x*x;b=2*x*h.mean(-1)-x*x
        dims=tuple(range(x.ndim-3,x.ndim));return np.minimum(a,b).mean(dims),np.maximum(a,b).mean(dims)
    for arm,q in pp.items():
        choice=q.argmax(-1);key=arm
        row=dict(arm=arm,N=n,controllers=nc,K=k,evaluation_seeds=[2,3],success_bounds=[float(select(qlo,choice).mean()),float(select(qhi,choice).mean())],
            Q_MSE_midpoint=float(np.mean((q-mid)**2)),selected_collision=int(select(raw['collision'][...,2:].sum(-1),choice).sum()),
            selected_unknown=int(select(raw['unknown'][...,2:].sum(-1),choice).sum()),selected_deadlock=int(select(raw['deadlock'][...,2:].sum(-1),choice).sum()),
            selected_timeout=int(select(raw['timeout'][...,2:].sum(-1),choice).sum()),
            oracle_bounds=[float(qlo.max(-1).mean()),float(qhi.max(-1).mean())])
        saved[arm+'__predictions']=q;saved[arm+'__choices']=choice;saved[arm+'__mismatched_choices']=choice[saved['permutations']]
        a,b=selection_difference(qlo,qhi,choice,pp['eta_only'].argmax(-1));row['minus_eta_only']=intervals(a.mean(1),b.mean(1),boot,key+'__eta',arr)
        a,b=mse_gain(pp['old_Q2_lookup_midpoint'],q,qlo,qhi);row['Q_MSE_gain_over_old_lookup']=intervals(a.mean((1,2)),b.mean((1,2)),boot,key+'__Qlookup',arr)
        al=np.zeros((n,n));au=al.copy();ql=al.copy();qu=al.copy()
        for donor in range(n):
            a,b=selection_difference(qlo,qhi,choice,np.broadcast_to(choice[donor],choice.shape));al[:,donor],au[:,donor]=a.mean(1),b.mean(1)
            a,b=mse_gain(q[donor],q,qlo,qhi);ql[:,donor],qu[:,donor]=a.mean((1,2)),b.mean((1,2))
        for name,l,h in [('matched_minus_mismatch',al,au),('wrong_state_Q_MSE_minus_matched',ql,qu)]:
            v,bl,ok=donor_bootstrap(l,boot);w,bu,ok2=donor_bootstrap(h,boot);good=ok&ok2
            row[name]=pack(v,w,bl[good],bu[good]);arr[key+'__'+name+'_lower']=bl;arr[key+'__'+name+'_upper']=bu
        dl,du=selection_difference(qlo[...,None,:],qhi[...,None,:],choice[...,None],np.arange(k)[None,None,:])
        bl=[];bu=[]
        for start in range(0,len(boot),500):
            ib=boot[start:start+500];bl.append(dl[ib].mean(1).min(-1).mean(-1));bu.append(du[ib].mean(1).min(-1).mean(-1))
        bl,bu=np.concatenate(bl),np.concatenate(bu)
        row['minus_reselected_best_per_C_fixed']=pack(dl.mean(0).min(-1).mean(),du.mean(0).min(-1).mean(),bl,bu)
        arr[key+'__fixed_lower']=bl;arr[key+'__fixed_upper']=bu
        row['reversal']=reversal(lo,hi,np.where(abs(q-q.mean(0,keepdims=True))<1e-7,q.mean(0,keepdims=True),q),boot)
        rows.append(row)
        for label,axes in [('raw_Q',[]),('state_eta',[0,2]),('state_controller_eta',[0,1,2])]:
            l,h=error_bounds(q,lo,hi,axes);sl,sh=skill_bounds(q,lo,hi,axes);el=[];eu=[];gl=[];gu=[]
            for start in range(0,len(boot),500):
                ib=boot[start:start+500];ba=[a+1 for a in axes];a,b=error_bounds(q[ib],lo[ib],hi[ib],ba);el.append(a);eu.append(b)
                a,b=skill_bounds(q[ib],lo[ib],hi[ib],ba);gl.append(a);gu.append(b)
            el,eu,gl,gu=map(np.concatenate,(el,eu,gl,gu));tag=key+'__'+label
            for name,value in [('error_lower',el),('error_upper',eu),('skill_lower',gl),('skill_upper',gu)]:arr[tag+'__'+name]=value
            xp=projected(q,axes);yp=projected(mid,axes)
            mechanism.append(dict(arm=arm,interaction=label,noise_corrected_MSE=pack(l,h,el,eu),skill_over_zero=pack(sl,sh,gl,gu),
                correlation_midpoint_sensitivity=corr(xp,yp) if min(xp.std(),yp.std())>1e-7 else None))
        for i,s in enumerate(states):
            for c,ctl in enumerate(cs):
                j=int(choice[i,c]);failures.append(dict(arm=arm,state_uid=s,controller=ctl,eta_index=eta[j],
                    success_lower=float(qlo[i,c,j]),success_upper=float(qhi[i,c,j]),unknown=int(raw['unknown'][i,c,j,2:].sum()),
                    collision=int(raw['collision'][i,c,j,2:].sum()),deadlock=int(raw['deadlock'][i,c,j,2:].sum()),timeout=int(raw['timeout'][i,c,j,2:].sum())))
    write(HERE/'results.json',dict(rows=rows,mechanism=mechanism,old_outcomes=1280,new_outcomes=1280,
        total_unknown_old=int(raw['unknown'][...,:2].sum()),total_unknown_new=int(raw['unknown'][...,2:].sum()),collision_old=int(raw['collision'][...,:2].sum()),collision_new=int(raw['collision'][...,2:].sum()),
        scope='Fixed SOURCE TRAIN physicalstates,independentrolloutseeds,never a fresh-stateorLOSOconfirmation. Models unchanged. Oldlookup is a fixed diagnostic predictor;notdeploymentgeneralization.'))
    np.savez_compressed(HERE/'bootstrap.npz',**arr);np.savez_compressed(HERE/'predictions_and_controls.npz',**saved)
    np.savez_compressed(HERE/'individual_seed_predictions.npz',**seedpred)
    with (HERE/'per_state_cases.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(failures[0]));w.writeheader();w.writerows(failures)
    print('Independentseedpanel complete; saved every outcome,prediction,control and physical-state bootstrap.',flush=True)


if __name__=='__main__':run()
