"""Zero-control upper completion and independent fixed-horizon association audit."""
import json
import hashlib
import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
import audit_c1_joint_closed_loop as experiment
from audit_c1_joint_witness_risk import ROOT, PROTOCOL, save, clean, auc
from single_integrator.evaluate import load_policy

OUT=ROOT/'results/c1_completed_waiting_risk_audit'
H=42.5;DT=.05;K=850;KAPPA=.05
MAXR=1+PROTOCOL['eta']*PROTOCOL['tau']*np.logaddexp(0.,(PROTOCOL['delta']+1)/PROTOCOL['tau'])
FIELDS=['risk','geometry','progress','stop_only','legacy_upper','unfinished_time','remaining_distance','risk_k025','risk_k010']

def completion(g,z,k=KAPPA):
    z=np.asarray(z,float);g=np.asarray(g,float)
    if np.any((z>0)&~np.isfinite(g)):raise ValueError('Undefined nonzero-direction risk must not be imputed.')
    g=np.where(z==0,1,g)
    return (k*k+z*g)/(k*k+z)

def math_checks():
    from audit_c1_joint_witness_risk import LocalRisk
    from audit_c1_fixed_horizon_waiting import mathematical_checks
    scales=np.array([0.,1e-9,1e-6,1e-3,.01,.05,.1,1.])
    rows=[]
    for g in [0.,.2,.8,1.]:
        values=completion(np.full(len(scales),g),scales**2)
        assert values[0]==1 and np.all(np.diff(values)<=1e-12)
        rows.append(dict(g=g,normalized_amplitudes=scales,risks=values))
    # Gradient at zero is zero: difference quotients converge to zero from all rays.
    eps=np.array([1e-3,1e-4,1e-5,1e-6])
    quotient=(completion(np.zeros(4),eps**2)-1)/eps
    assert abs(quotient[-1])<.001
    joint=[]
    for m in [4,16,64]:
        e=np.eye(m);A=np.vstack([e[0],e[1:],-e[1:]])
        q=np.zeros(m);q[:2]=[-.25,np.sqrt(1-.25**2)]
        model=LocalRisk(A,np.zeros(len(A)),np.full(m//2,.5));v=.1*q
        s=model.score(v);g=s['risk']/MAXR;c=float(completion(g,np.sum((v/.5)**2)))
        assert c>=g and s['B']==1
        joint.append(dict(m=m,original=g,completed=c))
    # Falsification: increasing a blocked command can lower the completion without moving.
    model=LocalRisk(np.eye(4),np.zeros(4),np.full(2,.5))
    q=-np.ones(4)/2;amplitude_counterexample=[]
    for a in [.001,.01,.1,.5]:
        raw=model.score(a*q);g=raw['risk']/MAXR
        amplitude_counterexample.append(dict(a=a,B=raw['B'],original=g,
            completed=float(completion(g,(a/.5)**2)),applied_norm=float(np.linalg.norm(model.physical(a*q)))))
    assert amplitude_counterexample[-1]['completed']<amplitude_counterexample[0]['completed']
    assert max(r['applied_norm'] for r in amplitude_counterexample)<1e-7
    return dict(scale_tests=rows,zero_derivative_quotients=quotient,joint_rays=joint,
        blocked_amplitude_increase_shortcut=amplitude_counterexample,
        old_epsilon_failure=mathematical_checks(),
        limitation='Differentiable at v=0 with zero gradient; angular QP/witness nonsmoothness away from zero is inherited. No BPTT proof.')

def series(z,stride):
    T=min(len(z['positions_before']),K)
    completed=bool(np.any(z['success'][:T]));assert T==K or completed
    ids=np.asarray(z['idx']);choose=np.flatnonzero((ids<T)&((ids%stride==0)|(ids==T-1)))
    idx=ids[choose];assert idx[0]==0
    original=np.asarray(z['risk'])[choose]/MAXR
    speed2=np.sum((np.asarray(z['candidate'])[idx].reshape(-1,4)/.5)**2,axis=1)
    if np.any(original[np.isfinite(original)] < -1e-7) or np.any(original[np.isfinite(original)] >1+1e-7):raise ValueError('risk bound violation')
    sampled={f'geometry{k}':completion(original,speed2,k) for k in [.025,.05,.1]}
    sampled['stop_only']=KAPPA**2/(KAPPA**2+speed2)
    sampled['legacy_geometry']=np.where(speed2==0,1,original)
    assert all(np.isfinite(a).all() for a in sampled.values())
    # Piecewise-constant time quadrature; only sampled-time values are local predictions.
    lookup=np.searchsorted(idx,np.arange(T),side='right')-1
    arrays={key:np.r_[v[lookup],np.zeros(K-T)] for key,v in sampled.items()}
    pos=np.concatenate([z['positions_after'][:T],np.repeat(z['positions_after'][T-1][None],K-T,axis=0)])
    dist=np.linalg.norm(pos-z['goals'],axis=-1)
    initial=np.linalg.norm(z['positions_before'][0]-z['goals'],axis=-1).sum()+1e-5
    V=np.r_[initial-1e-5,dist.sum(1)]/initial
    U=1-np.prod(1-expit((dist-.08)/.02),axis=1);U[T:]=0
    end=np.arange(1,K+1);start=np.maximum(end-80,0)
    rate=(V[start]-V[end])/((end-start)*DT)
    p=U*expit((.005-rate)/.00125);p[T:]=0
    arrays.update(progress=p,unfinished_time=U,remaining_distance=np.max(dist,axis=1))
    arrays['geometry']=arrays.pop('geometry0.05')
    arrays['risk']=.5*(arrays['geometry']+p)
    arrays['risk_k025']=.5*(arrays.pop('geometry0.025')+p)
    arrays['risk_k010']=.5*(arrays.pop('geometry0.1')+p)
    arrays['legacy_upper']=.5*(arrays['legacy_geometry']+p)
    return arrays,idx,T,completed

def scalar(a,end):
    return {f:float(np.mean(a[f][:end])) if f!='remaining_distance' else float(a[f][end-1]) for f in FIELDS}

def evaluate(rows):
    result=dict(n=len(rows),n_deadlock=sum(r['deadlock'] for r in rows),n_failure=sum(r['failure'] for r in rows))
    for label in ['deadlock','failure']:
        y=np.array([r[label] for r in rows]);result[label]={}
        for f in FIELDS:
            s=np.array([r[f] for r in rows]);r=float(spearmanr(y,s).statistic) if len(set(y))==2 and len(set(s))>1 else None
            result[label][f]=dict(auc=auc(y,s),spearman=r)
    return result

def scheduled_wait_check():
    """A planned long wait can trigger the event and still finish within H."""
    from single_integrator.environment import GiveWayEnv, Config
    from single_integrator.cbf import CBFConfig, barrier_constraints
    meta=json.loads(experiment.META.read_text());meta['environment']['terminate_on_deadlock']=False
    env=GiveWayEnv(Config(**meta['environment']));cbf=CBFConfig(**meta['cbf'])
    with np.load(ROOT/'results/c1_fixed_horizon_waiting_audit/traces/0003_baseline.npz') as old:
        env.reset(old['positions_before'][0]);delay=160
        data={k:[] for k in ['positions_before','positions_after','candidate','applied','risk','success','deadlock']}
        for t in range(K):
            if t<delay:u=np.zeros(4);v=u;r=np.nan
            else:u=old['applied'][t-delay];v=old['candidate'][t-delay];r=old['risk'][t-delay]
            x=env.positions.copy();A,b,_=barrier_constraints(env.snapshot(),cbf)
            assert np.min(A@u-b)>-1e-8
            _,_,done,info=env.step(u.reshape(2,2))
            if t>=delay:np.testing.assert_array_equal(env.positions,old['positions_after'][t-delay])
            values=dict(positions_before=x,positions_after=env.positions.copy(),candidate=v,applied=u,risk=r,
                        success=info['task_success'],deadlock=info['deadlock'])
            for k in data:data[k].append(values[k])
            if done:break
        z={k:np.asarray(v) for k,v in data.items()};z.update(idx=np.arange(len(z['risk'])),goals=env.goals,dt=np.array(DT))
        a,_,T,done=series(z,1);events=np.flatnonzero(z['deadlock']);assert done and len(events)
        baseline,_,_,_=series(old,1)
        return dict(rid=3,wait_seconds=8.,first_deadlock_seconds=float((events[0]+1)*DT),completion_seconds=T*DT,
            deadlock_event=True,unfinished_at_H=False,risk=scalar(a,K),baseline=scalar(baseline,K),
            note='The actual plant/detector was stepped. After waiting, exact baseline positions/actions with shifted policy noise; detector state is absent from policy observation. This is a protocol event, not permanent deadlock.')

def main():
    OUT.mkdir(exist_ok=True);(OUT/'traces').mkdir(exist_ok=True)
    protocol=dict(horizon=H,kappa=KAPPA,sensitivity=[.025,.1],direction_completion='(kappa^2+z*g)/(kappa^2+z), z=sum_a ||v_a/s_a||^2; v=0 -> 1',
        scale='s_a=.5 from physical caps; kappa dimensionless, fixed before association scoring, no optimization.',
        primary='200 distinct historical baseline initial IDs; seven deadlocks actually continued to H.',
        primary_geometry_stride_seconds=1.,paired_geometry_stride_seconds=.05,
        heldout='Historical internal split rid%5==0; only one positive, not fresh external validation.',
        labels='ever independent environment deadlock by H; separately unfinished at H; no labels in score.',
        bootstrap='Primary trajectory resampling, seed 20260914, 1000 draws.',
        caution='Overlap with earlier audits; diagnostic speed term partly resembles outcome protocol. Compare progress and speed ablations.')
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:save(OUT/'protocol.json',protocol)
    save(OUT/'mathematical_checks.json',math_checks())
    index=json.loads((ROOT/'results/c1_joint_witness_risk_audit/index.json').read_text())
    rows=[];local=[]
    for rec in [r for r in index if r['source']=='primary']:
        rid=rec['rid'];source=ROOT/rec['score_path']
        if rec['outcome']=='safe_deadlock':
            source=ROOT/f'results/c1_fixed_horizon_waiting_audit/traces/{rid:04d}_baseline.npz'
            if not source.exists():
                source=OUT/'traces'/f'{rid:04d}_baseline.npz'
                if not source.exists():
                    meta=json.loads(experiment.META.read_text());meta['environment']['terminate_on_deadlock']=False
                    experiment.OUT=OUT;policy,_=load_policy(meta['checkpoint'])
                    experiment.rollout(policy,meta,rid,'baseline',require_historical_length=False)
        with np.load(source) as z:
            a,idx,T,done=series(z,20)
            events=np.flatnonzero(z['deadlock'][:K]);event=int(events[0]+1) if len(events) else K+1
            base=dict(source='primary',rid=rid,split=rec['split'],deadlock=event<=K,failure=not done,first_event=event*DT if event<=K else None)
            for seconds in [10,20,H]:
                end=round(seconds/DT)
                # Prefix prediction excludes already-observed success/deadlock.
                if seconds<H and (event<=end or T<=end):continue
                rows.append(dict(**base,seconds=seconds,**scalar(a,end)))
            for t in idx:
                observed=t+1 # progress uses this transition's endpoint, so forecast after it
                if observed>=event or observed+100>K:continue
                # Strictly before the first deadlock; no trigger/post-event alarm frames.
                local.append(dict(rid=rid,split=rec['split'],y=bool(observed<event<=observed+100),
                    geometry=float(a['geometry'][t]),risk=float(a['risk'][t]),
                    progress=float(a['progress'][t]),stop_only=float(a['stop_only'][t]),legacy_geometry=float(a['legacy_geometry'][t])))
    save(OUT/'primary_scores.json',rows)
    metrics={}
    for split in ['all','development','heldout']:
        for seconds in [10,20,H]:
            rr=[r for r in rows if r['seconds']==seconds and (split=='all' or r['split']==split)]
            metrics[f'{split}/{seconds}']=evaluate(rr)
    save(OUT/'primary_metrics.json',metrics)
    # IID bootstrap only across distinct primary initial IDs, never pooled paired arms.
    rng=np.random.default_rng(20260914);ci={}
    for seconds in [20,H]:
        rr=[r for r in rows if r['seconds']==seconds]
        for label in ['deadlock','failure']:
            y=np.array([r[label] for r in rr]);ss=np.array([[r[f] for f in ['risk','geometry','progress','stop_only','legacy_upper']] for r in rr])
            vals=[]
            for _ in range(1000):
                ii=rng.integers(0,len(rr),len(rr));v=[auc(y[ii],ss[ii,j]) for j in range(ss.shape[1])]
                if v[0] is not None:vals.append(v)
            ci[f'{seconds}/{label}']=dict(fields=['risk','geometry','progress','stop_only','legacy_upper'],
                interval95=np.quantile(vals,[.025,.975],axis=0).T,
                risk_minus_progress_interval95=np.quantile(np.array(vals)[:,0]-np.array(vals)[:,2],[.025,.975]),replicates=len(vals))
    save(OUT/'bootstrap.json',ci)
    lm={}
    for split in ['all','development','heldout']:
        ll=[r for r in local if split=='all' or r['split']==split];counts={i:sum(r['rid']==i for r in ll) for i in {r['rid'] for r in ll}}
        w=[1/counts[r['rid']] for r in ll]
        lm[split]=dict(n=len(ll),positives=sum(r['y'] for r in ll),positive_episodes=len({r['rid'] for r in ll if r['y']}),
            auc={f:auc([r['y'] for r in ll],[r[f] for r in ll],w) for f in ['risk','geometry','progress','stop_only','legacy_geometry']})
    save(OUT/'local_prediction.json',lm)
    # Matched intervention/delay pairs are external to parameter choice but not independent IDs.
    paired=[]
    oldrows=json.loads((ROOT/'results/c1_fixed_horizon_waiting_audit/episodes.json').read_text())
    for r in oldrows:
        with np.load(ROOT/f'results/c1_fixed_horizon_waiting_audit/traces/{r["rid"]:04d}_{r["arm"]}.npz') as z:
            a,idx,T,done=series(z,1)
            paired.append(dict(rid=r['rid'],arm=r['arm'],deadlock=r['first_deadlock_seconds'] is not None,failure=not done,**scalar(a,K)))
    comparisons=[]
    for r in paired:
        if r['arm']=='baseline':continue
        b=next(b for b in paired if b['rid']==r['rid'] and b['arm']=='baseline')
        comparisons.append(dict(rid=r['rid'],arm=r['arm'],risk_change=r['risk']-b['risk'],
            deadlock_before=b['deadlock'],deadlock_after=r['deadlock'],failure_before=b['failure'],failure_after=r['failure']))
    save(OUT/'paired_scores.json',dict(rows=paired,comparisons=comparisons))
    save(OUT/'scheduled_wait_counterexample.json',scheduled_wait_check())
    print(json.dumps(clean(dict(metrics={k:v for k,v in metrics.items() if k in ['all/42.5','all/20','heldout/42.5']},local=lm)),indent=2))

if __name__=='__main__':main()
