"""Fixed 42.5s audit: continue past deadlock; never invent a zero-risk failure tail."""
import json
import shutil
import numpy as np
from scipy.special import expit
import audit_c1_joint_closed_loop as experiment
from audit_c1_joint_witness_risk import ROOT, save, PROTOCOL
from single_integrator.evaluate import load_policy

OLD=ROOT/'results/c1_joint_closed_loop_audit'
OUT=ROOT/'results/c1_fixed_horizon_waiting_audit'
H=42.5

def fixed_scores(z):
    dt=float(z['dt']);K=round(H/dt);L=min(len(z['risk']),K)
    success=np.flatnonzero(z['success'][:L]);completed=bool(len(success))
    assert L==K or completed,'Unobserved failure tail cannot be padded.'
    # A task-completed absorbing state is a scoring convention, not a zero q.
    active=np.zeros(K,bool);active[:L]=True
    before=np.concatenate([z['positions_before'][:L],np.repeat(z['positions_after'][L-1][None],K-L,axis=0)])
    after=np.concatenate([z['positions_after'][:L],np.repeat(z['positions_after'][L-1][None],K-L,axis=0)])
    dist=np.linalg.norm(after-z['goals'],axis=-1)
    initialdist=np.linalg.norm(before[0]-z['goals'],axis=-1)
    denom=initialdist.sum()+1e-5
    V=np.r_[initialdist.sum()/denom,dist.sum(1)/denom]
    U=1-np.prod(1-expit((dist-.08)/.02),axis=1);U[~active]=0
    end=np.arange(1,K+1);W=round(4/dt);start=np.maximum(end-W,0)
    available=(end-start)*dt
    progress_rate=(V[start]-V[end])/available
    # Original window thresholds expressed as rates; startup uses actual duration.
    p=U*expit((.02/4-progress_rate)/(.005/4));p[~active]=0
    old_p=p.copy();old_p[end<W]=0 # diagnostic only: previously unscored startup
    maxrisk=1+PROTOCOL['eta']*PROTOCOL['tau']*np.logaddexp(0.,(PROTOCOL['delta']+1)/PROTOCOL['tau'])
    g=np.zeros(K);g[:L]=z['risk'][:L]/maxrisk
    unknown=active&~np.isfinite(g)
    lo=np.where(unknown,0,g);hi=np.where(unknown,1,g)
    return dict(horizon=H,observed_seconds=L*dt,completed=completed,
        geometry_unknown_seconds=float(unknown.sum()*dt),
        geometry_lower=float(np.mean(lo)),geometry_upper=float(np.mean(hi)),
        progress=float(np.mean(p)),progress_without_startup=float(np.mean(old_p)),
        risk_lower=float(.5*np.mean(lo)+.5*np.mean(p)),risk_upper=float(.5*np.mean(hi)+.5*np.mean(p)),
        unfinished_seconds=float(U.sum()*dt),remaining_distance=float(dist[-1].max()),
        progress_first2_seconds=float(p[:round(2/dt)].mean()))

def mathematical_checks():
    # Nonconstant degree-zero directional risk cannot have a unique continuous limit at zero.
    qs=[np.array([1.,0.]),np.array([-1.,0.])]
    eps=[1.,1e-3,1e-6]
    regularized=[]
    for a in eps:
        v=-a*qs[0];qeps=v/np.sqrt(v@v+1e-4)
        regularized.append(float(np.minimum(qeps,0)@np.minimum(qeps,0)))
    assert regularized[-1]<regularized[0]/1000
    # Joint progress: an agent may wait while another advances; all waiting differs.
    rho=.005;tau=.00125
    stall=lambda rate:float(expit((rho-rate)/tau))
    return dict(epsilon_normalization_B_for_blocked_ray=regularized,
        all_wait_stall=stall(0),one_wait_other_progress_stall=stall(.02),
        note='Rate examples assume unfinished joint task, not empirical deadlock probabilities.')

def main():
    OUT.mkdir(exist_ok=True);(OUT/'traces').mkdir(exist_ok=True)
    meta=json.loads(experiment.META.read_text());meta['environment']['max_steps']=round(H/meta['environment']['dt'])
    meta['environment']['terminate_on_deadlock']=False
    oldrows=json.loads((OLD/'episodes.json').read_text())
    save(OUT/'protocol.json',dict(horizon_seconds=H,steps=850,dt=.05,
        deadlock='Keep detector, continue closed loop after trigger; report ever-deadlock and recovery separately.',
        success='Absorb only after actual joint task completion; remaining task costs zero.',
        waiting='Keep undefined geometry as [0,1] interval; no NaN deletion or point imputation.',
        progress='Partial startup window normalized by actual elapsed time; original thresholds converted to rates.',
        inherited= str((OLD/'protocol.json').relative_to(ROOT))))
    experiment.OUT=OUT
    rows=json.loads((OUT/'episodes.json').read_text()) if (OUT/'episodes.json').exists() else []
    policy=None
    for old in oldrows:
        rid,arm=old['rid'],old['arm'];name=f'{rid:04d}_{arm}.npz'
        if any(r['rid']==rid and r['arm']==arm for r in rows):continue
        if old['outcome']=='safe_deadlock':
            if policy is None:policy,_=load_policy(meta['checkpoint'])
            experiment.rollout(policy,meta,rid,arm,require_historical_length=False)
        else:
            shutil.copyfile(OLD/'traces'/name,OUT/'traces'/name)
        with np.load(OUT/'traces'/name) as z:
            score=fixed_scores(z);L=min(len(z['risk']),850)
            events=np.flatnonzero(z['deadlock'][:L]);success=np.flatnonzero(z['success'][:L])
            first=float((events[0]+1)*.05) if len(events) else None
            completed=bool(len(success));label='success' if completed else 'unfinished_at_horizon'
            # Continued rollout must reproduce the entire original pre-terminal prefix.
            original=np.load(OLD/'traces'/name);n=min(L,len(original['risk']))
            error=float(np.max(np.abs(z['positions_after'][:n]-original['positions_after'][:n])))
            assert error==0, (rid,arm,error)
            row=dict(rid=rid,arm=arm,original_outcome=old['outcome'],fixed_outcome=label,
                first_deadlock_seconds=first,recovered_after_deadlock=bool(completed and len(events)),
                completion_seconds=float((success[0]+1)*.05) if completed else None,prefix_error=error,**score)
        rows.append(row);save(OUT/'episodes.json',rows)
        print(f'{rid} {arm}: {label}, deadlock={first}, risk=[{score["risk_lower"]:.4f},{score["risk_upper"]:.4f}]',flush=True)
    comparisons=[]
    for r in rows:
        if r['arm']=='baseline':continue
        b=next(b for b in rows if b['rid']==r['rid'] and b['arm']=='baseline')
        comparisons.append(dict(rid=r['rid'],arm=r['arm'],before=b['fixed_outcome'],after=r['fixed_outcome'],
            risk_change_lower=r['risk_lower']-b['risk_upper'],risk_change_upper=r['risk_upper']-b['risk_lower'],
            progress_change=r['progress']-b['progress'],unfinished_seconds_change=r['unfinished_seconds']-b['unfinished_seconds']))
    save(OUT/'summary.json',dict(n=len(rows),resumed_deadlocks=sum(r['original_outcome']=='safe_deadlock' for r in rows),
        recovered=[dict(rid=r['rid'],arm=r['arm'],completion_seconds=r['completion_seconds']) for r in rows if r['recovered_after_deadlock']],
        comparisons=comparisons,mathematical_checks=mathematical_checks()))

if __name__=='__main__':main()
