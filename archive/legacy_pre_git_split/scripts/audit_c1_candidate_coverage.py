"""Expanded interventions only; fixed two-horizon scorer and controller."""
import json
import time
import hashlib
from dataclasses import replace
from itertools import product
from pathlib import Path
import numpy as np
from scipy.special import expit
import jax
import jax.numpy as jnp
from audit_c1_joint_witness_risk import ROOT,LocalRisk,save
from audit_c1_completed_waiting_risk import MAXR,KAPPA
from single_integrator.environment import Config,GiveWayEnv,bounded_nominal
from single_integrator.cbf import CBFConfig,barrier_constraints,project_velocity
from single_integrator.evaluate import load_policy

OUT=ROOT/'results/c1_candidate_coverage_generic'
BASE=ROOT/'results/c1_multistep_objective_audit/traces'
DT=.05;K=850;START=100;SHORT=300;LONG=500;SEED=20260916

def candidates():
    result=[dict(cid='zero',family='zero',stages=[],original_action=0)]
    directions=[]
    for i in range(4):
        for sign in [1,-1]:
            q=np.zeros(4);q[i]=sign
            directions.append((f'axis{i}_{sign}',q,'axis',1+i if sign==1 else 5+i))
    for i,j,s1,s2 in product([0,1],[2,3],[1,-1],[1,-1]):
        q=np.zeros(4);q[i]=s1;q[j]=s2;q/=np.linalg.norm(q)
        directions.append((f'joint{i}{j}_{s1}_{s2}',q,'joint',None))
    for name,q,family,original in directions:
        for amp,duration in product([.02,.06,.12],[2.,4.]):
            result.append(dict(cid=f'{name}_a{amp}_d{duration}',family=family,
                stages=[dict(seconds=duration,delta=(amp*q).tolist())],
                original_action=original if amp==.02 and duration==2 else None))
    # Only symmetric coordinate perturbations remain. Scene-oriented upward
    # pulses followed by hard-coded goal-x pulses were removed on user request.
    assert len(result)==145 and len({x['cid'] for x in result})==145
    return result

def delta_at(candidate,t):
    elapsed=(t-START)*DT
    if elapsed<0:return np.zeros(4)
    for stage in candidate['stages']:
        if elapsed<stage['seconds']-1e-9:return np.array(stage['delta'])
        elapsed-=stage['seconds']
    return np.zeros(4)

def metrics(z):
    T=len(z['success']);success=bool(np.any(z['success']));collision=bool(np.any(z['collision']))
    outcome=dict(success=success and not collision,deadlock=bool(np.any(z['deadlock'])),collision=collision,
        seconds=T*DT,stagnation_seconds=float(np.sum(z['candidate_deadlock'])*DT),
        future_stagnation_seconds=float(np.sum(z['candidate_deadlock'][SHORT:])*DT),
        remaining=float(np.linalg.norm(z['positions_after'][-1]-z['goals'],axis=1).max()),
        tail_stagnation_fraction=float(np.r_[z['candidate_deadlock'],np.zeros(K-T)][-100:].mean()))
    if collision or not np.all(np.isfinite(z['g'])):return None,outcome
    assert T==K or success
    pos=np.concatenate([z['positions_after'],np.repeat(z['positions_after'][-1][None],K-T,axis=0)])
    dist=np.linalg.norm(pos-z['goals'],axis=2)
    initial=np.linalg.norm(z['positions_before'][0]-z['goals'],axis=1).sum()+1e-5
    V=np.r_[initial-1e-5,dist.sum(1)]/initial
    Ut=1-np.prod(1-expit((dist-.08)/.02),axis=1);Ut[T:]=0
    end=np.arange(1,K+1);start=np.maximum(end-80,0)
    rate=(V[start]-V[end])/((end-start)*DT)
    P=Ut*expit((.005-rate)/.00125);P[T:]=0
    ds=dist[SHORT-1].sum()/initial;dl=dist[LONG-1].sum()/initial
    cl=float(expit((.005-(ds-dl)/10)/.00125))
    n=max(0,min(T,SHORT)-START);x=z['positions_before'][START:START+n];v=z['candidate'][START:START+n]
    d=np.linalg.norm(x-z['goals'],axis=2);U=1-np.prod(1-expit((d-.08)/.02),axis=1)
    cs=expit((.005-(d.sum(1)/initial-ds)/((SHORT-np.arange(START,START+n))*DT))/.00125)
    S=KAPPA**2/(KAPPA**2+np.sum((v/.5)**2,axis=1))
    two=U*S*(cs*cl+(1-cs)*cl**2);g=z['g']
    assert len(g)==n
    G=two+(1-two)*g
    costs=dict(P=float(P[START:SHORT].mean()),g=float(g.sum()/(SHORT-START)),
        S=float(S.sum()/(SHORT-START)),Stwo=float(two.sum()/(SHORT-START)),Gtwo=float(G.sum()/(SHORT-START)),Cl=cl)
    costs['score']=costs['P']+costs['Gtwo']
    return costs,outcome

def run(policy,rid,candidate):
    name=f"{rid:04d}_{candidate['cid']}";target=OUT/'traces'/f'{name}.npz';report=target.with_suffix('.json')
    if report.exists():return json.loads(report.read_text())
    prefix=np.load(BASE/f'{rid:04d}_{SEED}_0.npz')
    env=GiveWayEnv(replace(Config(corridor_half_length=1.3),terminate_on_deadlock=False));env.reset(prefix['positions_before'][0])
    cfg=CBFConfig();key=jax.random.fold_in(jax.random.PRNGKey(SEED),rid)
    buf={k:[] for k in ['positions_before','positions_after','candidate','applied','success','deadlock','candidate_deadlock','collision']};geom=[];geometry_errors=[]
    for t in range(K):
        x=env.positions.copy()
        if t<START:v=prefix['candidate'][t];u=prefix['applied'][t]
        else:
            raw=np.asarray(policy.sample_actions(jnp.asarray(env.observation()[None]),seed=jax.random.fold_in(key,t)))[0]
            A,b,_=barrier_constraints(env.snapshot(),cfg)
            safe=project_velocity(bounded_nominal(raw,.5),A,b,.5,cfg)[0].reshape(4)
            delta=delta_at(candidate,t);v=safe+delta
            u=project_velocity(v,A,b,.5,cfg)[0].reshape(4) if np.any(delta) else safe
            if t<SHORT:
                try:
                    risk=LocalRisk(A,b,np.full(2,.5)).score(v)
                    if not risk['valid']:raise ValueError('Undefined geometry, cannot silently rank')
                    geom.append(risk['risk']/MAXR)
                except (RuntimeError,ValueError) as error:
                    geometry_errors.append(dict(step=t,error=str(error)))
                    geom.append(np.nan)
        _,_,done,info=env.step(u.reshape(2,2))
        if t<START:np.testing.assert_array_equal(env.positions,prefix['positions_after'][t])
        row=dict(positions_before=x,positions_after=env.positions.copy(),candidate=v,applied=u,success=info['task_success'],
            deadlock=info['deadlock'],candidate_deadlock=info['candidate_deadlock'],collision=info['wall_collision'] or info['agent_collision'])
        for k in buf:buf[k].append(row[k])
        if done:break
    z={k:np.asarray(v) for k,v in buf.items()};z.update(g=np.asarray(geom),goals=env.goals)
    costs,outcome=metrics(z)
    row=dict(rid=rid,cid=candidate['cid'],family=candidate['family'],intervention=candidate,costs=costs,outcome=outcome,geometry_errors=geometry_errors)
    if candidate['original_action'] is not None:
        old=np.load(BASE/f"{rid:04d}_{SEED}_{candidate['original_action']}.npz")
        np.testing.assert_array_equal(z['positions_after'],old['positions_after'])
        oldrows=json.loads((ROOT/'results/c1_wait_horizon_sweep/rows.json').read_text())
        ref=next(r for r in oldrows if r['rid']==rid and r['action']==candidate['original_action'] and r['horizon']==25)
        for k in ['P','g','S','Stwo','Gtwo']:assert abs(costs[k]-ref['costs'][k])<1e-10
    np.savez_compressed(target,**z);save(report,row);return row

def summarize(rows):
    out={}
    for rid in [68,208]:
        rr=[r for r in rows if r['rid']==rid];eligible=[r for r in rr if r['costs'] is not None]
        ranked=sorted(eligible,key=lambda r:(r['costs']['score'],r['cid']));good=[r for r in ranked if r['outcome']['success']]
        if not ranked:continue
        best=good[0] if good else None
        out[rid]=dict(evaluated=len(rr),eligible=len(eligible),successful=sum(r['outcome']['success'] for r in rr),scorable_successful=len(good),collisions=sum(r['outcome']['collision'] for r in rr),
            unscorable=[dict(cid=r['cid'],outcome=r['outcome'],errors=r.get('geometry_errors',[])) for r in rr if r['costs'] is None],
            selected=ranked[0],best_success=best,best_success_rank=ranked.index(best)+1 if best else None,
            failed_ranked_above_best_success=ranked.index(best) if best else None,
            success_count_by_family={f:sum(r['outcome']['success'] for r in rr if r['family']==f) for f in ['zero','axis','joint']},
            successful_candidates=[dict(cid=r['cid'],score=r['costs']['score'],outcome=r['outcome']) for r in good],
            top10=[dict(cid=r['cid'],score=r['costs']['score'],outcome=r['outcome']) for r in ranked[:10]])
    return out

def main():
    OUT.mkdir(exist_ok=True);(OUT/'traces').mkdir(exist_ok=True);library=candidates()
    protocol=dict(ids=[68,208],seed=SEED,decision_seconds=5,score_window=[5,15],lookahead_endpoint=25,execution_endpoint=42.5,
        scorer='Frozen P+Gtwo, same two-horizon formula and parameters; scalar score sees no states beyond25.',
        library=library,total_per_initial=len(library),directions='8 signed axes +16 signed cross-agent coordinate pairs, all unit joint norm',
        amplitudes=[.02,.06,.12],durations=[2,4],excluded='Scene-specific two-stage pulses removed; historical results not overwritten',
        scope='Expanded actions only on remaining two failures. Fixed-noise closed-loop candidates, not retraining or scoring redesign.',
        safety='Each pulse added to freshly computed baseline safe control then original joint projection. Collision candidate excluded from ranking, reported.',
        outcomes='Original environment success and stagnation; finite-horizon labels not permanent infeasibility proof.')
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:save(OUT/'protocol.json',protocol)
    policy,_=load_policy(ROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl');rows=[];start=time.perf_counter()
    for candidate in library:
        for rid in [68,208]:
            r=run(policy,rid,candidate);rows.append(r)
        if len(rows)%20==0 or len(rows)==2 or len(rows)==2*len(library):
            summary=summarize(rows);save(OUT/'rows.json',rows);save(OUT/'summary.json',summary)
            print(len(rows),round(time.perf_counter()-start,1),{rid:dict(successful=s['successful'],selected_success=s['selected']['outcome']['success'],best_rank=s['best_success_rank']) for rid,s in summary.items()},flush=True)
    old=json.loads((ROOT/'results/c1_completed_waiting_risk_audit/verification.json').read_text())['sha256']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in old};assert hashes==old
    save(OUT/'verification.json',dict(original9_per_state_replayed_exactly=True,original_score_reproduced=True,
        core_hashes_unchanged=True,hashes=hashes,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        elapsed_seconds=time.perf_counter()-start,branches=len(rows)))

if __name__=='__main__':main()
