"""Prospective 64-state evaluation of the frozen 161-candidate protocol."""
import argparse
import copy
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from audit_c1_joint_witness_risk import ROOT,LocalRisk,save,SOLVER_AUDIT
from audit_c1_candidate_coverage import candidates,delta_at,metrics,DT,K,START,SHORT
from audit_c1_completed_waiting_risk import MAXR
from single_integrator.environment import Config,GiveWayEnv,bounded_nominal
from single_integrator.cbf import CBFConfig,barrier_constraints,project_velocity,CBFSolverError
from single_integrator.evaluate import load_policy
from c1_heldout_safety import check as safety_check

OUT=ROOT/'results/c1_frozen_unseen_64'
N=64;INITIAL_SEED=20260923;PREFIX_SEED=20260915;PLAN_SEED=20260916
FIELDS=['positions_before','positions_after','candidate','applied','success','deadlock','candidate_deadlock','collision']

def manifest():
    paths=['scripts/audit_c1_joint_witness_risk.py','scripts/audit_c1_candidate_coverage.py',
        'scripts/audit_c1_completed_waiting_risk.py','single_integrator/cbf.py','single_integrator/environment.py',
        'flowbc/giveway_flowbc_agent.py','scripts/evaluate_c1_frozen_unseen.py','scripts/c1_heldout_safety.py',
        'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl']
    return {p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}

def prepare():
    OUT.mkdir(exist_ok=True);(OUT/'states').mkdir(exist_ok=True);(OUT/'traces').mkdir(exist_ok=True)
    rng=np.random.default_rng(INITIAL_SEED);x=rng.uniform(.55,1.05,N);y=rng.uniform(-.025,.025,(N,2))
    initials=np.stack([np.c_[-x,y[:,0]],np.c_[x,y[:,1]]],axis=1)
    old=[]
    for f in (ROOT/'results/c1_large_risk_association_400/traces').glob('*.npz'):
        with np.load(f) as z:old.append(z['positions_before'][0])
    assert len(old)==400
    distances=np.max(np.abs(initials[:,None]-np.asarray(old)[None]),axis=(2,3));assert np.min(distances)>0
    library=candidates();old_library=json.loads((ROOT/'results/c1_candidate_coverage_expansion/protocol.json').read_text())['library'];assert library==old_library
    protocol=dict(n=N,initial_seed=INITIAL_SEED,initials=initials.tolist(),ids=list(range(100000,100000+N)),
        distribution='Same symmetric x_abs Uniform(.55,1.05), independent y_i Uniform(-.025,.025); no outcome filtering.',
        previously_unseen='New random draw after numerical validation; exact comparison against all prior400 states, min L-infinity difference recorded.',
        minimum_difference_from_prior400=float(distances.min()),prefix_seed=PREFIX_SEED,planning_and_execution_seed=PLAN_SEED,
        decision_seconds=5,score_window=[5,15],lookahead_endpoint=25,execution_endpoint=42.5,library=library,
        score='Frozen P+Gtwo from candidate-coverage protocol; no new coefficients or mathematical changes.',
        numerics='Base QP equilibration off; exact convex SLSQP fallback with KKT certificate; failures recorded, never filled with a risk value.',
        primary='All161 candidates on each of64 fresh states; success coverage and conditional ranking at initial-state level.',
        noise='Same fixed random input across candidate branches and prediction/execution within a state; distinct keys across states. Ideal prediction consistency, not noisy-model robustness.',
        failure_handling='Unscorable branches have execution outcomes but cannot be ranked; controller-error branches are incomplete and reported separately.',
        hashes=manifest())
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:save(OUT/'protocol.json',protocol)
    print('Frozen64 states and161 candidates; no exact reuse of prior400 states.',flush=True)

def values(x,v,u,env,info):
    return dict(positions_before=x,positions_after=env.positions.copy(),candidate=v,applied=u,
        success=info['task_success'],deadlock=info['deadlock'],candidate_deadlock=info['candidate_deadlock'],
        collision=info['wall_collision'] or info['agent_collision'])

def prefix(policy,initial,rid,cfg):
    env=GiveWayEnv(replace(Config(corridor_half_length=1.3),terminate_on_deadlock=False));env.reset(initial)
    buf={k:[] for k in FIELDS};key=jax.random.fold_in(jax.random.PRNGKey(PREFIX_SEED),rid)
    for t in range(START):
        x=env.positions.copy();raw=np.asarray(policy.sample_actions(jnp.asarray(env.observation()[None]),seed=jax.random.fold_in(key,t)))[0]
        A,b,_=barrier_constraints(env.snapshot(),cfg);u=project_velocity(bounded_nominal(raw,.5),A,b,.5,cfg)[0].reshape(4)
        _,_,done,info=env.step(u.reshape(2,2))
        for k,v in values(x,u,u,env,info).items():buf[k].append(v)
        if done:break
    return env,buf

def branch(policy,rid,candidate,prefix_env,prefix_buf,cfg):
    dest=OUT/'traces'/str(rid)/candidate['cid'];report=Path(str(dest)+'.json');trace=Path(str(dest)+'.npz')
    if report.exists():return json.loads(report.read_text())
    env=copy.deepcopy(prefix_env);buf={k:list(v) for k,v in prefix_buf.items()};g=[];errors=[];ambiguous=0
    counts=dict(SOLVER_AUDIT);key=jax.random.fold_in(jax.random.PRNGKey(PLAN_SEED),rid);controller_error=None
    for t in range(START,K) if len(buf['success'])==START and not buf['success'][-1] and not buf['collision'][-1] else []:
        x=env.positions.copy()
        raw=np.asarray(policy.sample_actions(jnp.asarray(env.observation()[None]),seed=jax.random.fold_in(key,t)))[0]
        A,b,_=barrier_constraints(env.snapshot(),cfg)
        try:
            safe=project_velocity(bounded_nominal(raw,.5),A,b,.5,cfg)[0].reshape(4)
            delta=delta_at(candidate,t);v=safe+delta
            u=project_velocity(v,A,b,.5,cfg)[0].reshape(4) if np.any(delta) else safe
        except CBFSolverError as e:controller_error=str(e);break
        if t<SHORT:
            try:
                risk=LocalRisk(A,b,np.full(2,.5)).score(v)
                if not risk['valid']:raise ValueError(str(risk))
                g.append(risk['risk']/MAXR);ambiguous+=int(risk['ambiguous_witnesses']>0)
            except (RuntimeError,ValueError) as e:errors.append(dict(step=t,error=str(e)));g.append(np.nan)
        _,_,done,info=env.step(u.reshape(2,2))
        for k,value in values(x,v,u,env,info).items():buf[k].append(value)
        if done:break
    z={k:np.asarray(v) for k,v in buf.items()};z.update(g=np.asarray(g),goals=env.goals)
    if controller_error:
        costs=None;outcome=dict(success=False,deadlock=bool(np.any(z['deadlock'])),collision=bool(np.any(z['collision'])),
            seconds=len(z['success'])*DT,stagnation_seconds=float(z['candidate_deadlock'].sum()*DT),
            future_stagnation_seconds=float(z['candidate_deadlock'][SHORT:].sum()*DT),remaining=float(np.linalg.norm(env.positions-env.goals,axis=1).max()),
            tail_stagnation_fraction=None)
    else:costs,outcome=metrics(z)
    safety=safety_check(z,env,cfg)
    row=dict(rid=rid,cid=candidate['cid'],family=candidate['family'],original_action=candidate['original_action'],
        costs=costs,outcome=outcome,safety=safety,geometry_errors=errors,controller_error=controller_error,
        ambiguous_geometry_frames=ambiguous,numerical_counts={k:SOLVER_AUDIT[k]-counts[k] for k in counts})
    np.savez_compressed(trace,**z);save(report,row);return row

def state_summary(rid,rows):
    eligible=sorted([r for r in rows if r['costs'] is not None],key=lambda r:(r['costs']['score'],r['cid']))
    good=[r for r in rows if r['outcome']['success']];scorable_good=[r for r in eligible if r['outcome']['success']]
    selected=eligible[0] if eligible else None
    old=sorted([r for r in eligible if r['original_action'] is not None],key=lambda r:(r['costs']['score'],r['cid']))
    return dict(rid=rid,candidates=len(rows),successful_candidates=len(good),scorable_successful_candidates=len(scorable_good),
        coverage=bool(good),coverage_unknown=not good and any(r['controller_error'] for r in rows),
        selected=selected,baseline=next(r for r in rows if r['cid']=='zero'),original9_selected=old[0] if old else None,
        original9_coverage=any(r['outcome']['success'] for r in rows if r['original_action'] is not None),
        best_success_rank=eligible.index(scorable_good[0])+1 if scorable_good else None,
        unscorable=sum(r['costs'] is None for r in rows),controller_errors=sum(r['controller_error'] is not None for r in rows),
        numerical_counts={k:sum(r['numerical_counts'][k] for r in rows) for k in SOLVER_AUDIT},
        safety_counts={k:sum(r['safety'][k] for r in rows) for k in ['steps','agent_collision_steps','wall_collision_steps','outside_endpoints','cbf_violations','speed_violations']},
        safety_minima={k:min(r['safety'][k] for r in rows) for k in ['min_center_separation','min_agent_surface_clearance','min_wall_surface_clearance','min_pair_cbf_residual','min_wall_cbf_residual','min_pair_h','min_wall_h']})

def worker(index,total):
    p=json.loads((OUT/'protocol.json').read_text());assert manifest()==p['hashes']
    policy,_=load_policy(ROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl');cfg=CBFConfig();started=time.perf_counter()
    for i in range(index,N,total):
        rid=p['ids'][i];done=OUT/'states'/f'{rid}.json'
        if done.exists():continue
        (OUT/'traces'/str(rid)).mkdir(exist_ok=True)
        env,buf=prefix(policy,np.asarray(p['initials'][i]),rid,cfg);rows=[]
        for j,candidate in enumerate(p['library']):
            rows.append(branch(policy,rid,candidate,env,buf,cfg))
            if (j+1)%40==0:print('progress',rid,j+1,round(time.perf_counter()-started,1),flush=True)
        summary=state_summary(rid,rows);save(done,summary)
        print('STATE',rid,'coverage',summary['successful_candidates'],'selected',summary['selected']['outcome'] if summary['selected'] else None,'numerics',summary['numerical_counts'],flush=True)
    assert manifest()==p['hashes']

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--worker',type=int);parser.add_argument('--workers',type=int,default=8);a=parser.parse_args()
    if a.prepare:prepare()
    else:worker(a.worker,a.workers)
