"""Isolated causal probe of local-risk direction descent, never a training module."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse
import hashlib
import json
import time
from dataclasses import replace
import numpy as np
from audit_c1_joint_witness_risk import ROOT, LocalRisk, save, trajectory_values
from single_integrator.cbf import CBFConfig, barrier_constraints, project_velocity
from single_integrator.environment import Config, GiveWayEnv, bounded_nominal
from single_integrator.evaluate import load_policy
from single_integrator.outcomes import first_event
import jax
import jax.numpy as jnp

OUT=ROOT/'results/c1_joint_closed_loop_audit'
OLD=ROOT/'results/c1_joint_witness_risk_audit'
META=ROOT/'results/c1_dataset_paired_seed0_baselines/config.json'
ARMS={'baseline':0.,'descent_002':.02,'descent_010':.1,'random_010':.1}

def choose_ids():
    records=json.loads((OLD/'index.json').read_text())
    records=[r for r in records if r['source']=='primary' and r['split']=='development']
    ids=[]
    for label,n in [('safe_deadlock',99),('success',3),('other_timeout',3)]:
        ids.extend([r['rid'] for r in records if r['outcome']==label][:n])
    return sorted(ids)

def direction_step(model,v,arm,rng):
    initial=model.score(v);a=np.linalg.norm(v)
    if not initial['valid']:return v,initial,initial,False
    q=v/a;theta=ARMS.get(arm,0.)
    if not theta:return v,initial,initial,False
    if arm.startswith('random'):
        g=rng.standard_normal(len(q));g-=q*np.dot(q,g)
    else:
        h=3e-5
        g=np.array([(model.score(a*(q+h*e))['risk']-model.score(a*(q-h*e))['risk'])/(2*h) for e in np.eye(len(q))])
        g-=q*np.dot(q,g)
    ng=np.linalg.norm(g)
    if ng<1e-10:return v,initial,initial,False
    g/=ng
    # A fixed maximum angular perturbation, with monotone backtracking for descent.
    for scale in ([1.] if arm.startswith('random') else [1.,.5,.25,.125]):
        qn=np.cos(theta*scale)*q-np.sin(theta*scale)*g
        vn=a*qn;after=model.score(vn)
        if arm.startswith('random') or after['risk']<initial['risk']-1e-10:
            return vn,initial,after,True
    return v,initial,initial,False

def rollout(policy,meta,rid,arm,require_historical_length=True):
    delayed=arm in ['wait_2s','loop_2s'];delay=40 if delayed else 0
    plant=Config(**meta['environment']);plant=replace(plant,max_steps=plant.max_steps+delay)
    cbf=CBFConfig(**meta['cbf']);env=GiveWayEnv(plant)
    initial=np.asarray(meta['initial_positions'][rid]);env.reset(initial)
    key=jax.random.fold_in(jax.random.PRNGKey(meta['seed']),rid)
    rng=np.random.default_rng(np.random.SeedSequence([20260914,rid]))
    buf={k:[] for k in ['positions_before','positions_after','candidate','applied','risk','risk_lower','B','E','valid','risk_before',
        'changed','deadlock','success','window_progress','stuck_timer','candidate_deadlock','goal_errors']}
    for t in range(plant.max_steps):
        x=env.positions.copy();A,b,_=barrier_constraints(env.snapshot(),cbf)
        if t<delay:
            v=np.zeros(4)
            if arm=='loop_2s' and t<36:
                # Small closed square, both agents move together; final .2 s at rest.
                d=[(1,0),(0,1),(-1,0),(0,-1)][t//9]
                v=np.tile(np.asarray(d)*.01,2)
            safe,_=project_velocity(v,A,b,plant.max_speed,cbf);safe=safe.reshape(-1)
        else:
            raw=np.asarray(policy.sample_actions(jnp.asarray(env.observation()[None]),seed=jax.random.fold_in(key,t-delay)))[0]
            nominal=bounded_nominal(raw,plant.max_speed)
            safe,_=project_velocity(nominal,A,b,plant.max_speed,cbf);safe=safe.reshape(-1)
        model=LocalRisk(A,b,np.full(2,plant.max_speed))
        v,before,after,changed=direction_step(model,safe,arm,rng)
        # Baseline's already-feasible safe control is mathematically its final projection.
        applied=project_velocity(v,A,b,plant.max_speed,cbf)[0].reshape(-1) if changed else safe
        _,_,done,info=env.step(applied.reshape(2,2))
        data=dict(positions_before=x,positions_after=env.positions.copy(),candidate=v,applied=applied,
            risk=after.get('risk',np.nan),risk_lower=after.get('risk_lower',np.nan),B=after.get('B',np.nan),E=after.get('E',np.nan),
            valid=after['valid'],risk_before=before.get('risk',np.nan),changed=changed,
            deadlock=info['deadlock'],success=info['task_success'],window_progress=info['window_progress'],
            stuck_timer=info['stuck_timer'],candidate_deadlock=info['candidate_deadlock'],goal_errors=info['goal_errors'])
        for k in buf:buf[k].append(data[k])
        if done:break
    z={k:np.asarray(v) for k,v in buf.items()};T=len(z['valid'])
    z.update(idx=np.arange(T),dt=np.array(plant.dt),goals=env.goals,old_risk=np.full(T,np.nan))
    np.savez_compressed(OUT/'traces'/f'{rid:04d}_{arm}.npz',**z)
    score=trajectory_values(z,T)
    row=dict(rid=rid,arm=arm,outcome=first_event(info),steps=T,seconds=T*plant.dt,delay_seconds=delay*plant.dt,
        final_remaining_distance=float(np.max(info['goal_errors'])),candidate_deadlock_seconds=float(z['candidate_deadlock'].sum()*plant.dt),
        max_stuck_timer=float(z['stuck_timer'].max()),interventions=int(z['changed'].sum()),
        mean_instant_risk_change=float(np.nanmean(z['risk']-z['risk_before'])),**score)
    if arm=='baseline':
        source=np.load(ROOT/f'results/c1_dataset_paired_seed0_baselines/mac_cbf/rollout_{rid:04d}.npz')
        row['historical_same_length']=T==len(source['positions'])
        common=min(T,len(source['positions']))
        row['historical_max_position_error']=float(np.max(np.abs(z['positions_after'][:common]-source['positions'][:common])))
        if (require_historical_length and not row['historical_same_length']) or row['historical_max_position_error']>1e-7:
            raise AssertionError(f'Baseline reproduction failed {row}')
    return row

def summarize(rows):
    result={}
    for arm in sorted({r['arm'] for r in rows}):
        rr=[r for r in rows if r['arm']==arm]
        result[arm]=dict(n=len(rr),outcomes={s:sum(r['outcome']==s for r in rr) for s in sorted({r['outcome'] for r in rr})},
            mean_exposure=float(np.mean([r['exposure'] for r in rr])),mean_remaining=float(np.mean([r['remaining_distance'] for r in rr])),
            mean_candidate_deadlock_seconds=float(np.mean([r['candidate_deadlock_seconds'] for r in rr])))
    pairs=[]
    for r in rows:
        if r['arm']=='baseline':continue
        base=next(b for b in rows if b['rid']==r['rid'] and b['arm']=='baseline')
        pairs.append(dict(rid=r['rid'],arm=r['arm'],before=base['outcome'],after=r['outcome'],
            exposure_change=r['exposure']-base['exposure'],remaining_change=r['remaining_distance']-base['remaining_distance'],
            stall_seconds_change=r['candidate_deadlock_seconds']-base['candidate_deadlock_seconds']))
    save(OUT/'summary.json',dict(arms=result,pairs=pairs));print(json.dumps(result,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--analyze',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);(OUT/'traces').mkdir(exist_ok=True)
    rows=json.loads((OUT/'episodes.json').read_text()) if (OUT/'episodes.json').exists() else []
    if args.analyze:summarize(rows);return
    meta=json.loads(META.read_text());ids=choose_ids()
    protocol=dict(ids=ids,selection='All six development deadlocks, first three development successes and timeouts; exploratory enriched sample.',
        arms=ARMS,fd_step=3e-5,backtracking=[1.,.5,.25,.125],seed=meta['seed'],random_direction_seed=20260914,
        intervention='One fixed-magnitude joint direction step per closed-loop step, post-Safety; authoritative final joint projection.',
        delayed_controls='First three selected historical successes, add 2s safe waiting/square loop, shift same policy noise stream and horizon by 40 steps.',
        checkpoint_sha256=hashlib.sha256((ROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl').read_bytes()).hexdigest(),
        metadata_sha256=hashlib.sha256(META.read_bytes()).hexdigest())
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:save(OUT/'protocol.json',protocol)
    policy,_=load_policy(meta['checkpoint']);start=time.perf_counter()
    success_ids=[r['rid'] for r in json.loads((OLD/'index.json').read_text()) if r['source']=='primary' and r['split']=='development' and r['outcome']=='success'][:3]
    for rid in ids:
        for arm in list(ARMS)+(['wait_2s','loop_2s'] if rid in success_ids else []):
            if any(r['rid']==rid and r['arm']==arm for r in rows):continue
            row=rollout(policy,meta,rid,arm);rows.append(row);save(OUT/'episodes.json',rows)
            print(f"ID {rid} {arm}: {row['outcome']}, {row['seconds']:.2f}s, exposure={row['exposure']:.4f}; elapsed {time.perf_counter()-start:.1f}s",flush=True)
    summarize(rows)

if __name__=='__main__':main()
