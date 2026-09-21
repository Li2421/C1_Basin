"""Matched full-horizon objective ablation through small closed-loop interventions.

No controller/R_geom changes. One intervention is selected using an independent
planning noise stream; it is then executed with the original evaluation noise.
"""
import json, hashlib, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from audit_c1_joint_witness_risk import ROOT, LocalRisk, save
from audit_c1_completed_waiting_risk import series, scalar
from single_integrator.environment import Config, GiveWayEnv, bounded_nominal
from single_integrator.cbf import CBFConfig, barrier_constraints, project_velocity
from single_integrator.evaluate import load_policy

OUT=ROOT/'results/c1_multistep_objective_audit'
SOURCE=ROOT/'results/c1_large_risk_association_400'
PLAN_SEED=20260916;EXEC_SEED=20260915;DT=.05;K=850;START=100;DURATION=40
OBJECTIVES={'P':[1,0,0,0],'GP':[1,1,0,0],'GPF':[1,1,1,0],
            'PF':[1,0,1,0],'PS':[1,0,0,1]}
ACTIONS=np.vstack([np.zeros(4),.02*np.eye(4),-.02*np.eye(4)])

def cohort():
    rr=json.loads((SOURCE/'rows.json').read_text());rng=np.random.default_rng(20260917);ids=[]
    for group in ['success','deadlock','timeout']:
        candidates=[r['rid'] for r in rr if (not r['unresolved'] if group=='success' else r['deadlock_event'] if group=='deadlock' else r['unresolved'] and not r['deadlock_event'])]
        ids.extend(rng.choice(candidates,4,replace=False).tolist())
    return sorted(ids)

def run(policy,rid,seed,action):
    target=OUT/'traces'/f'{rid:04d}_{seed}_{action}.npz'
    report=target.with_suffix('.json')
    if report.exists():
        cached=json.loads(report.read_text())
        with np.load(target) as z:
            cached['remaining']=float(np.linalg.norm(z['positions_after'][-1]-z['goals'],axis=1).max())
        return cached
    source=np.load(SOURCE/'traces'/f'{rid:04d}.npz')
    env=GiveWayEnv(replace(Config(corridor_half_length=1.3),terminate_on_deadlock=False))
    env.reset(source['positions_before'][0]);cbf=CBFConfig();key=jax.random.fold_in(jax.random.PRNGKey(seed),rid)
    buf={k:[] for k in ['positions_before','positions_after','candidate','applied','success','deadlock','candidate_deadlock','stuck_timer']}
    for t in range(K):
        x=env.positions.copy()
        if t<START:
            v=source['candidate'][t];u=source['applied'][t]
        else:
            raw=np.asarray(policy.sample_actions(jnp.asarray(env.observation()[None]),seed=jax.random.fold_in(key,t)))[0]
            A,b,_=barrier_constraints(env.snapshot(),cbf)
            safe=project_velocity(bounded_nominal(raw,.5),A,b,.5,cbf)[0].reshape(4)
            delta=ACTIONS[action] if t<START+DURATION else np.zeros(4)
            v=safe+delta
            u=project_velocity(v,A,b,.5,cbf)[0].reshape(4) if np.any(delta) else safe
        _,_,done,info=env.step(u.reshape(2,2))
        if info['wall_collision'] or info['agent_collision']:raise RuntimeError(f'Unexpected collision: {rid} {action} {t}')
        if t<START or (seed==EXEC_SEED and action==0):
            np.testing.assert_array_equal(env.positions,source['positions_after'][t])
        vals=dict(positions_before=x,positions_after=env.positions.copy(),candidate=v,applied=u,
                  success=info['task_success'],deadlock=info['deadlock'],candidate_deadlock=info['candidate_deadlock'],stuck_timer=info['stuck_timer'])
        for k in buf:buf[k].append(vals[k])
        if done:break
    z={k:np.asarray(v) for k,v in buf.items()};T=len(z['success']);idx=np.unique(np.r_[np.arange(0,T,10),T-1]);risk=[]
    for t in idx:
        A,b,_=barrier_constraints(dict(positions=z['positions_before'][t],walls=env.walls,config=env.config.to_dict()),cbf)
        value=LocalRisk(A,b,np.full(2,.5)).score(z['candidate'][t])
        if not value['valid'] and np.linalg.norm(z['candidate'][t])>0:raise RuntimeError('Unresolved nonzero geometric risk')
        risk.append(value.get('risk',np.nan))
    z.update(idx=idx,risk=np.array(risk),goals=env.goals,dt=np.array(DT))
    arrays,_,_,success=series(z,10);cost=scalar(arrays,K)
    excess=max(float(np.linalg.norm(env.positions-env.goals,axis=1).max())-.08,0)
    F=excess/(excess+.1) # existing smooth terminal failure proxy; never execution labels
    tail=np.r_[z['candidate_deadlock'],np.zeros(K-T,bool)][-100:]
    row=dict(rid=rid,seed=seed,action=action,success=success,ever_deadlock=bool(z['deadlock'].any()),
             seconds=T*DT,stagnation_seconds=float(z['candidate_deadlock'].sum()*DT),
             tail_stagnation_fraction=float(tail.mean()),max_stuck_timer=float(z['stuck_timer'].max()),
             P=cost['progress'],G=cost['geometry'],S=cost['stop_only'],F=F,remaining=float(np.linalg.norm(env.positions-env.goals,axis=1).max()))
    row['objectives']={name:float(np.dot(weights,[row['P'],row['G'],row['F'],row['S']])) for name,weights in OBJECTIVES.items()}
    np.savez_compressed(target,**z);save(report,row)
    return row

def main():
    OUT.mkdir(exist_ok=True);(OUT/'traces').mkdir(exist_ok=True)
    ids=cohort();protocol=dict(ids=ids,cohort='4 random IDs from each original success/deadlock/other-timeout stratum, selection seed 20260917; enriched exploratory test.',
        horizon_seconds=42.5,decision_seconds=5.,intervention_seconds=2.,actions=ACTIONS.tolist(),
        objectives=OBJECTIVES,geometry='Frozen previous smooth completion of joint B+eta E; original constants and witness formula unchanged.',
        progress='Frozen actual-window normalized joint progress, computed every .05 seconds.',
        terminal='Existing max remaining-distance excess/(excess+.1), zero at joint goal completion; continuous failure proxy.',
        planning_seed=PLAN_SEED,execution_seed=EXEC_SEED,
        planning='Nine full remaining-horizon closed-loop branches; same planning noise across candidates; different noise from execution. Select once, execute 2s additive candidate perturbation.',
        sampling_seconds=.5,success='First joint success absorbs remaining task costs; deadlock does not truncate.',
        controls='PF controls terminal attribution; PS controls low-speed attribution. Progress coefficient unchanged across arms.')
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text())==protocol
    else:save(OUT/'protocol.json',protocol)
    policy,_=load_policy(ROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl');rows=[];choices=[];started=time.perf_counter()
    for rid in ids:
        branches=[run(policy,rid,PLAN_SEED,j) for j in range(len(ACTIONS))]
        baseline=run(policy,rid,EXEC_SEED,0);rows.append(dict(arm='baseline',**baseline))
        for arm in OBJECTIVES:
            selected=min(branches,key=lambda r:r['objectives'][arm])
            executed=run(policy,rid,EXEC_SEED,selected['action'])
            rows.append(dict(arm=arm,**executed))
            choices.append(dict(rid=rid,arm=arm,action=selected['action'],
                predicted_objective_change=selected['objectives'][arm]-branches[0]['objectives'][arm],
                realized_objective_change=executed['objectives'][arm]-baseline['objectives'][arm]))
        save(OUT/'rows.json',rows);save(OUT/'choices.json',choices)
        print(f'ID {rid} done ({time.perf_counter()-started:.1f}s): '+str({r['arm']:(r['action'],r['success']) for r in rows if r['rid']==rid}),flush=True)
    summary={}
    for arm in ['baseline']+list(OBJECTIVES):
        rr=[r for r in rows if r['arm']==arm]
        summary[arm]=dict(n=len(rr),successes=sum(r['success'] for r in rr),deadlocks=sum(r['ever_deadlock'] for r in rr),
            mean_stagnation_seconds=float(np.mean([r['stagnation_seconds'] for r in rr])),
            mean_tail_stagnation_fraction=float(np.mean([r['tail_stagnation_fraction'] for r in rr])),
            mean_remaining=float(np.mean([r['remaining'] for r in rr])))
    pairs={}
    rng=np.random.default_rng(20260918)
    for a,b in [('GP','P'),('GPF','GP'),('GPF','PF'),('GP','PS')]:
        aa=sorted([r for r in rows if r['arm']==a],key=lambda r:r['rid']);bb=sorted([r for r in rows if r['arm']==b],key=lambda r:r['rid'])
        ds=np.array([int(x['success'])-int(y['success']) for x,y in zip(aa,bb)])
        dt=np.array([x['stagnation_seconds']-y['stagnation_seconds'] for x,y in zip(aa,bb)])
        ix=rng.integers(0,len(ds),(2000,len(ds)))
        pairs[f'{a}_minus_{b}']=dict(success_gains=int(np.sum(ds>0)),success_losses=int(np.sum(ds<0)),
            success_rate_change=float(ds.mean()),success_rate_change_bootstrap95=np.quantile(ds[ix].mean(1),[.025,.975]) if np.any(ds) else None,
            stagnation_seconds_change=float(dt.mean()),stagnation_change_bootstrap95=np.quantile(dt[ix].mean(1),[.025,.975]),
            different_choices=sum(x['action']!=y['action'] for x,y in zip(aa,bb)),
            bootstrap_note='No success interval when all paired differences are zero; empirical degeneracy is not proof of population equivalence.')
    save(OUT/'summary.json',dict(arms=summary,paired=pairs));print(json.dumps(summary,indent=2))
    save(OUT/'manifest.json',dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),finished_ids=ids))

if __name__=='__main__':main()
