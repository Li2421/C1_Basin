"""One small, gated full C1 run: frozen Flow-BC, shared residual, paired test.

The exact risk is piecewise differentiable. No scene-specific escape rule,
candidate selection, risk-only warm start, or residual fit per test case.
"""
import argparse
import json
from pathlib import Path
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from single_integrator.c1.train_deadlock_union import setup, noise, digest, source_hashes
from single_integrator.c1.rollout_deadlock_union import rollout
from single_integrator.c1.risk.exact_margin import trajectory
from single_integrator.c1.evaluate_deadlock_union import execute
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.c1.training.step_control_restart import guarded_step, finite
from single_integrator.c1.training.primal_dual import PrimalDualState, dual_update
from single_integrator.c1.training.selection import better_feasible_candidate
from single_integrator.c1.acceptance import assess
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace
from single_integrator.environment import GiveWayEnv

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--probe',type=Path,required=True)
    parser.add_argument('--updates',type=int,default=64)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('new output required')
    probe=json.loads((args.probe/'complete.json').read_text())
    proto=json.loads((args.probe/'protocol.json').read_text())
    ordering=json.loads((args.probe/'ordering_audit.json').read_text())
    if not ordering['scores']['exact']['strict_ordering']:
        raise ValueError('candidate score ordering failed; do not train')
    if proto.get('candidate')!='exact' or probe['status']!='complete' or probe['replays']!=360:
        raise ValueError('complete exact-risk diagnostic required')
    a=probe['variants']['Safety'];b=probe['variants']['bounded_minus_0.005'];c=probe['variants']['bounded_plus_0.005']
    gate=(b['deadlocks']<=.7*a['deadlocks'] and b['deadlock_to_success']>=.4*a['deadlocks']
          and b['successes']>a['successes'] and b['deadlocks']<c['deadlocks'])
    if not gate:raise ValueError('local gradient gate failed; analyze before training')
    params,field,plant,cbf,baseline=setup(0,0)
    if jax.default_backend()!='gpu':raise RuntimeError('Slurm GPU required')
    original_params=params
    goals=jnp.asarray(GiveWayEnv(plant).goals)
    kw=dict(dt=plant.dt,max_speed=plant.max_speed,goal_tolerance=plant.goal_tolerance,
        hold_seconds=plant.deadlock_hold_seconds,progress_window_seconds=plant.progress_window_seconds,
        progress_epsilon=plant.progress_epsilon,speed_epsilon_fraction=plant.speed_epsilon_fraction)
    score=jax.jit(lambda b,a,u,m,to:trajectory(b,a,u,goals,m,terminal_timeout=to,**kw)['J_live'])
    @jax.jit
    def fn(phi,x,draws):
        terms,t=rollout(phi,field,x,draws,plant,cbf)
        r=score(t['before'],t['after'],t['applied'],t['alive'],terms['timeout'])
        return dict(J_live=r,J_def=terms['J_def'],deadlock=terms['deadlock'],
            either_deadlock=terms['either_deadlock'],success=terms['success'],timeout=terms['timeout'])
    oldsets=json.loads((ROOT/'results/c1_deadlock_union/sets.json').read_text())
    # Existing development pools remain development. New independent test only.
    rng_test=np.random.default_rng(20260917020)
    starts=np.zeros((100,2,2));starts[:,:,0]=rng_test.uniform(.55,1.05,(100,2))*np.array([-1,1])
    starts[:,:,1]=rng_test.uniform(-.025,.025,(100,2))
    prior=np.array([r['initial'] for group in oldsets['pools'].values() for r in group])
    if np.any(np.all(starts[:,None]==prior[None],axis=(2,3))):raise ValueError('test overlap')
    data=dict(train=oldsets['pools']['train'],validation=oldsets['pools']['validation'],
        test=[dict(rid=10000+i,initial=x.tolist()) for i,x in enumerate(starts)],
        validation_noise_seeds=[82201,82202],test_noise_seeds=[82301,82302])
    args.out.mkdir(parents=True)
    atomic_save(args.out/'sets.json',data)
    config=dict(risk='relu(1 + exact temporal union robustness)',epsilon=.01,lr=.0001,
        dual_lr=.1,initial_dual=1.,batch_size=8,updates=args.updates,validation_every=8,
        baseline_sha256=digest(baseline),environment=plant.to_dict(),seed=0,
        sets_sha256=digest(args.out/'sets.json'),probe_sha256=digest(args.probe/'complete.json'),
        source_hashes={**source_hashes(),**{p:digest(ROOT/p) for p in (
            'single_integrator/c1/train_exact_margin.py','single_integrator/c1/risk/exact_margin.py',
            'single_integrator/c1/evaluate_deadlock_union.py','single_integrator/c1/acceptance.py',
            'single_integrator/c1/paired_statistics.py','scripts/c1_heldout_safety.py')}},
        test_budget='100 starts x 2 noises x 2 policies = 400 replays',
        difficulty='Before training: unfiltered validation Safety >=10% deadlock and >=10 deadlock start clusters',
        selection='minimum validation J_def among J_live<=.01',
        local_gate_passed=gate,objective='min J_def subject to E R <= epsilon')
    atomic_save(args.out/'config.json',config)
    begun=time.monotonic()
    def measure(phi,pool,seeds):
        rows=[]
        for item in pool:
            for seed in seeds:
                v=fn(phi,jnp.asarray(item['initial']),noise(seed,item['rid']))
                if not finite(v):raise FloatingPointError('nonfinite rollout')
                rows.append(dict(rid=item['rid'],noise_seed=seed,**{k:float(x) for k,x in v.items()}))
        return dict(**{k:float(np.mean([r[k] for r in rows])) for k in v},episodes=rows)
    baseline_val=measure(params,data['validation'],data['validation_noise_seeds'])
    for r in baseline_val['episodes']:
        if bool(r['either_deadlock']) != (r['J_live']>1.):
            raise AssertionError('exact score and original detector disagree on development')
    difficult_clusters=len({r['rid'] for r in baseline_val['episodes'] if r['either_deadlock']})
    difficulty=baseline_val['either_deadlock']>=.1 and difficult_clusters>=10
    atomic_save(args.out/'difficulty.json',dict(passed=difficulty,deadlock_clusters=difficult_clusters,**baseline_val))
    if not difficulty:raise RuntimeError('Safety development difficulty gate failed; do not run easy comparison')
    eps=config['epsilon'];optimizer=optax.adam(config['lr']);state=optimizer.init(params)
    dual=PrimalDualState(1.);rng=np.random.default_rng(0);history=[];validations=[];best=None
    def validate(update,precomputed=None):
        nonlocal best
        row=measure(params,data['validation'],data['validation_noise_seeds']) if precomputed is None else precomputed
        row.update(update=update,epsilon=eps,constraint=row['J_live']-eps)
        validations.append(row);atomic_save(args.out/'validation.json',validations)
        if better_feasible_candidate(row,best):
            best=row
            atomic_save(args.out/'best_feasible.pkl',dict(params=jax.device_get(params),config=config,selection=row),binary=True)
        print(json.dumps(dict(validation={k:v for k,v in row.items() if k!='episodes'},elapsed=time.monotonic()-begun)),flush=True)
    validate(0,baseline_val)
    for step in range(args.updates):
        ids=rng.choice(len(data['train']),size=config['batch_size'],replace=False)
        items=[data['train'][int(i)] for i in ids];seed=int(rng.integers(1000000,2**31-1))
        draws=[noise(seed,i['rid']) for i in items]
        def objective(phi):
            terms=[fn(phi,jnp.asarray(i['initial']),n) for i,n in zip(items,draws)]
            live=jnp.mean(jnp.stack([v['J_live'] for v in terms]));dev=jnp.mean(jnp.stack([v['J_def'] for v in terms]))
            return dev+dual.dual*(live-eps),dict(J_live=live,J_def=dev)
        before,g=jax.value_and_grad(objective,has_aux=True)(params)
        used=dual.dual
        params,state,after,decision=guarded_step(params,state,g,optimizer,objective,before)
        dual=dual_update(dual,before[1]['J_live'],eps,config['dual_lr'])
        history.append(dict(update=step,rids=[i['rid'] for i in items],noise_seed=seed,lambda_used=used,
            lambda_after=dual.dual,gradient_norm=float(optax.global_norm(g)),decision=decision,
            pre={k:float(v) for k,v in before[1].items()},post={k:float(v) for k,v in after[1].items()}))
        atomic_save(args.out/'checkpoint.pkl',dict(params=jax.device_get(params),optimizer=jax.device_get(state),
            dual=dual.dual,rng_state=rng.bit_generator.state,completed_updates=step+1,history=history,config=config),binary=True)
        atomic_save(args.out/'history.json',history)
        print(json.dumps(dict(update=step+1,**history[-1]['post'],elapsed=time.monotonic()-begun)),flush=True)
        if (step+1)%8==0 or step+1==args.updates:validate(step+1)
    atomic_save(args.out/'training_complete.json',dict(updates=len(history),validation_feasible=best is not None,
        elapsed_seconds=time.monotonic()-begun))
    if best is None:
        atomic_save(args.out/'complete.json',dict(status='no_validation_feasible_checkpoint',test_opened=False))
        return
    if any(digest(ROOT/p)!=sha for p,sha in config['source_hashes'].items()):
        raise RuntimeError('source changed during training; refuse independent test')
    if digest(baseline)!=config['baseline_sha256'] or digest(args.out/'sets.json')!=config['sets_sha256']:
        raise RuntimeError('baseline or frozen sets changed; refuse independent test')
    selected=pickle.loads((args.out/'best_feasible.pkl').read_bytes())
    selected_params=jax.tree_util.tree_map(jnp.asarray,selected['params'])
    atomic_save(args.out/'test_protocol.json',dict(checkpoint_sha256=digest(args.out/'best_feasible.pkl'),
        selected_update=best['update'],episodes=400,bootstrap_family_size=1))
    records=[];evaluation_started=time.monotonic()
    for item in data['test']:
        for seed in data['test_noise_seeds']:
            for method,phi in [('Safety',original_params),('C1',selected_params)]:
                row,t=execute(phi,field,np.asarray(item['initial']),noise(seed,item['rid']),plant,cbf)
                outcome='safe_deadlock' if row['any_deadlock'] else ('success' if row['success'] else 'other_timeout')
                label,_=classify_timeout_trace(dict(max_speed=np.linalg.norm(t['applied'].reshape(-1,2,2),axis=-1).max(axis=1),
                    goal_errors=np.linalg.norm(t['positions_after']-np.asarray(goals),axis=-1)),outcome,plant.dt)
                n=row['steps'];pad=lambda x:np.concatenate([x,np.repeat(x[-1:],plant.max_steps-n,axis=0)])
                risk=float(score(pad(t['positions_before']),pad(t['positions_after']),
                    np.pad(t['applied'],((0,plant.max_steps-n),(0,0))),np.arange(plant.max_steps)<n,row['timeout']))
                records.append(dict(rid=item['rid'],noise_seed=seed,method=method,six_class_outcome=label,Rrisk=risk,**row))
                np.savez_compressed(args.out/f"test_{method}_{item['rid']}_{seed}.npz",**t)
        atomic_save(args.out/'test_records.json',records)
    result=assess(records,[r['rid'] for r in data['test']],data['test_noise_seeds'],family_size=1)
    dead=lambda r:r['any_deadlock'] or r['six_class_outcome']=='stalled_deadlock'
    dr=[r['Rrisk'] for r in records if dead(r)];nr=[r['Rrisk'] for r in records if not dead(r)]
    ordering=dict(min_deadlock=min(dr) if dr else None,max_non_deadlock=max(nr) if nr else None,
        strict_ordering=bool(dr and nr and min(dr)>max(nr)))
    result['risk_ordering']=ordering
    result['gates']['all_non_deadlock_scores_below_all_deadlock_scores']=ordering['strict_ordering']
    result['gates']['independent_Safety_deadlock_rate_at_least_10_percent']=result['statistics']['safety_deadlocks']/200>=.1
    result['all_endpoint_gates_passed']=all(result['gates'].values())
    atomic_save(args.out/'acceptance.json',result)
    atomic_save(args.out/'complete.json',dict(status='complete',test_opened=True,episodes=len(records),
        all_endpoint_gates_passed=result['all_endpoint_gates_passed'],
        evaluation_seconds=time.monotonic()-evaluation_started,elapsed_seconds=time.monotonic()-begun))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
