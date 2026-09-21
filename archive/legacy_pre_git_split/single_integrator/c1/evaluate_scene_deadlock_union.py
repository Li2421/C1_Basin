"""Paired direct-policy V3/Safety execution; no candidate search or test tuning."""
import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.train_scene_deadlock_union import setup, noise, digest
from single_integrator.evaluate import ROOT
from single_integrator.diagnostics.stalled_outcomes import classify_timeout_trace, OUTCOMES
from single_integrator.c1.risk.deadlock_union import trajectory as risk_trajectory
from dataclasses import replace
import hashlib
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.cbf import barrier_constraints, project_velocity
from single_integrator.environment import GiveWayEnv
from scripts.c1_heldout_safety import check


def execute(params, field, initial, draws, plant, cbf):
    env = GiveWayEnv(plant)
    env.reset(np.asarray(initial))
    buffers = {k:[] for k in ('positions_before','positions_after','applied',
                             'candidate','safe','success','deadlock','candidate_deadlock')}
    for t in range(plant.max_steps):
        before = env.positions.copy()
        A,b,_ = barrier_constraints(env.snapshot(),cbf)
        def project(v,A,b,s):
            return jnp.asarray(project_velocity(np.asarray(v)[0],A,b,s,cbf)[0].reshape(1,4))
        ctrl = field.prepare(params,jnp.asarray(env.observation()[None]),draws[t:t+1],
                             A,b,plant.max_speed,project)
        candidate = np.asarray(ctrl['safe'][0]+ctrl['correction'][0])
        u = project_velocity(candidate,A,b,plant.max_speed,cbf)[0].reshape(4)
        _,_,done,info = env.step(u.reshape(2,2))
        if info['wall_collision'] or info['agent_collision']:
            raise RuntimeError('V3 safety failure; do not publish a complete evaluation')
        row = dict(positions_before=before,positions_after=env.positions.copy(),
            applied=u,candidate=candidate,safe=np.asarray(ctrl['safe'][0]),success=info['task_success'],deadlock=info['deadlock'],
            candidate_deadlock=info['candidate_deadlock'])
        for k in buffers:
            buffers[k].append(row[k])
        if done:
            break
    trace = {k:np.asarray(v) for k,v in buffers.items()}
    success = bool(trace['success'].any())
    deadlock = bool(trace['deadlock'].any())
    safety = check(trace,env,cbf)
    if any(safety[k] for k in ('agent_collision_steps','wall_collision_steps','outside_endpoints',
                               'cbf_violations','speed_violations')):
        raise RuntimeError('V3 safety audit failed')
    return dict(success=success,deadlock=(deadlock and not success),any_deadlock=deadlock,
        recovered_deadlock=(deadlock and success),first_event_success=(success and not deadlock),timeout=(not success and not deadlock),
        stagnation_seconds=float(trace['candidate_deadlock'].sum()*plant.dt),
        steps=len(trace['success']),J_def=float(np.mean(np.sum((trace['applied']-trace['safe'])**2,axis=-1))),safety=safety), trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--sets',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=('validation','test'),default='validation')
    p.add_argument('--validation-noise-seeds',type=int,nargs='+')
    args = p.parse_args()
    if args.validation_noise_seeds and args.split != 'validation':
        p.error('noise overrides are development-validation only')
    if args.out.exists():
        raise FileExistsError('paired evaluation requires a new output directory')
    checkpoint_bytes = args.checkpoint.read_bytes()
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    saved = pickle.loads(checkpoint_bytes)
    if args.split != 'validation' and ('selection' not in saved or saved['config']['objective'] != 'constrained'):
        raise ValueError('Final test requires a constraint-feasible selected checkpoint')
    config = saved['config']
    if config['version']!='c1_scene_deadlock_union_v1' or config['sets_sha256']!=digest(args.sets):
        raise ValueError('V3 checkpoint/set mismatch')
    if 'selection' in saved and (saved['selection']['J_live'] > saved['selection']['epsilon']):
        raise ValueError('selected checkpoint is not feasible')
    if any(digest(ROOT/path)!=expected for path,expected in config['source_hashes'].items()):
        raise ValueError('V3 source changed since training')
    if digest(Path(config['frozen_baseline'])) != config['frozen_manifest_sha256']:
        raise ValueError('Frozen matched manifest changed')
    baseline,field,plant,cbf,path = setup(config['seed'],config['baseline_seed'],Path(config['frozen_baseline']))
    if digest(path)!=config['baseline_sha256']:
        raise ValueError('baseline mismatch')
    data = json.loads(args.sets.read_text())
    params = jax.tree_util.tree_map(jnp.asarray,saved['params'])
    args.out.mkdir(parents=True)
    if args.split == 'legacy':
        suite = ROOT/'baseline_309_314/planning/wide_initial_states_200.npz'
        if digest(suite) != '30a575df16d56b65cb92b97a95fbcad8b454f49621de45a4359e6bbf947090cb':
            raise ValueError('original benchmark suite changed')
        with np.load(suite) as z:
            pool = [dict(rid=i, initial=x.tolist()) for i,x in enumerate(z['test_initial_positions'])]
        seeds = [42]
        plant = replace(plant, terminate_on_deadlock=True)
    else:
        pool, seeds = data['pools'][args.split], data[args.split+'_noise_seeds']
    if args.validation_noise_seeds:
        seeds=args.validation_noise_seeds
    goals = jnp.asarray(GiveWayEnv(plant).goals)
    score = jax.jit(lambda before,after,u,alive,timeout:
        risk_trajectory(before,after,u,goals,alive, terminal_timeout=timeout, dt=plant.dt, max_speed=plant.max_speed,
            goal_tolerance=plant.goal_tolerance, hold_seconds=plant.deadlock_hold_seconds,
            progress_window_seconds=plant.progress_window_seconds, progress_epsilon=plant.progress_epsilon,
            speed_epsilon_fraction=plant.speed_epsilon_fraction))
    records = []
    for item in pool:
        for seed in seeds:
            draws = noise(seed,item['rid'])
            for name,phi in [('Safety',baseline),('C1',params)]:
                row,trace = execute(phi,field,item['initial'],draws,plant,cbf)
                outcome = 'safe_deadlock' if row['any_deadlock'] else ('success' if row['success'] else 'other_timeout')
                label, _ = classify_timeout_trace(dict(
                    max_speed=np.max(np.linalg.norm(trace['applied'].reshape(-1,2,2),axis=-1),axis=-1),
                    goal_errors=np.linalg.norm(trace['positions_after']-np.asarray(goals),axis=-1)), outcome,plant.dt)
                row.update(method=name,rid=item['rid'],noise_seed=seed,strict_outcome=outcome,six_class_outcome=label)
                if args.split != 'legacy':
                    n=len(trace['success']); pad=850-n
                    before=np.concatenate([trace['positions_before'],np.repeat(trace['positions_after'][-1][None],pad,axis=0)])
                    after=np.concatenate([trace['positions_after'],np.repeat(trace['positions_after'][-1][None],pad,axis=0)])
                    u=np.concatenate([trace['applied'],np.zeros((pad,4))])
                    metrics=score(jnp.asarray(before),jnp.asarray(after),jnp.asarray(u),jnp.arange(850)<n,jnp.asarray(row['timeout']))
                    row['risk']={k:float(metrics[k]) for k in ['J_live','hard_risk','deadlock_bound','strict_bound','stalled_bound']}
                records.append(row)
                stem = f'{name}_{item["rid"]}_{seed}'
                np.savez_compressed(args.out/(stem+'.npz'),**trace)
                atomic_save(args.out/(stem+'.json'),row)
                print({k:v for k,v in row.items() if k!='safety'},flush=True)
    aggregate = {}
    for name in ('Safety','C1'):
        rows = [r for r in records if r['method']==name]
        aggregate[name] = dict(n=len(rows),**{k:sum(r[k] for r in rows)
            for k in ('success','deadlock','any_deadlock','timeout','recovered_deadlock','first_event_success')},
            mean_stagnation_seconds=float(np.mean([r['stagnation_seconds'] for r in rows])),
            J_def=float(np.mean([r['J_def'] for r in rows])),
            six_class_counts={label:sum(r['six_class_outcome']==label for r in rows) for label in OUTCOMES},
            risk_mean={k:float(np.mean([r['risk'][k] for r in rows])) for k in ['J_live','hard_risk','deadlock_bound','strict_bound','stalled_bound']} if args.split!='legacy' else None)
    paired = {(r['rid'],r['noise_seed']):{} for r in records}
    for r in records:
        paired[(r['rid'],r['noise_seed'])][r['method']] = r
    delta = dict(new_success=0,lost_success=0,resolved_deadlock=0,new_deadlock=0)
    for pair in paired.values():
        a,b = pair['Safety'],pair['C1']
        delta['new_success'] += int(b['success'] and not a['success'])
        delta['lost_success'] += int(a['success'] and not b['success'])
        delta['resolved_deadlock'] += int(a['deadlock'] and b['success'])
        delta['new_deadlock'] += int(b['deadlock'] and not a['deadlock'])
    atomic_save(args.out/'summary.json',dict(aggregate=aggregate,paired=delta,
        split=args.split,checkpoint_sha256=checkpoint_sha,
        protocol='C1 deadlock-primary certificate; step0 residual; original first-event for every split',
        evaluator_sha256=digest(Path(__file__)), baseline_seed=config['baseline_seed'],
        sets_sha256=digest(args.sets), execution_noise_seeds=seeds,
        selected_feasible='selection' in saved,objective=config['objective']))
    atomic_save(args.out/'complete.json',dict(episodes=len(records)))


if __name__=='__main__':
    main()
