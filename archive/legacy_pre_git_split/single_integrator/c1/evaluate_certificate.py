"""Paired direct-policy V3/Safety execution; no candidate search or test tuning."""
import argparse
import json
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.train_certificate import setup, noise, digest, source_hashes
from single_integrator.c1.training.persistence import atomic_save
from single_integrator.cbf import barrier_constraints, project_velocity
from single_integrator.environment import GiveWayEnv
from scripts.c1_heldout_safety import check


def execute(params, field, initial, draws, plant, cbf):
    env = GiveWayEnv(plant)
    env.reset(np.asarray(initial))
    buffers = {k:[] for k in ('positions_before','positions_after','applied',
                             'candidate','success','deadlock','candidate_deadlock')}
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
            applied=u,candidate=candidate,success=info['task_success'],deadlock=info['deadlock'],
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
        steps=len(trace['success']),safety=safety), trace


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--sets',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--split',choices=('validation','test'),default='test')
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('paired evaluation requires a new output directory')
    saved = pickle.loads(args.checkpoint.read_bytes())
    if args.split == 'test' and 'selection' not in saved:
        raise ValueError('Final test requires a constraint-feasible selected checkpoint')
    config = saved['config']
    if config['version']!='c1_temporal_certificate_v1' or config['sets_sha256']!=digest(args.sets):
        raise ValueError('V3 checkpoint/set mismatch')
    if config['source_hashes']!=source_hashes():
        raise ValueError('V3 source changed since training')
    baseline,field,plant,cbf,path = setup(config['seed'],config['baseline_seed'])
    if digest(path)!=config['baseline_sha256']:
        raise ValueError('baseline mismatch')
    data = json.loads(args.sets.read_text())
    params = jax.tree_util.tree_map(jnp.asarray,saved['params'])
    args.out.mkdir(parents=True)
    records = []
    for item in data['pools'][args.split]:
        for seed in data[args.split+'_noise_seeds']:
            draws = noise(seed,item['rid'])
            for name,phi in [('Safety',baseline),('C1',params)]:
                row,trace = execute(phi,field,item['initial'],draws,plant,cbf)
                row.update(method=name,rid=item['rid'],noise_seed=seed)
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
            mean_stagnation_seconds=float(np.mean([r['stagnation_seconds'] for r in rows])))
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
        split=args.split,checkpoint_sha256=digest(args.checkpoint),
        protocol='Certificate candidate; residual active from step0; recoverable and first-event labels reported',
        selected_feasible='selection' in saved,objective=config['objective']))
    atomic_save(args.out/'complete.json',dict(episodes=len(records)))


if __name__=='__main__':
    main()

