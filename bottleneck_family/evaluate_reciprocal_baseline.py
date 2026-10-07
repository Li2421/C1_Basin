"""Development-only safety-first reciprocal traffic baseline for Gap 1.

Every robot follows the same static gate route. A single common speed scale
keeps the joint swept step safe; without a right-of-way rule, opposing robots
can mutually block in the opening. The detector requires reciprocal demand,
spatial proximity at the gate, and five seconds of measured non-progress.
"""
from __future__ import annotations

from dataclasses import replace
import argparse
import json
from pathlib import Path

import numpy as np
from single_integrator.environment import point_segment_distance, segment_distance
from new_benchmark_common.dev_closed_loop import load_dev_nominal_cases
from shared_rollout_db.src.planner import preflight
from shared_rollout_db.src.rollout_db import eta_identity, uid

from .environment import BottleneckEnv
from .expert import SequentialExpert
from .scenario import Config


DEADLOCK_CRITERION = ('reciprocal_gate_wait_v1: opposite-side unresolved pair within '
                      'physical contact+0.001 m and |x-gate|<1.0 m; both desired velocities point '
                      'toward the other; each moves <0.015 m over 5.0 s')


def _safe_scaled(env, desired, scale):
    p = env.positions
    q = p + env.config.dt*scale*desired
    geometry = env.instance.geometry
    wall = segment_distance(p[:, None], q[:, None],
                            env.walls[None, :, 0], env.walls[None, :, 1]).min()-geometry.margin
    relative_p = p[env._i]-p[env._j]
    relative_q = q[env._i]-q[env._j]
    peer = point_segment_distance(np.zeros(2), relative_p, relative_q).min()-(
        2*env.config.agent_radius+env.config.agent_collision_margin)
    return bool(wall > .0001 and peer > .0001 and geometry.valid_points(q).all())


def _deadlock_pair(env, desired, history):
    c=env.config
    window=int(round(5.0/c.dt))
    if len(history) <= window:
        return None
    segment=np.asarray(history[-window-1:])
    displacement=np.linalg.norm(segment-segment[0],axis=2).max(axis=0)
    gate_x=c.barrier_x[0]
    distance=np.linalg.norm(env.positions[:,None]-env.positions[None,:],axis=2)
    pending=np.linalg.norm(env.positions-env.goals,axis=1)>c.goal_tolerance
    for i in range(c.num_agents):
        if not (pending[i] and displacement[i]<.015 and abs(env.positions[i,0]-gate_x)<1.0):
            continue
        for j in range(i+1,c.num_agents):
            if not (pending[j] and displacement[j]<.015 and
                    abs(env.positions[j,0]-gate_x)<1.0 and
                    distance[i,j]<2*c.agent_radius+c.agent_collision_margin+.001):
                continue
            gi=np.sign(env.goals[i,0]-gate_x)
            gj=np.sign(env.goals[j,0]-gate_x)
            relative=env.positions[j]-env.positions[i]
            if (gi == -gj and np.dot(desired[i],relative)>0 and
                    np.dot(desired[j],-relative)>0):
                return [i,j]
    return None


def run(dataset, output, *, max_cases=None, case_indices=None, pair_release=False):
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=True)
    manifest,cases,audit=load_dev_nominal_cases(dataset)
    if max_cases is not None and case_indices is not None:
        raise ValueError('choose max_cases or case_indices')
    if case_indices is not None:
        cases=[cases[i] for i in case_indices]
    elif max_cases is not None:
        cases=cases[:max_cases]
    config=Config(**manifest['scenario_config'])
    if len(config.barrier_x)!=1 or len(config.openings[0])!=1:
        raise ValueError('reciprocal baseline supports single-door Gap 1 only')
    controller_name=('paired_release_reciprocal_common_scale_v1' if pair_release else
                     'reciprocal_gate_common_scale_v1')
    controller_uid=uid('ctl',{'protocol':controller_name,
                              'deadlock_criterion':DEADLOCK_CRITERION})
    requests=[]
    for case in cases:
        state_uid=uid('state',{'physical_fingerprint':config.physical_fingerprint,
                               'state':case.initial_state})
        requests.append(dict(state_uid=state_uid,eta_uid=eta_identity((0.,0.,0.))[0],
                             controller_uid=controller_uid,seed_keys=['0']))
    plan=output/'planned_rollouts.json'
    plan.write_text(json.dumps({'requests':requests},indent=2)+'\n')
    cache=preflight(plan)
    (output/'cache_preflight.json').write_text(json.dumps(cache,indent=2)+'\n')
    if cache['summary']['ambiguous']:
        raise RuntimeError('ambiguous cache evidence')
    rows=[]
    for case in cases:
        state=case.initial_state
        env=BottleneckEnv(replace(config,seed=state['episode_seed'],split='dev'))
        env.reset(np.asarray(state['positions']),np.asarray(state['goals']))
        initial_hash=env.initial_state_sha256()
        gate_x=config.barrier_x[0];gate_y=config.openings[0][0][0]
        offset=config.barrier_thickness/2+env.instance.geometry.margin+.12
        destinations=[]
        for goal in env.goals:
            direction=1 if goal[0]>gate_x else -1
            destinations.append((np.array([gate_x-direction*offset,gate_y]),
                                 np.array([gate_x+direction*offset,gate_y]),goal))
        phases=np.zeros(config.num_agents,dtype=int)
        if pair_release:
            approach=[]
            planner=SequentialExpert()
            for i in (0,1):
                route=planner._route_to(env,i,env.positions[i],destinations[i][0])
                if route is None:
                    raise ValueError(f'no pair-release approach for agent {i}')
                approach.append([p.copy() for p in route[1:]])
            approach_index=[0,0]
            released=False
        positions=[env.positions.copy()]
        clearances=[]
        terminal='running';pair=None
        for _ in range(config.max_steps):
            desired=np.zeros(env.action_shape)
            if pair_release:
                if not released:
                    for i in (0,1):
                        while (approach_index[i]<len(approach[i]) and
                               np.linalg.norm(env.positions[i]-approach[i][approach_index[i]])<.02):
                            approach_index[i]+=1
                        if approach_index[i]<len(approach[i]):
                            delta=approach[i][approach_index[i]]-env.positions[i]
                            length=np.linalg.norm(delta)
                            desired[i]=delta*min(config.max_speed/length,1/config.dt)
                    if all(approach_index[i]==len(approach[i]) for i in (0,1)):
                        released=True
                else:
                    for i in (0,1):
                        delta=destinations[i][1]-env.positions[i]
                        length=np.linalg.norm(delta)
                        if length>.001:
                            desired[i]=delta*min(config.max_speed/length,1/config.dt)
            else:
                for i in range(config.num_agents):
                    while phases[i]<2 and np.linalg.norm(env.positions[i]-destinations[i][phases[i]])<.04:
                        phases[i]+=1
                    delta=destinations[i][phases[i]]-env.positions[i]
                    length=np.linalg.norm(delta)
                    threshold=config.goal_tolerance if phases[i]==2 else .001
                    if length>threshold:
                        desired[i]=delta*min(config.max_speed/length,1/config.dt)
            if _safe_scaled(env,desired,1.):
                scale=1.
            else:
                low,high=0.,1.
                for _ in range(22):
                    middle=(low+high)/2
                    if _safe_scaled(env,desired,middle):low=middle
                    else:high=middle
                scale=low
            _,done,info=env.step(scale*desired,diagnose_stalls=False)
            positions.append(env.positions.copy());clearances.append(info['swept_clearance'])
            pair=_deadlock_pair(env,desired,positions)
            if pair is not None:
                terminal='deadlock'
                break
            if done:
                terminal=info['termination']
                break
        if terminal=='running':terminal='timeout'
        row=dict(rollout_id=case.rollout_id,termination=terminal,
                 deadlock_detected=pair is not None,deadlock_pair=pair,
                 pair_release=pair_release,
                 steps=len(clearances),collision=env.collided,
                 minimum_swept_clearance=float(min(clearances)))
        rows.append(row);print(json.dumps(row),flush=True)
        trace_dir=output/'traces';trace_dir.mkdir(exist_ok=True)
        meta=dict(batch_id=output.name,scenario='bottleneck_family',
                  rollout_id=case.rollout_id,controller=controller_name,
                  checkpoint=f'nonlearned {controller_name}',
                  seed=state['episode_seed'],controller_rng_seed=None,
                  source_record=str(case.archive_path),initial_state_sha256=initial_hash,
                  config=env.config.to_dict(),start_step=0,episode_steps=len(clearances),
                  complete=True,termination=terminal,collision=bool(env.collided),
                  safety_enabled=True,deadlock_detected=pair is not None,
                  deadlock_criterion=DEADLOCK_CRITERION,deadlock_pair=pair)
        np.savez_compressed(trace_dir/f'{case.rollout_id}.npz',
                            positions=np.asarray(positions),steps=np.arange(len(positions)),
                            goals=env.goals,walls=env.walls,
                            swept_clearance=np.asarray(clearances),
                            metadata_json=np.asarray(json.dumps(meta)))
    result=dict(schema='gap_reciprocal_baseline_dev_v1',split='dev',
                pair_release=pair_release,
                opened_test_archives=audit['opened_test_archives'],rollouts=rows,
                aggregate=dict(count=len(rows),deadlock_count=sum(r['deadlock_detected'] for r in rows),
                               collision_count=sum(r['collision'] for r in rows)))
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-cases',type=int)
    parser.add_argument('--case-indices',type=int,nargs='+')
    parser.add_argument('--pair-release',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.dataset,args.output,max_cases=args.max_cases,
                         case_indices=args.case_indices,pair_release=args.pair_release)['aggregate'],indent=2))


if __name__=='__main__':main()
