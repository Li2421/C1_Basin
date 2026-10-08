"""Non-opposing training-state recovery data for large-N Gap1 Flow.

The reference is a state-only same-direction queue: agents already beyond
the opening continue to their goals, and the nearest pending entrant is
permitted to approach. It has no hidden release clock, opposing-side rule,
eta, or evaluation-time control role. All new states originate from failed
or incomplete Flow rollouts of *train* initial conditions only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from new_benchmark_common.dataset import (DatasetWriter, JointTransitionDataset,
                                          RecoveryAudit, Trajectory)
from new_benchmark_common.macflow import load_checkpoint
from shared_control.hard_projection import HardProjectionConfig
from shared_rollout_db.src.rollout_db import eta_identity, uid
from shared_rollout_db.src.planner import preflight
from .collect_competence import teacher_reference
from .environment import BottleneckEnv
from .evaluate_competence import cases_from_dataset
from .evaluate_safety_audit import flow_action
from .flow_dataset import GapFlowScenario
from .observation import policy_observation_competence
from .scenario import Config


def state_queue_rollout(config, positions, velocities, goals, moving, projector):
    env = BottleneckEnv(config)
    env.reset(positions, goals)
    env.velocities = np.asarray(velocities, dtype=np.float64).copy()
    states = [env.positions.copy()]
    observations = [policy_observation_competence(env)]
    actions = []
    clearance = float('inf')
    moving = np.asarray(moving, dtype=int)
    for _ in range(config.max_steps):
        pending = moving[env.positions[moving, 0] <= .8]
        blocked = pending
        allowed = int(pending[np.argmax(env.positions[pending, 0])]) if len(pending) else None
        reference = teacher_reference(env)
        reference[blocked] = 0
        if allowed is not None:
            reference[allowed] = teacher_reference(env)[allowed]
        safe = np.asarray(projector(env.snapshot(), reference).velocity, dtype=np.float64)
        _, done, info = env.step(safe, diagnose_stalls=False)
        states.append(env.positions.copy())
        observations.append(policy_observation_competence(env))
        actions.append(safe.copy())
        clearance = min(clearance, info['swept_clearance'])
        if done:
            break
    return dict(success=env.termination == 'success' and not env.collided,
                termination=env.termination, states=np.asarray(states),
                observations=np.asarray(observations), actions=np.asarray(actions),
                min_swept_clearance=clearance)


def replay_transform(config, p, velocity, goals, actions, *, mirror):
    pp, vv, gg, aa = (np.asarray(value).copy() for value in (p, velocity, goals, actions))
    if mirror:
        pp[:, 0] *= -1
        vv[:, 0] *= -1
        gg[:, 0] *= -1
        aa[:, :, 0] *= -1
    env = BottleneckEnv(config)
    env.reset(pp, gg)
    env.velocities = vv
    states = [env.positions.copy()]
    observations = [policy_observation_competence(env)]
    clearance = float('inf')
    for action in aa:
        _, done, info = env.step(action, diagnose_stalls=False)
        states.append(env.positions.copy())
        observations.append(policy_observation_competence(env))
        clearance = min(clearance, info['swept_clearance'])
        if done:
            break
    if env.termination != 'success' or len(states) != len(aa) + 1 or clearance <= 0:
        raise RuntimeError('recovery mirror replay failed')
    return pp, vv, gg, np.asarray(states), np.asarray(observations), aa, clearance


def collect(source, checkpoint, output, *, cases_per_category=8,
            categories=('solo_LR', 'full_LR'), horizon=2000,
            anchors=(250,750,1500), seed=701027,
            goal_stop=False, hold_initially_passive=False,
            source_id_substring=None):
    source, checkpoint, output = Path(source), Path(checkpoint), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    manifest, all_cases = cases_from_dataset(source, 'train', categories=categories)
    counts = {}
    cases = []
    for case in all_cases:
        name, category, _ = case
        if source_id_substring is not None and source_id_substring not in name:
            continue
        if counts.get(category, 0) >= cases_per_category:
            continue
        counts[category] = counts.get(category, 0) + 1
        cases.append(case)
    if not cases:
        raise ValueError('no train cases matched the recovery selection')
    config = Config(**manifest['scenario_config'])
    if config.num_agents not in (10, 50):
        raise ValueError('scale recovery collection supports N=10 or N=50')
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    agent, _ = load_checkpoint(checkpoint,
        expected_environment_fingerprint=manifest['environment_fingerprint'])
    projector = CertifiedHardSafetyFilter(HardProjectionConfig())
    requests = []
    for name, category, state in cases:
        requests.append(dict(state_uid=uid('state', {
            'physical_fingerprint':config.physical_fingerprint,
            'positions':state['positions'], 'goals':state['goals'],
            'episode_seed':state['episode_seed'], 'split':'train'}),
            eta_uid=eta_identity((0.,0.,0.))[0],
            controller_uid=uid('ctl', {'flow_sha256':checkpoint_sha,
                'controller':'gap1_large_n_flow_plus_accepted_safety',
                'observation':'competence_v2','samples':1,'horizon':horizon,
                'goal_stop':goal_stop,
                'hold_initially_passive':hold_initially_passive}),
            seed_keys=[str(seed)]))
    model_plan = output / 'model_planned_rollouts.json'
    model_plan.write_text(json.dumps({'requests':requests},indent=2)+'\n')
    model_cache = preflight(model_plan)
    (output/'model_cache_preflight.json').write_text(json.dumps(model_cache,indent=2)+'\n')
    if model_cache['summary']['ambiguous'] or model_cache['summary']['genuinely_missing'] != len(cases):
        raise RuntimeError('model rollout cache requires review/reuse')
    candidates = []
    model_rows = []
    for index,(name,category,state) in enumerate(cases):
        episode_config = replace(config,seed=state['episode_seed'],split='train')
        env = BottleneckEnv(episode_config)
        env.reset(state['positions'],state['goals'])
        moving = np.flatnonzero(np.linalg.norm(env.goals-env.positions,axis=1) > config.goal_tolerance)
        passive = np.setdiff1d(np.arange(config.num_agents),moving)
        episode_key = jax.random.fold_in(jax.random.PRNGKey(seed), index)
        positions = [env.positions.copy()]
        velocities = [env.velocities.copy()]
        for step in range(min(horizon,config.max_steps)):
            reference = flow_action(agent,env,episode_key,step,1,
                                    observation_version='competence_v2')
            if hold_initially_passive:
                reference[passive] = 0
            if goal_stop:
                reference[np.linalg.norm(env.goals-env.positions,axis=1)
                          <=config.goal_tolerance] = 0
            safe = np.asarray(projector(env.snapshot(),reference).velocity,dtype=np.float64)
            _,done,info = env.step(safe,diagnose_stalls=False)
            positions.append(env.positions.copy())
            velocities.append(env.velocities.copy())
            if done:
                break
        positions = np.asarray(positions)
        velocities = np.asarray(velocities)
        trace_dir = output/'model_traces'
        trace_dir.mkdir(exist_ok=True)
        np.savez_compressed(trace_dir/f'{name}.npz', positions=positions,
            velocities=velocities, goals=env.goals,
            metadata_json=np.asarray(json.dumps({
                'source_rollout_id':name,'split':'train','category':category,
                'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_sha,
                'controller_seed':seed,'episode_seed':state['episode_seed'],
                'goal_stop':goal_stop,
                'hold_initially_passive':hold_initially_passive,
                'termination':env.termination,'collision':bool(env.collided),
                'physical_fingerprint':config.physical_fingerprint})))
        model_rows.append(dict(source=name,category=category,steps=env.step_count,
                               termination=env.termination,collision=env.collided,
                               trace=str(trace_dir/f'{name}.npz')))
        for anchor in anchors:
            if anchor < len(positions)-1:
                recovery_name = f'{name}_flowrecovery_t{anchor:04d}'
                candidates.append((recovery_name,episode_config,positions[anchor],
                                   velocities[anchor],env.goals.copy(),moving,name,anchor))
        print(json.dumps({'model_case':index+1,'source':name,
                          'steps':env.step_count,'termination':env.termination}),flush=True)
    (output/'model_rollouts.json').write_text(json.dumps(model_rows,indent=2)+'\n')
    recovery_requests = []
    for name,episode_config,p,v,g,moving,parent,anchor in candidates:
        recovery_requests.append(dict(state_uid=uid('state',{
            'physical_fingerprint':config.physical_fingerprint,
            'positions':p.tolist(),'last_velocity':v.tolist(),'goals':g.tolist(),
            'episode_seed':episode_config.seed,'parent':parent,'anchor':anchor}),
            eta_uid=eta_identity((0.,0.,0.))[0],
            controller_uid=uid('ctl',{'controller':'state_only_same_direction_queue_teacher_v1',
                'horizon':config.max_steps}),seed_keys=[str(episode_config.seed)]))
    recovery_plan = output/'recovery_planned_rollouts.json'
    recovery_plan.write_text(json.dumps({'requests':recovery_requests},indent=2)+'\n')
    recovery_cache=preflight(recovery_plan)
    (output/'recovery_cache_preflight.json').write_text(json.dumps(recovery_cache,indent=2)+'\n')
    if recovery_cache['summary']['ambiguous'] or recovery_cache['summary']['genuinely_missing']!=len(candidates):
        raise RuntimeError('recovery rollout cache requires review/reuse')
    writer = DatasetWriter(output/'dataset',GapFlowScenario(config),scenario_config=asdict(config))
    source_manifest = json.loads((source/'manifest.json').read_text())
    for split in ('train','dev','test'):
        if source_manifest['counts']['split'].get(split, 0) == 0:
            continue
        original=JointTransitionDataset(source,split)
        for trajectory in original.trajectories:
            writer.add(trajectory)
    report={'schema':'gap1_scale_dagger_v1','source_dataset':str(source),
            'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_sha,
            'goal_stop':goal_stop,
            'source_id_substring':source_id_substring,
            'hold_initially_passive':hold_initially_passive,
            'model_rollouts':model_rows,'candidate_count':len(candidates),
            'accepted':{'LR':{'trajectories':0,'transitions':0},
                        'RL':{'trajectories':0,'transitions':0}},
            'teacher_failures':[],'model_preflight':model_cache['summary'],
            'recovery_preflight':recovery_cache['summary']}
    for index,(name,episode_config,p,v,g,moving,parent,anchor) in enumerate(candidates):
        result = state_queue_rollout(episode_config,p,v,g,moving,projector)
        if not result['success']:
            report['teacher_failures'].append(dict(name=name,termination=result['termination'],
                final_goal_distance=np.linalg.norm(result['states'][-1]-g,axis=1).tolist()))
            continue
        for mirror in (False,True):
            direction='RL' if mirror else 'LR'
            pp,vv,gg,states,obs,actions,clearance=replay_transform(
                episode_config,p,v,g,result['actions'],mirror=mirror)
            recovery=RecoveryAudit(parent,'train',anchor,None,None,None,seed+index,True)
            writer.add(Trajectory(f'{name}_{direction}','train','uniform_recovery',
                {'positions':pp,'velocities':vv,'goals':gg,
                 'episode_seed':episode_config.seed,'split':'train',
                 'mode':'nonopposing_flow_recovery','direction':direction,
                 'source_model_rollout_id':parent,'source_time':anchor},
                states,obs,actions,
                {'success':True,'terminal_reason':'success','direction':direction,
                 'teacher':'state_only_same_direction_queue_plus_accepted_safety',
                 'source_model_checkpoint_sha256':checkpoint_sha,
                 'min_swept_clearance':clearance},recovery))
            report['accepted'][direction]['trajectories'] += 1
            report['accepted'][direction]['transitions'] += len(actions)
        if (index+1)%10==0:
            print(json.dumps({'recovery_done':index+1,
                              'accepted':report['accepted'],'failed':len(report['teacher_failures'])}),flush=True)
    if report['accepted']['LR'] != report['accepted']['RL']:
        raise RuntimeError('directional recovery imbalance')
    writer.finalize(extra_report=report)
    (output/'collection_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cases-per-category',type=int,default=8)
    parser.add_argument('--categories',nargs='+',default=('solo_LR','full_LR'))
    parser.add_argument('--horizon',type=int,default=2000)
    parser.add_argument('--anchors',type=int,nargs='+',default=(250,750,1500))
    parser.add_argument('--seed',type=int,default=701027)
    parser.add_argument('--goal-stop',action='store_true')
    parser.add_argument('--hold-initially-passive',action='store_true')
    parser.add_argument('--source-id-substring')
    args=parser.parse_args()
    print(json.dumps(collect(args.source,args.checkpoint,args.output,
        cases_per_category=args.cases_per_category,horizon=args.horizon,
        categories=tuple(args.categories),anchors=tuple(args.anchors),seed=args.seed,
        goal_stop=args.goal_stop,
        hold_initially_passive=args.hold_initially_passive,
        source_id_substring=args.source_id_substring),indent=2),flush=True)


if __name__=='__main__':
    main()
