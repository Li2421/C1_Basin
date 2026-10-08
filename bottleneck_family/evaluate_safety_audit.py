"""Frozen Gap1 Flow + accepted hard safety projection, with competence controls.

No eta, route planner, priority rule, trained weight update, or new safety law.
The N=2 single-active-task control keeps the checkpoint's required two-row
observation: one robot has its goal at its start and receives zero reference
velocity before the unchanged joint safety projection.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from new_benchmark_common.dev_closed_loop import load_nominal_cases
from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions
from shared_control.hard_projection import CBFSolverError, HardProjectionConfig
from shared_rollout_db.src.rollout_db import eta_identity, uid
from shared_rollout_db.src.planner import preflight
from .environment import BottleneckEnv
from .observation import policy_observation, policy_observation_competence
from .scenario import Config


ROOT = Path('diagnostics/gap_flow_v1')
SPECS = {2: 'n2_recovery_wide', 10: 'n10_wide_recovery', 50: 'n50_wide_recovery'}
MODES = ('solo_even', 'solo_odd', 'solo_odd_remote', 'one_way', 'temporal_even_first',
         'temporal_odd_first', 'temporal_release_even_first',
         'temporal_release_odd_first', 'opposing', 'one_way_mirror',
         'single_even', 'single_odd', 'few_even', 'few_odd',
         'half_even', 'half_odd')


def control_state(state, mode):
    positions = np.asarray(state['positions'], dtype=np.float64).copy()
    goals = np.asarray(state['goals'], dtype=np.float64).copy()
    if mode == 'solo_even':
        goals[1::2] = positions[1::2]
    elif mode == 'solo_odd':
        goals[0::2] = positions[0::2]
    elif mode == 'solo_odd_remote':
        # Reuse the other task's original, valid opposite-side goal point as
        # the passive position. The active odd robot keeps its exact task.
        positions[0::2] = np.asarray(state['goals'], dtype=np.float64)[0::2]
        goals[0::2] = positions[0::2]
    elif mode == 'one_way':
        positions[1::2] = np.asarray(state['goals'], dtype=np.float64)[1::2]
        goals[1::2] = np.asarray(state['positions'], dtype=np.float64)[1::2]
    elif mode == 'one_way_mirror':
        positions[1::2] = np.asarray(state['goals'], dtype=np.float64)[1::2]
        goals[1::2] = np.asarray(state['positions'], dtype=np.float64)[1::2]
        positions[:, 0] *= -1
        goals[:, 0] *= -1
    elif mode in ('single_even', 'single_odd', 'few_even', 'few_odd',
                  'half_even', 'half_odd'):
        original = goals.copy()
        goals = positions.copy()
        side = np.arange(0 if mode.endswith('even') else 1, len(goals), 2)
        if mode.startswith('single_'):
            side = side[:1]
        elif mode.startswith('few_'):
            side = side[:5]
        goals[side] = original[side]
    elif mode.startswith('temporal_release_'):
        waiting = np.arange(1 if mode.endswith('even_first') else 0,len(goals),2)
        goals[waiting] = positions[waiting]
        # A 0.15 m same-room staging goal keeps the complete task from being
        # marked successful before the second group is released.  Its control
        # velocity remains explicitly zero until then.
        goals[waiting,0] -= .15*np.sign(positions[waiting,0])
    return positions, goals


def flow_action(agent, env, episode_key, step, samples, *, observation_version='historical',
                latent_mode='per_step'):
    observer = policy_observation_competence if observation_version == 'competence_v2' else policy_observation
    observation = (observer(env) if agent.config['obs_dim'] == 8 else
                   env.observation()['agents'].astype(np.float32))
    sampled = np.asarray(sample_bounded_actions(agent,
        np.repeat(observation[None], samples, axis=0),
        jax.random.fold_in(episode_key, step) if latent_mode=='per_step'
        else episode_key), dtype=np.float64)
    action = sampled.mean(axis=0)
    norms = np.linalg.norm(action, axis=1, keepdims=True)
    return action * np.minimum(1.0, env.config.max_speed / np.maximum(norms, 1e-30))


def run(agents: int, mode: str, output: Path, *, samples=1, seed=17,
        checkpoint_override=None, split='test', observation_version='historical',
        fresh_count=0, fresh_seed=271828, goal_stop=False,
        latent_mode='per_step'):
    if mode not in MODES or agents not in SPECS or samples < 1:
        raise ValueError('invalid N, mode, or sample count')
    if mode.startswith('solo_') and agents != 2:
        raise ValueError('single-active-task control is defined only for N=2')
    if (split not in ('dev','test') or observation_version not in ('historical','competence_v2')
            or latent_mode not in ('per_step','per_episode') or fresh_count < 0):
        raise ValueError('invalid split or observation version')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    root = ROOT / SPECS[agents]
    if fresh_count:
        manifest_path=root/'dataset/manifest.json'
        manifest=json.loads(manifest_path.read_text())
        audit={'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
        cases=[]
    else:
        manifest, cases, audit = load_nominal_cases(root/'dataset', split=split,
                                                   final_evaluation=(split=='test'))
    config = Config(**manifest['scenario_config'])
    if fresh_count:
        rng=np.random.default_rng(fresh_seed)
        fresh=[]
        for index in range(fresh_count):
            episode_seed=int(rng.integers(0,2**31-1))
            instance=BottleneckEnv(replace(config,seed=episode_seed,split=split))
            state={'positions':instance.positions.tolist(),
                   'goals':instance.goals.tolist(),
                   'episode_seed':episode_seed,'split':split}
            fresh.append(SimpleNamespace(rollout_id=f'fresh_control_{index:04d}',
                                         initial_state=state,archive_path=output/'fresh_control_states.json'))
        cases=fresh
        (output/'fresh_control_states.json').write_text(json.dumps(
            {'fresh_seed':fresh_seed,'split':split,'cases':[c.initial_state for c in cases]},indent=2)+'\n')
    checkpoint = Path(checkpoint_override) if checkpoint_override else root/'train/best.pkl'
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    agent, checkpoint_meta = load_checkpoint(
        checkpoint, expected_environment_fingerprint=manifest['environment_fingerprint'])
    if agent.config['num_agents'] != agents:
        raise ValueError('checkpoint N mismatch')
    cbf = HardProjectionConfig()
    projector = CertifiedHardSafetyFilter(cbf)
    controller_uid = uid('ctl', {'flow_sha256':checkpoint_sha,
        'control':'gap_joint_macflow_stage1_plus_certified_hard_safety_v1',
        'mode':mode,'sample_count':samples,'evaluation_seed':seed,
        'observation_version':observation_version,'split':split,
        'safety_config':cbf.to_dict(),'goal_stop':goal_stop,'latent_mode':latent_mode})
    requests=[]
    for case in cases:
        p,g=control_state(case.initial_state,mode)
        state_uid=uid('state',{'physical_fingerprint':config.physical_fingerprint,
            'positions':p.tolist(),'goals':g.tolist(),
            'episode_seed':case.initial_state['episode_seed'],'control_mode':mode,
            'scheduled_goals':case.initial_state['goals'] if mode.startswith('temporal_release_') else None})
        requests.append(dict(state_uid=state_uid,
            eta_uid=eta_identity((0.,0.,0.))[0],controller_uid=controller_uid,
            seed_keys=[str(seed)]))
    plan=output/'planned_rollouts.json'
    plan.write_text(json.dumps({'requests':requests},indent=2)+'\n')
    cache=preflight(plan)
    (output/'cache_preflight.json').write_text(json.dumps(cache,indent=2)+'\n')
    if cache['summary']['ambiguous']:
        raise RuntimeError('ambiguous preflight evidence')
    if cache['summary']['genuinely_missing'] != cache['summary']['total_requested']:
        raise RuntimeError('cached compatible rollouts present; select/reuse them before running new seeds')
    rows=[]
    for index,case in enumerate(cases):
        state=case.initial_state
        p,g=control_state(state,mode)
        env=BottleneckEnv(replace(config,seed=state['episode_seed'],split=split))
        env.reset(p,g)
        initial_hash=env.initial_state_sha256()
        episode_key=jax.random.fold_in(jax.random.PRNGKey(seed),index)
        positions=[env.positions.copy()]
        series={key:[] for key in ('u_flow','u_ref','u_safe','swept_clearance',
            'swept_wall_clearance','swept_agent_clearance','projection_norm',
            'active_pair_count','active_wall_count','active_speed_count',
            'min_wall_h','goal_error','status')}
        numerical_error=None;phase=0;phase_transition=None
        even_first=mode.endswith('even_first')
        first=np.arange(0 if even_first else 1,agents,2)
        second=np.arange(1 if even_first else 0,agents,2)
        for step in range(config.max_steps):
            flow=flow_action(agent,env,episode_key,step,samples,
                             observation_version=observation_version,
                             latent_mode=latent_mode)
            reference=flow.copy()
            if mode=='solo_even':
                reference[1::2]=0
            elif mode in ('solo_odd','solo_odd_remote'):
                reference[0::2]=0
            elif mode.startswith('temporal_'):
                if phase==0 and np.all(np.linalg.norm(env.positions[first]-env.goals[first],axis=1)
                                       <=config.goal_tolerance):
                    phase=1;phase_transition=step
                    if mode.startswith('temporal_release_'):
                        env.goals[second]=np.asarray(state['goals'],dtype=np.float64)[second]
                # Hold only the unreleased group.  Once released, the first
                # group must retain ordinary goal feedback: the safety QP can
                # displace an agent just outside tolerance during phase two.
                if phase==0:
                    reference[second]=0
            if goal_stop:
                reference[np.linalg.norm(env.goals-env.positions,axis=1)
                          <=config.goal_tolerance]=0
            try:
                result=projector(env.snapshot(),reference)
                safe=np.asarray(result.velocity,dtype=np.float64)
                _,done,info=env.step(safe,diagnose_stalls=False)
            except (CBFSolverError,ValueError,FloatingPointError) as exc:
                numerical_error={'step':step,'type':type(exc).__name__,'message':str(exc)}
                break
            diagnostics=result.diagnostics
            active=np.asarray(diagnostics['active_linear_constraints'],dtype=int)
            pair_count=int(diagnostics['num_pair_constraints'])
            series['u_flow'].append(flow)
            series['u_ref'].append(reference)
            series['u_safe'].append(safe)
            series['swept_clearance'].append(info['swept_clearance'])
            series['swept_wall_clearance'].append(info['min_swept_wall_clearance'])
            series['swept_agent_clearance'].append(info['min_swept_agent_clearance'])
            series['projection_norm'].append(float(np.linalg.norm(safe-reference)))
            series['active_pair_count'].append(int(np.sum(active<pair_count)))
            series['active_wall_count'].append(int(np.sum(active>=pair_count)))
            series['active_speed_count'].append(len(diagnostics['active_speed_constraints']))
            series['min_wall_h'].append(diagnostics['min_wall_h'])
            series['goal_error'].append(float(np.linalg.norm(env.positions-env.goals,axis=1).sum()))
            series['status'].append(str(result.status))
            positions.append(env.positions.copy())
            if done:break
        terminal='numerical_failure' if numerical_error else env.termination
        if terminal=='running': terminal='timeout'
        correction=np.asarray(series['projection_norm'])
        wall=np.asarray(series['swept_wall_clearance'])
        pair=np.asarray(series['swept_agent_clearance'])
        remaining=np.linalg.norm(env.positions-env.goals,axis=1)
        row=dict(rollout_id=case.rollout_id,N=agents,mode=mode,termination=terminal,
            success=terminal=='success' and not env.collided,collision=bool(env.collided),
            steps=env.step_count,individual_goal_fraction=float(np.mean(remaining<=config.goal_tolerance)),
            final_mean_goal_distance=float(remaining.mean()),
            min_swept_wall_clearance=float(wall.min()) if len(wall) else None,
            min_swept_agent_clearance=float(pair.min()) if len(pair) else None,
            projection_active_fraction=float(np.mean(correction>cbf.intervention_tol)) if len(correction) else None,
            mean_projection_norm=float(correction.mean()) if len(correction) else None,
            active_pair_fraction=float(np.mean(np.asarray(series['active_pair_count'])>0)) if len(correction) else None,
            active_wall_fraction=float(np.mean(np.asarray(series['active_wall_count'])>0)) if len(correction) else None,
            phase_transition_step=phase_transition,numerical_error=numerical_error)
        rows.append(row)
        print(json.dumps(row),flush=True)
        trace_dir=output/'traces';trace_dir.mkdir(exist_ok=True)
        meta=dict(batch_id='gap_flow_safety_audit_v1',scenario='bottleneck_family',
            rollout_id=f'{case.rollout_id}_{mode}',controller=f'joint_macflow_stage1_mc{samples}_certified_hard_safety',
            checkpoint=f'{checkpoint} sha256={checkpoint_sha}',seed=state['episode_seed'],
            controller_rng_seed=seed,source_record=str(case.archive_path),
            initial_state_sha256=initial_hash,config=env.config.to_dict(),
            start_step=0,episode_steps=env.step_count,complete=numerical_error is None,
            termination=terminal,collision=bool(env.collided),safety_enabled=True,
            control_mode=mode,phase_transition_step=phase_transition,
            goal_release=mode.startswith('temporal_release_'),goal_stop=goal_stop,
            latent_mode=latent_mode,
            observation_version=observation_version)
        payload={key:np.asarray(value) for key,value in series.items()}
        for key in ('u_flow','u_ref','u_safe'):
            payload[key]=payload[key].reshape((-1,agents,2))
        np.savez_compressed(trace_dir/f'{case.rollout_id}.npz',positions=np.asarray(positions),
            steps=np.arange(len(positions)),goals=env.goals,walls=env.walls,
            metadata_json=np.asarray(json.dumps(meta)),**payload)
    result=dict(schema='gap_flow_safety_audit_v1',N=agents,mode=mode,
        checkpoint_sha256=checkpoint_sha,checkpoint_metadata=checkpoint_meta,
        dataset_manifest_sha256=audit['manifest_sha256'],samples_per_step=samples,
        safety='CertifiedHardSafetyFilter(HardProjectionConfig())',
        safety_config=cbf.to_dict(),cache_preflight=cache['summary'],rollouts=rows,
        observation_version=observation_version,split=split,goal_stop=goal_stop,
        latent_mode=latent_mode,
        fresh_count=fresh_count,fresh_seed=fresh_seed if fresh_count else None)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agents',type=int,choices=tuple(SPECS),required=True)
    parser.add_argument('--mode',choices=MODES,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples-per-step',type=int,default=1)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--checkpoint-override',type=Path)
    parser.add_argument('--split',choices=('dev','test'),default='test')
    parser.add_argument('--observation-version',choices=('historical','competence_v2'),default='historical')
    parser.add_argument('--fresh-count',type=int,default=0)
    parser.add_argument('--fresh-seed',type=int,default=271828)
    parser.add_argument('--goal-stop',action='store_true',
                        help='At-goal zero-action guard fixed before opposing evaluation')
    parser.add_argument('--latent-mode',choices=('per_step','per_episode'),default='per_step')
    args=parser.parse_args()
    run(args.agents,args.mode,args.output,samples=args.samples_per_step,seed=args.seed,
        checkpoint_override=args.checkpoint_override,split=args.split,
        observation_version=args.observation_version,
        fresh_count=args.fresh_count,fresh_seed=args.fresh_seed,
        goal_stop=args.goal_stop,latent_mode=args.latent_mode)


if __name__=='__main__':main()
