"""Frozen-checkpoint, held-out solo/one-way Gap1 competence evaluation."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import jax
import numpy as np

from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions
from shared_control.hard_projection import CBFSolverError, HardProjectionConfig
from shared_rollout_db.src.rollout_db import eta_identity, uid
from shared_rollout_db.src.planner import preflight
from .environment import BottleneckEnv
from .observation import policy_observation_competence
from .scenario import Config


def cases_from_dataset(root, split, *, max_cases_per_category=None, categories=None,
                       rollout_id_substring=None):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    cases = []
    counts = {}
    for row in manifest['files']:
        if row['split'] != split:
            continue
        if rollout_id_substring is not None and rollout_id_substring not in row['rollout_id']:
            continue
        with np.load(root / row['file'], allow_pickle=False) as data:
            state = json.loads(str(data['initial_state_json'].item()))
        if not row['rollout_id'].endswith(('swap0','perm0')):
            continue
        category = f"{state['mode']}_{state['direction']}"
        if categories is not None and category not in categories:
            continue
        if max_cases_per_category is not None and counts.get(category, 0) >= max_cases_per_category:
            continue
        counts[category] = counts.get(category, 0) + 1
        cases.append((row['rollout_id'], category, state))
    return manifest, cases


def run(dataset_root, checkpoint, split, output, *, seed=17, samples=1,
        max_cases_per_category=None, pilot_horizon=None, categories=None,
        hold_initially_passive=False, goal_stop=False,
        latent_mode='per_step', rollout_id_substring=None):
    if latent_mode not in ('per_step','per_episode'):
        raise ValueError('invalid Flow latent mode')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest, cases = cases_from_dataset(dataset_root, split,
        max_cases_per_category=max_cases_per_category,categories=categories,
        rollout_id_substring=rollout_id_substring)
    if not cases:
        raise ValueError('no competence cases match the requested split and filters')
    config = Config(**manifest['scenario_config'])
    checkpoint = Path(checkpoint)
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    agent, checkpoint_meta = load_checkpoint(checkpoint,
        expected_environment_fingerprint=manifest['environment_fingerprint'])
    if agent.config['num_agents'] != config.num_agents:
        raise ValueError('checkpoint/competence-case agent count mismatch')
    cbf = HardProjectionConfig()
    projector = CertifiedHardSafetyFilter(cbf)
    controller_uid = uid('ctl', {'flow_sha256':checkpoint_sha,
        'control':'gap_competence_flow_plus_certified_hard_safety_v1',
        'sample_count':samples,'seed':seed,'safety':cbf.to_dict(),
        'observation':'competence_v2','goal_stop':goal_stop,
        'hold_initially_passive':hold_initially_passive,
        'latent_mode':latent_mode})
    requests=[]
    for rollout_id,category,state in cases:
        requests.append(dict(state_uid=uid('state',{
            'physical_fingerprint':config.physical_fingerprint,
            'positions':state['positions'],'goals':state['goals'],
            'episode_seed':state['episode_seed'],'category':category}),
            eta_uid=eta_identity((0.,0.,0.))[0],controller_uid=controller_uid,
            seed_keys=[str(seed)]))
    (output/'planned_rollouts.json').write_text(json.dumps({'requests':requests},indent=2)+'\n')
    cache=preflight(output/'planned_rollouts.json')
    (output/'cache_preflight.json').write_text(json.dumps(cache,indent=2)+'\n')
    if cache['summary']['ambiguous'] or cache['summary']['genuinely_missing'] != len(cases):
        raise RuntimeError('preflight requires cache reuse/review')
    rows=[]
    for index,(rollout_id,category,state) in enumerate(cases):
        env=BottleneckEnv(replace(config,seed=state['episode_seed'],split=split))
        env.reset(state['positions'],state['goals'])
        initial_hash=env.initial_state_sha256()
        moving=np.flatnonzero(np.linalg.norm(env.goals-env.positions,axis=1)>config.goal_tolerance)
        passive=np.setdiff1d(np.arange(config.num_agents),moving)
        episode_key=jax.random.fold_in(jax.random.PRNGKey(seed),index)
        positions=[env.positions.copy()]
        series={name:[] for name in ('u_flow','u_ref','u_safe','projection_norm',
            'wall_active','pair_active','swept_wall_clearance','swept_agent_clearance',
            'swept_clearance','goal_error','wall_constraint_count','pair_constraint_count')}
        numerical_error=None
        for step in range(min(config.max_steps,pilot_horizon or config.max_steps)):
            observation=policy_observation_competence(env)
            sampled=np.asarray(sample_bounded_actions(agent,
                np.repeat(observation[None],samples,axis=0),
                jax.random.fold_in(episode_key,step) if latent_mode=='per_step'
                else episode_key),dtype=np.float64)
            flow=sampled.mean(axis=0)
            norms=np.linalg.norm(flow,axis=1,keepdims=True)
            flow*=np.minimum(1.,config.max_speed/np.maximum(norms,1e-30))
            reference=flow.copy()
            if hold_initially_passive:
                reference[passive]=0
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
            active_rows=np.asarray(result.diagnostics['active_linear_constraints'],dtype=int)
            pair_count=int(result.diagnostics['num_pair_constraints'])
            series['u_flow'].append(flow.copy())
            series['u_ref'].append(reference)
            series['u_safe'].append(safe)
            series['projection_norm'].append(float(np.linalg.norm(safe-reference)))
            series['wall_active'].append(bool(np.any(active_rows>=pair_count)))
            series['pair_active'].append(bool(np.any(active_rows<pair_count)))
            series['wall_constraint_count'].append(int(np.sum(active_rows>=pair_count)))
            series['pair_constraint_count'].append(int(np.sum(active_rows<pair_count)))
            series['swept_wall_clearance'].append(info['min_swept_wall_clearance'])
            series['swept_agent_clearance'].append(info['min_swept_agent_clearance'])
            series['swept_clearance'].append(info['swept_clearance'])
            series['goal_error'].append(float(np.linalg.norm(env.positions-env.goals,axis=1).sum()))
            positions.append(env.positions.copy())
            if done:break
        termination='numerical_failure' if numerical_error else env.termination
        if termination=='running':termination='pilot_horizon' if pilot_horizon else 'timeout'
        p=np.asarray(positions)
        direction=1 if category.endswith('LR') else -1
        oriented=(p[:,moving,0]-config.barrier_x[0])*direction
        remaining=np.linalg.norm(env.positions-env.goals,axis=1)
        correction=np.asarray(series['projection_norm'])
        wall=np.asarray(series['swept_wall_clearance'])
        pair=np.asarray(series['swept_agent_clearance'])
        row=dict(rollout_id=rollout_id,category=category,success=termination=='success' and not env.collided,
            termination=termination,collision=bool(env.collided),steps=env.step_count,
            final_goal_distances=remaining.tolist(),final_mean_goal_distance=float(remaining.mean()),
            entered_gate=int(np.sum(np.any(abs(oriented)<config.barrier_thickness/2,axis=0))),
            crossed_gate=int(np.sum(np.any(oriented>config.barrier_thickness/2,axis=0))),
            moving_agents=len(moving),
            min_swept_wall_clearance=float(wall.min()) if len(wall) else None,
            min_swept_agent_clearance=float(pair.min()) if len(pair) else None,
            projection_active_fraction=float(np.mean(correction>cbf.intervention_tol)) if len(correction) else None,
            mean_projection_norm=float(correction.mean()) if len(correction) else None,
            wall_active_fraction=float(np.mean(series['wall_active'])) if len(correction) else None,
            pair_active_fraction=float(np.mean(series['pair_active'])) if len(correction) else None,
            numerical_error=numerical_error)
        rows.append(row)
        print(json.dumps(row),flush=True)
        trace_dir=output/'traces';trace_dir.mkdir(exist_ok=True)
        meta=dict(batch_id='gap_flow_competence_v1',scenario='bottleneck_family',
            rollout_id=rollout_id,controller=f'joint_macflow_stage1_mc{samples}_certified_hard_safety',
            checkpoint=f'{checkpoint} sha256={checkpoint_sha}',seed=state['episode_seed'],
            controller_rng_seed=seed,initial_state_sha256=initial_hash,
            source_record=str(Path(dataset_root)/'rollouts'/split/f'{rollout_id}.npz'),
            config=env.config.to_dict(),start_step=0,episode_steps=env.step_count,
            complete=numerical_error is None and pilot_horizon is None,
            termination=termination,collision=bool(env.collided),
            safety_enabled=True,category=category,observation='competence_v2',
            hold_initially_passive=hold_initially_passive,goal_stop=goal_stop,
            latent_mode=latent_mode)
        payload={name:np.asarray(values) for name,values in series.items()}
        for name in ('u_flow','u_ref','u_safe'):
            payload[name]=payload[name].reshape((-1,config.num_agents,2))
        np.savez_compressed(trace_dir/f'{rollout_id}.npz',positions=p,
            steps=np.arange(len(p)),goals=env.goals,walls=env.walls,
            metadata_json=np.asarray(json.dumps(meta)),**payload)
    categories=sorted(set(row['category'] for row in rows))
    summary={category:{'runs':sum(row['category']==category for row in rows),
        'success':sum(row['category']==category and row['success'] for row in rows),
        'collision':sum(row['category']==category and row['collision'] for row in rows),
        'timeout':sum(row['category']==category and row['termination']=='timeout' for row in rows),
        'median_wall_active_fraction':float(np.median([row['wall_active_fraction'] for row in rows if row['category']==category])),
        'median_projection_norm':float(np.median([row['mean_projection_norm'] for row in rows if row['category']==category]))}
        for category in categories}
    result=dict(schema='gap_flow_competence_evaluation_v1',checkpoint=str(checkpoint),
        checkpoint_sha256=checkpoint_sha,checkpoint_metadata=checkpoint_meta,
        dataset_manifest=str(Path(dataset_root)/'manifest.json'),split=split,
        samples_per_step=samples,seed=seed,safety='CertifiedHardSafetyFilter(HardProjectionConfig())',
        summary=summary,rollouts=rows,pilot_horizon=pilot_horizon,
        hold_initially_passive=hold_initially_passive,
        rollout_id_substring=rollout_id_substring)
    result['goal_stop']=goal_stop
    result['latent_mode']=latent_mode
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--split',choices=('dev','test'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--samples-per-step',type=int,default=1)
    parser.add_argument('--max-cases-per-category',type=int)
    parser.add_argument('--pilot-horizon',type=int)
    parser.add_argument('--categories',nargs='+')
    parser.add_argument('--rollout-id-substring')
    parser.add_argument('--hold-initially-passive',action='store_true',
                        help='Legacy clamped control; ordinary nominal Flow leaves this false')
    parser.add_argument('--goal-stop',action='store_true',
                        help='Reversible at-goal zero-action guard; included in controller identity')
    parser.add_argument('--latent-mode',choices=('per_step','per_episode'),default='per_step')
    args=parser.parse_args()
    run(args.dataset_root,args.checkpoint,args.split,args.output,
        seed=args.seed,samples=args.samples_per_step,
        max_cases_per_category=args.max_cases_per_category,
        pilot_horizon=args.pilot_horizon,categories=args.categories,
        hold_initially_passive=args.hold_initially_passive,
        goal_stop=args.goal_stop,latent_mode=args.latent_mode,
        rollout_id_substring=args.rollout_id_substring)


if __name__=='__main__':main()
