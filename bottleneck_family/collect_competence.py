"""Balanced Gap1 navigation demonstrations without opposing-agent negotiation.

Only solo and same-direction tasks are accepted.  Each successful base rollout
is reflected and row-permuted, giving exact left/right transition balance.
The teacher follows the public gate waypoint and uses the existing hard safety
projection; it has no priority, route search, or opposing-traffic rule.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from new_benchmark_common.dataset import DatasetWriter, Trajectory
from shared_control.hard_projection import HardProjectionConfig
from shared_rollout_db.src.rollout_db import eta_identity, uid
from shared_rollout_db.src.planner import preflight
from .environment import BottleneckEnv
from .flow_dataset import GapFlowScenario
from .observation import gate_waypoints_competence, policy_observation_competence
from .scenario import Config, build_instance


def teacher_reference(env):
    delta = gate_waypoints_competence(env) - env.positions
    distance = np.linalg.norm(delta, axis=1, keepdims=True)
    action = delta * np.minimum(1 / env.config.dt,
                                env.config.max_speed / np.maximum(distance, 1e-30))
    action[np.linalg.norm(env.goals - env.positions, axis=1) <= env.config.goal_tolerance] = 0
    return action


def task_state(instance, mode):
    positions = instance.positions.copy()
    goals = instance.goals.copy()
    if mode == 'solo':
        goals[1] = positions[1]
    elif mode == 'one_way':
        positions[1] = instance.goals[1]
        goals[1] = instance.positions[1]
    else:
        raise ValueError(mode)
    return positions, goals


def transform(positions, goals, *, mirror=False, swap=False):
    positions = np.asarray(positions).copy()
    goals = np.asarray(goals).copy()
    if mirror:
        positions[:, 0] *= -1
        goals[:, 0] *= -1
    if swap:
        positions = positions[::-1].copy()
        goals = goals[::-1].copy()
    return positions, goals


def rollout(config, positions, goals, projector, *, actions=None, max_horizon=None):
    env = BottleneckEnv(config)
    env.reset(positions, goals)
    states = [env.positions.copy()]
    observations = [policy_observation_competence(env)]
    commands = []
    correction = []
    wall_active = []
    min_wall_clearance = float('inf')
    horizon = len(actions) if actions is not None else min(config.max_steps,
                                                           max_horizon or config.max_steps)
    for step in range(horizon):
        if actions is None:
            reference = teacher_reference(env)
            result = projector(env.snapshot(), reference)
            velocity = np.asarray(result.velocity)
            correction.append(float(np.linalg.norm(velocity - reference)))
            active = np.asarray(result.diagnostics['active_linear_constraints'], dtype=int)
            wall_active.append(bool(np.any(active >= result.diagnostics['num_pair_constraints'])))
        else:
            velocity = np.asarray(actions[step])
        _, done, info = env.step(velocity, diagnose_stalls=False)
        commands.append(velocity.copy())
        states.append(env.positions.copy())
        observations.append(policy_observation_competence(env))
        min_wall_clearance = min(min_wall_clearance, info['min_swept_wall_clearance'])
        if done:
            break
    termination=env.termination if env.termination!='running' else 'generation_horizon'
    return dict(success=termination == 'success' and not env.collided,
                termination=termination, states=np.asarray(states),
                observations=np.asarray(observations), actions=np.asarray(commands),
                mean_projection_norm=float(np.mean(correction)) if correction else None,
                wall_active_fraction=float(np.mean(wall_active)) if wall_active else None,
                min_swept_wall_clearance=min_wall_clearance)


def collect(root, *, seed=314159, counts=None):
    counts = counts or {'train': 48, 'dev': 12, 'test': 12}
    if set(counts) != {'train', 'dev', 'test'} or min(counts.values()) <= 0:
        raise ValueError('positive train/dev/test base counts required')
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=True)
    # Match the reported Gap1 N=2 checkpoint/test geometry exactly.  The
    # scalable template's default 0.50 m door is a different benchmark.
    config = Config(num_agents=2, max_steps=2000, openings=(((0., .62),),))
    scenario = GapFlowScenario(config)
    rng = np.random.default_rng(seed)
    planned = []
    examples = []
    for split in ('train', 'dev', 'test'):
        for index in range(counts[split]):
            episode_seed = int(rng.integers(0, 2**31 - 1))
            episode_config = replace(config, seed=episode_seed, split=split)
            instance = build_instance(episode_config)
            for mode in ('solo', 'one_way'):
                p, g = task_state(instance, mode)
                for mirror in (False, True):
                    for swap in (False, True):
                        pp, gg = transform(p, g, mirror=mirror, swap=swap)
                        examples.append((split, index, episode_config, mode, mirror, swap, pp, gg))
                        planned.append(dict(
                            state_uid=uid('state', {'physical_fingerprint':config.physical_fingerprint,
                                'positions':pp.tolist(), 'goals':gg.tolist(),
                                'episode_seed':episode_seed, 'split':split}),
                            eta_uid=eta_identity((0., 0., 0.))[0],
                            controller_uid=uid('ctl', {'controller':'gate_waypoint_v2_with_accepted_hard_safety',
                                'mode':mode, 'mirror':mirror, 'swap':swap}),
                            seed_keys=[str(episode_seed)]))
    (root / 'planned_rollouts.json').write_text(json.dumps({'requests':planned}, indent=2) + '\n')
    cache = preflight(root / 'planned_rollouts.json')
    (root / 'cache_preflight.json').write_text(json.dumps(cache, indent=2) + '\n')
    if cache['summary']['ambiguous'] or cache['summary']['genuinely_missing'] != len(planned):
        raise RuntimeError('preflight requires review/reuse before new rollouts')
    projector = CertifiedHardSafetyFilter(HardProjectionConfig())
    writer = DatasetWriter(root / 'dataset', scenario, scenario_config=asdict(config))
    report = {'schema':'gap1_competence_dataset_v1', 'seed':seed,
              'base_count_per_split':counts, 'counts':{}, 'teacher_failures':[],
              'preflight':cache['summary'], 'waypoint_version':'competence_v2',
              'teacher':'public_gate_waypoint_plus_existing_certified_hard_safety'}
    for split, index, episode_config, mode, mirror, swap, p, g in examples:
        result = rollout(episode_config, p, g, projector)
        if not result['success']:
            report['teacher_failures'].append(dict(split=split,index=index,mode=mode,
                mirror=mirror,swap=swap,termination=result['termination'],
                final_goal_distance=np.linalg.norm(result['states'][-1]-g,axis=1).tolist()))
            continue
        direction = 'RL' if mirror else 'LR'
        name = f'{split}_nav_{index:04d}_{mode}_{direction}_swap{int(swap)}'
        writer.add(Trajectory(name, split, 'nominal',
            {'positions':p, 'goals':g, 'episode_seed':episode_config.seed,
             'split':split, 'mode':mode, 'direction':direction},
            result['states'], result['observations'], result['actions'],
            {'success':True, 'terminal_reason':'success',
             'teacher':'public_gap_waypoint_v2_plus_accepted_hard_safety',
             'mode':mode, 'direction':direction, 'mirror':mirror, 'swap':swap,
             'episode_seed':episode_config.seed,
             'mean_projection_norm':result['mean_projection_norm'],
             'wall_active_fraction':result['wall_active_fraction'],
             'min_swept_wall_clearance':result['min_swept_wall_clearance']}))
        key = f'{split}_{mode}_{direction}'
        bucket = report['counts'].setdefault(key, {'trajectories':0,'agent_transitions':0})
        bucket['trajectories'] += 1
        bucket['agent_transitions'] += result['actions'].shape[0] * (1 if mode == 'solo' else 2)
        if (index + 1) % 12 == 0 and mirror and swap and mode == 'one_way':
            print(json.dumps({'split':split,'bases_done':index+1,'report_counts':report['counts']}),flush=True)
    # Do not conceal failures by accepting only easy directions or seeds.
    for split in counts:
        for mode in ('solo','one_way'):
            left=report['counts'].get(f'{split}_{mode}_LR',{})
            right=report['counts'].get(f'{split}_{mode}_RL',{})
            if left != right:
                raise RuntimeError(f'unbalanced accepted data {split} {mode}: {left} vs {right}')
    writer.finalize(extra_report=report)
    (root / 'collection_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=314159)
    parser.add_argument('--train',type=int,default=48)
    parser.add_argument('--dev',type=int,default=12)
    parser.add_argument('--test',type=int,default=12)
    args=parser.parse_args()
    report=collect(args.output,seed=args.seed,counts={'train':args.train,'dev':args.dev,'test':args.test})
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
