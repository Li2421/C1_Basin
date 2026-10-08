"""Balanced non-opposing Gap1 demonstrations for separate large-N Flow models.

The data-generation reference releases same-direction agents into the gate
after the previous entrant has exited. Earlier entrants remain free to
correct goal error. This is ordinary one-way throughput supervision: no
opposing groups, side priority, eta, or runtime release module is supplied
to the learned Flow controller.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from diagnostics.double_bottleneck_eta3_basin.tools.exact_projection_retry import CertifiedHardSafetyFilter
from new_benchmark_common.dataset import DatasetWriter, JointTransitionDataset, Trajectory
from shared_control.hard_projection import HardProjectionConfig
from shared_rollout_db.src.rollout_db import eta_identity, uid
from shared_rollout_db.src.planner import preflight
from .collect_competence import teacher_reference
from .environment import BottleneckEnv
from .flow_dataset import GapFlowScenario
from .observation import policy_observation_competence
from .scenario import Config, build_instance


def one_way_state(instance, mode: str, index: int):
    """Return a left-to-right task; all opposite-side agents are passive or relocated."""
    p = instance.positions.copy()
    g = p.copy()
    left = np.arange(0, instance.config.num_agents, 2)
    if mode == 'solo':
        moving = left[[index % len(left)]]
    elif mode == 'five':
        moving = left[(np.arange(min(5, len(left))) + index) % len(left)]
    elif mode == 'half':
        moving = left
    elif mode == 'parked_half':
        # Non-opposing second-stage navigation: the other group is already
        # stationary at its destination, with no release or priority signal.
        # This covers the goal-side occupancy absent from ordinary half mode.
        p[1::2] = instance.goals[1::2]
        g[1::2] = p[1::2]
        moving = left
    elif mode == 'full':
        p[1::2] = instance.goals[1::2]
        moving = np.arange(instance.config.num_agents)
    else:
        raise ValueError(mode)
    if mode == 'full':
        g[0::2] = instance.goals[0::2]
        g[1::2] = instance.positions[1::2]
    else:
        g[moving] = instance.goals[moving]
    if mode == 'solo' and np.sum(np.linalg.norm(g - p, axis=1) >
                                 instance.config.goal_tolerance) != 1:
        raise RuntimeError('solo task must have exactly one initially active agent')
    return p, g, np.asarray(moving, dtype=int)


def pipeline_rollout(config: Config, positions, goals, moving, projector, *, exit_x=.8):
    env = BottleneckEnv(config)
    env.reset(positions, goals)
    order = np.asarray(sorted(moving, key=lambda i: -positions[i, 0]), dtype=int)
    released = 1
    states = [env.positions.copy()]
    observations = [policy_observation_competence(env)]
    actions = []
    wall_clearance = float('inf')
    pair_clearance = float('inf')
    projection_norm = []
    simultaneous = []
    for _ in range(config.max_steps):
        if released < len(order) and env.positions[order[released - 1], 0] > exit_x:
            released += 1
        reference = teacher_reference(env)
        reference[order[released:]] = 0
        simultaneous.append(int(np.sum(np.linalg.norm(reference, axis=1) > .1)))
        result = projector(env.snapshot(), reference)
        safe = np.asarray(result.velocity, dtype=np.float64)
        projection_norm.append(float(np.linalg.norm(safe - reference)))
        _, done, info = env.step(safe, diagnose_stalls=False)
        actions.append(safe.copy())
        states.append(env.positions.copy())
        observations.append(policy_observation_competence(env))
        wall_clearance = min(wall_clearance, info['min_swept_wall_clearance'])
        pair_clearance = min(pair_clearance, info['min_swept_agent_clearance'])
        if done:
            break
    return dict(success=env.termination == 'success' and not env.collided,
                termination=env.termination, states=np.asarray(states),
                observations=np.asarray(observations), actions=np.asarray(actions),
                min_swept_wall_clearance=wall_clearance,
                min_swept_agent_clearance=pair_clearance,
                mean_projection_norm=float(np.mean(projection_norm)),
                max_simultaneously_moving=max(simultaneous),
                mean_simultaneously_moving=float(np.mean(simultaneous)),
                released=released)


def transformed_replay(config, p, g, result, *, mirror, permutation):
    pp = p[permutation].copy()
    gg = g[permutation].copy()
    actions = result['actions'][:, permutation].copy()
    if mirror:
        pp[:, 0] *= -1
        gg[:, 0] *= -1
        actions[:, :, 0] *= -1
    env = BottleneckEnv(config)
    env.reset(pp, gg)
    states = [env.positions.copy()]
    observations = [policy_observation_competence(env)]
    min_clearance = float('inf')
    for action in actions:
        _, done, info = env.step(action, diagnose_stalls=False)
        states.append(env.positions.copy())
        observations.append(policy_observation_competence(env))
        min_clearance = min(min_clearance, info['swept_clearance'])
        if done:
            break
    if env.termination != 'success' or len(states) != len(actions) + 1 or min_clearance <= 0:
        raise RuntimeError('mirrored/permuted full-simulator replay failed')
    return pp, gg, np.asarray(states), np.asarray(observations), actions, min_clearance


def collect(output: Path, *, agents: int, counts: dict[str, int], modes: tuple[str, ...],
            permutations: int = 2, seed: int = 701027, source_dataset: Path | None = None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    if agents not in (10, 20, 50) or permutations < 1:
        raise ValueError('supported sizes are N=10, N=20 and N=50; permutations must be positive')
    if agents == 20:
        # The N=20 replacement keeps the same Gap1 plant and 0.62 m opening.
        # No archived N=20 model or dataset exists to import.
        config = Config(num_agents=20, max_steps=16000,
                        openings=(((0., .62),),))
    else:
        original = Path(f'diagnostics/gap_flow_v1/n{agents}_wide_recovery/dataset/manifest.json')
        config = Config(**json.loads(original.read_text())['scenario_config'])
    if config.openings[0][0][1] != .62:
        raise ValueError('exact original Gap1 0.62 m geometry required')
    rng = np.random.default_rng(seed)
    cases = []
    planned = []
    for split in ('train', 'dev'):
        for index in range(counts[split]):
            episode_seed = int(rng.integers(0, 2**31 - 1))
            episode_config = replace(config, seed=episode_seed, split=split)
            instance = build_instance(episode_config)
            for mode in modes:
                p, g, moving = one_way_state(instance, mode, index)
                cases.append((split, index, episode_config, mode, p, g, moving))
                planned.append(dict(state_uid=uid('state', {
                    'physical_fingerprint':config.physical_fingerprint,
                    'positions':p.tolist(), 'goals':g.tolist(),
                    'episode_seed':episode_seed, 'split':split}),
                    eta_uid=eta_identity((0., 0., 0.))[0],
                    controller_uid=uid('ctl', {'controller':'same_direction_gate_pipeline_teacher_v1',
                        'mode':mode, 'exit_x':.8, 'horizon':config.max_steps}),
                    seed_keys=[str(episode_seed)]))
    plan = output / 'planned_rollouts.json'
    plan.write_text(json.dumps({'requests':planned}, indent=2) + '\n')
    cache = preflight(plan)
    (output / 'cache_preflight.json').write_text(json.dumps(cache, indent=2) + '\n')
    if cache['summary']['ambiguous'] or cache['summary']['genuinely_missing'] != len(planned):
        raise RuntimeError('preflight requires cache review/reuse before collection')
    writer = DatasetWriter(output / 'dataset', GapFlowScenario(config), scenario_config=asdict(config))
    if source_dataset is not None:
        source_dataset = Path(source_dataset)
        source_manifest = json.loads((source_dataset / 'manifest.json').read_text())
        for split in ('train', 'dev', 'test'):
            if source_manifest['counts']['split'].get(split, 0) == 0:
                continue
            data = JointTransitionDataset(source_dataset, split)
            if data.environment_fingerprint != config.physical_fingerprint:
                raise ValueError('source dataset geometry mismatch')
            for trajectory in data.trajectories:
                writer.add(trajectory)
    projector = CertifiedHardSafetyFilter(HardProjectionConfig())
    report = {'schema':'gap1_scale_competence_collection_v1', 'agents':agents,
              'counts_requested':counts, 'modes':modes, 'permutations_per_direction':permutations,
              'teacher_failures':[], 'accepted':{}, 'preflight':cache['summary'],
              'source_dataset':str(source_dataset) if source_dataset else None,
              'reference':'same_direction_gate_release_after_exit; no opposing tasks'}
    for case_index, (split, index, episode_config, mode, p, g, moving) in enumerate(cases):
        result = pipeline_rollout(episode_config, p, g, moving, projector)
        if not result['success']:
            report['teacher_failures'].append(dict(split=split, index=index, mode=mode,
                termination=result['termination'], steps=len(result['actions']),
                released=result['released'],
                final_goal_distance=np.linalg.norm(result['states'][-1] - g, axis=1).tolist()))
            print(json.dumps({'case':case_index,'mode':mode,'teacher_failure':True}),flush=True)
            continue
        for perm_index in range(permutations):
            permutation = (np.arange(agents) if perm_index == 0 else
                           np.random.default_rng(np.random.SeedSequence(
                               [seed, episode_config.seed, perm_index])).permutation(agents))
            for mirror in (False, True):
                pp, gg, states, obs, actions, clearance = transformed_replay(
                    episode_config, p, g, result, mirror=mirror, permutation=permutation)
                direction = 'RL' if mirror else 'LR'
                # A collection that imports an earlier dataset must use a
                # distinct rollout ID for its new seeds and perturbations.
                namespace = f's{seed}_' if source_dataset is not None else ''
                name = f'{split}_scale_{namespace}{index:04d}_{mode}_{direction}_perm{perm_index}'
                writer.add(Trajectory(name, split, 'nominal',
                    {'positions':pp, 'goals':gg, 'episode_seed':episode_config.seed,
                     'split':split, 'mode':mode, 'direction':direction,
                     'moving_agents':np.flatnonzero(np.isin(permutation, moving)).tolist()},
                    states, obs, actions,
                    {'success':True, 'terminal_reason':'success', 'mode':mode,
                     'direction':direction, 'permutation':permutation.tolist(),
                     'teacher':'same_direction_gate_pipeline_v1_plus_accepted_hard_safety',
                     'max_simultaneously_moving':result['max_simultaneously_moving'],
                     'mean_simultaneously_moving':result['mean_simultaneously_moving'],
                     'mean_projection_norm':result['mean_projection_norm'],
                     'min_swept_clearance':clearance}))
                key = f'{split}_{mode}_{direction}'
                bucket = report['accepted'].setdefault(key, {'trajectories':0,'transitions':0})
                bucket['trajectories'] += 1
                bucket['transitions'] += len(actions)
        print(json.dumps({'case':case_index + 1, 'mode':mode, 'steps':len(result['actions']),
                          'max_moving':result['max_simultaneously_moving']}),flush=True)
    for split in ('train','dev'):
        for mode in modes:
            if report['accepted'].get(f'{split}_{mode}_LR') != report['accepted'].get(f'{split}_{mode}_RL'):
                raise RuntimeError(f'directional imbalance: {split} {mode}')
    writer.finalize(extra_report=report)
    (output / 'collection_report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agents', type=int, choices=(10,20,50), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train', type=int, required=True)
    parser.add_argument('--dev', type=int, required=True)
    parser.add_argument('--modes', nargs='+', choices=('solo','five','half','parked_half','full'), default=('full',))
    parser.add_argument('--permutations', type=int, default=2)
    parser.add_argument('--seed', type=int, default=701027)
    parser.add_argument('--source-dataset', type=Path)
    args = parser.parse_args()
    report = collect(args.output, agents=args.agents,
                     counts={'train':args.train,'dev':args.dev}, modes=tuple(args.modes),
                     permutations=args.permutations, seed=args.seed,
                     source_dataset=args.source_dataset)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
