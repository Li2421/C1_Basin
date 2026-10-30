"""Evaluate a deliberately labelled Flow-guided sequential route supervisor.

The trained joint Flow policy ranks which robot gets the next turn. Its actual
velocity is used when it makes safe progress inside a narrow route corridor;
otherwise an A* path follower supplies the velocity. This is a hybrid baseline,
never raw Flow policy success or evidence that Flow alone resolved a deadlock.
"""
from __future__ import annotations

from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path

import jax
import numpy as np
from single_integrator.environment import point_segment_distance, segment_distance

from new_benchmark_common.dev_closed_loop import load_nominal_cases
from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions
from shared_rollout_db.src.planner import preflight
from shared_rollout_db.src.rollout_db import eta_identity, uid

from .environment import BottleneckEnv
from .expert import SequentialExpert
from .observation import policy_observation
from .scenario import Config


def _proposal(agent, env, key):
    observation = (policy_observation(env) if agent.config['obs_dim'] == 8 else
                   env.observation()['agents'].astype(np.float32))
    action = np.asarray(sample_bounded_actions(agent, observation[None], key)[0], dtype=float)
    norms = np.linalg.norm(action, axis=1, keepdims=True)
    return action * np.minimum(1., env.config.max_speed/np.maximum(norms, 1e-30))


def _safe_step(env, agent_index, velocity, *, reserve=1e-5):
    """Exact one-moving-disk swept check against stationary peers and walls."""
    p = env.positions[agent_index]
    q = p + env.config.dt*velocity
    other = np.delete(env.positions, agent_index, axis=0)
    geometry = env.instance.geometry
    wall_clearance = segment_distance(p, q, env.walls[:, 0], env.walls[:, 1]).min() - geometry.margin
    peer_clearance = point_segment_distance(other, p, q).min() - (
        2*env.config.agent_radius + env.config.agent_collision_margin)
    return bool(wall_clearance > reserve and peer_clearance > reserve and
                geometry.valid_points(q[None])[0])


def run(dataset, checkpoint, output, *, seed=0, max_cases=None,
        split='dev', final_evaluation=False):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest, cases, audit = load_nominal_cases(dataset,split=split,
                                                final_evaluation=final_evaluation)
    if max_cases is not None:
        cases = cases[:max_cases]
    config = Config(**manifest['scenario_config'])
    agent, _ = load_checkpoint(checkpoint,
                               expected_environment_fingerprint=manifest['environment_fingerprint'])
    if agent.config['num_agents'] != config.num_agents:
        raise ValueError('checkpoint/plant N mismatch')
    digest = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    controller_uid = uid('ctl', {'flow_sha256': digest,
                                 'protocol': 'flow_guided_sequential_route_v4'})
    requests = []
    for case in cases:
        state = case.initial_state
        state_uid = uid('state', {'physical_fingerprint': config.physical_fingerprint,
                                  'state': state})
        requests.append(dict(state_uid=state_uid, eta_uid=eta_identity((0., 0., 0.))[0],
                             controller_uid=controller_uid, seed_keys=[str(seed)]))
    plan = output / 'planned_rollouts.json'
    plan.write_text(json.dumps({'requests': requests}, indent=2) + '\n')
    cache = preflight(plan)
    (output / 'cache_preflight.json').write_text(json.dumps(cache, indent=2) + '\n')
    if cache['summary']['ambiguous']:
        raise RuntimeError('ambiguous cache evidence')
    rows = []
    for case_index, case in enumerate(cases):
        state = case.initial_state
        env = BottleneckEnv(replace(config, seed=state['episode_seed'], split=split))
        env.reset(np.asarray(state['positions']), np.asarray(state['goals']))
        initial_hash = env.initial_state_sha256()
        planner = SequentialExpert()
        remaining = set(range(config.num_agents))
        positions = [env.positions.copy()]
        clearances = []
        selected_agents = []
        flow_direct_steps = 0
        counter = 0
        terminal = 'running'
        episode_key = jax.random.fold_in(jax.random.PRNGKey(seed), case_index)
        while remaining and not env.done:
            # Flow alone sets order; the feasibility filter can skip a robot
            # whose route is temporarily blocked by parked peers.
            proposal = _proposal(agent, env, jax.random.fold_in(episode_key, counter))
            ranked = sorted(remaining, key=lambda i: -float(np.linalg.norm(proposal[i])))
            chosen = None
            for index in ranked:
                route = planner.route(env, index)
                if route is not None:
                    chosen = index
                    break
            if chosen is None:
                terminal = 'no_sequential_route'
                break
            selected_agents.append(chosen)
            waypoints = [p.copy() for p in route[1:]]
            waypoint_index = 0
            while waypoint_index < len(waypoints):
                waypoint = waypoints[waypoint_index]
                segment_start = env.positions[chosen].copy()
                while np.linalg.norm(env.positions[chosen] - waypoint) > 1e-9:
                    if env.step_count >= config.max_steps or env.done:
                        terminal = 'timeout' if env.step_count >= config.max_steps else env.termination
                        break
                    delta = waypoint - env.positions[chosen]
                    direction = delta / np.linalg.norm(delta)
                    proposal = _proposal(agent, env,
                                         jax.random.fold_in(episode_key, counter + 1))
                    counter += 1
                    candidate = proposal[chosen]
                    norm = float(np.linalg.norm(candidate))
                    after = env.positions[chosen] + config.dt*candidate
                    safe = (norm >= .15 and np.linalg.norm(delta) > .2 and
                            float(np.dot(candidate, direction)) / max(norm, 1e-30) >= .96 and
                            np.linalg.norm(after-waypoint) < np.linalg.norm(delta) and
                            point_segment_distance(after, segment_start, waypoint) <= .01 and
                            _safe_step(env, chosen, candidate, reserve=.025))
                    velocity = np.zeros(env.action_shape)
                    if safe:
                        velocity[chosen] = candidate
                    else:
                        speed = min(config.max_speed, float(np.linalg.norm(delta))/config.dt)
                        velocity[chosen] = direction*speed
                    if not _safe_step(env, chosen, velocity[chosen]):
                        detour = planner._route_to(env, chosen, env.positions[chosen], waypoint)
                        if detour is None or len(detour) < 3:
                            terminal = 'blocked_motion'
                            break
                        waypoints[waypoint_index:waypoint_index] = [p.copy() for p in detour[1:-1]]
                        break
                    _, done, info = env.step(velocity, diagnose_stalls=False)
                    flow_direct_steps += int(safe)
                    positions.append(env.positions.copy())
                    clearances.append(info['swept_clearance'])
                    terminal = info['termination']
                    if done:
                        break
                if env.done or terminal in ('blocked_motion', 'timeout'):
                    break
                if np.linalg.norm(env.positions[chosen] - waypoint) <= 1e-9:
                    waypoint_index += 1
            if terminal in ('blocked_motion', 'timeout'):
                break
            remaining.remove(chosen)
        if terminal == 'running':
            terminal = 'success' if env.done and env.termination == 'success' else 'timeout'
        row = dict(rollout_id=case.rollout_id, termination=terminal,
                   full_task_success=terminal == 'success', collision=env.collided,
                   steps=env.step_count, flow_direct_steps=flow_direct_steps,
                   flow_direct_fraction=flow_direct_steps / max(1, env.step_count),
                   selected_agents=selected_agents,
                   minimum_swept_clearance=float(min(clearances)) if clearances else None)
        rows.append(row)
        print(json.dumps(row), flush=True)
        trace_dir = output / 'traces'
        trace_dir.mkdir(exist_ok=True)
        video_meta = dict(batch_id=output.name, scenario='bottleneck_family',
                          rollout_id=case.rollout_id,
                          controller='flow_guided_sequential_route_v4',
                          checkpoint=f'{checkpoint} sha256={digest}',
                          seed=state['episode_seed'], controller_rng_seed=seed,
                          source_record=str(case.archive_path),
                          initial_state_sha256=initial_hash, config=env.config.to_dict(),
                          start_step=0, episode_steps=env.step_count, complete=True,
                          termination=terminal, collision=bool(env.collided),
                          safety_enabled=True,
                          note='centralized A* supplies safety; not raw Flow policy success')
        np.savez_compressed(trace_dir / f'{case.rollout_id}.npz',
                            positions=np.asarray(positions), steps=np.arange(len(positions)),
                            goals=env.goals, walls=env.walls,
                            swept_clearance=np.asarray(clearances),
                            metadata_json=np.asarray(json.dumps(video_meta)))
    result = dict(schema='gap_flow_guided_route_eval_v5', split=split,
                  checkpoint_sha256=digest, opened_test_archives=audit['opened_test_archives'],
                  rollouts=rows,
                  aggregate=dict(count=len(rows),
                                 full_task_success=float(np.mean([r['full_task_success'] for r in rows])),
                                 collision_rate=float(np.mean([r['collision'] for r in rows])),
                                 flow_direct_fraction=float(np.mean([r['flow_direct_fraction'] for r in rows]))),
                  cache_preflight=cache['summary'])
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-cases', type=int)
    parser.add_argument('--split', choices=('dev','test'), default='dev')
    parser.add_argument('--final-evaluation', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.checkpoint, args.output,
                         seed=args.seed, max_cases=args.max_cases,split=args.split,
                         final_evaluation=args.final_evaluation)['aggregate'], indent=2))


if __name__ == '__main__':
    main()
