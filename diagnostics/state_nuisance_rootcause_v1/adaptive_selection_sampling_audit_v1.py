"""Outcome-blind audit of the frozen global physical-state sample."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from ring_exchange.environment import LocalFrameConfig, sample_instance
from shared_rollout_db.src.rollout_db import canonical

OUT = HERE / 'heldout_adaptive_selection_v1'
PRIOR = HERE / 'real_state_expansion'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(values):
    return dict(min=float(np.min(values)), mean=float(np.mean(values)),
                sd=float(np.std(values)), max=float(np.max(values)))


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    earlier = json.loads((PRIOR / 'protocol.json').read_text())
    assert sha(ROOT / 'ring_exchange/environment.py') == protocol['generator_sha256']
    assert protocol['environment'] == earlier['environment']
    assert protocol['controller'] == earlier['controller']
    assert protocol['eta'] == earlier['eta']
    states = protocol['states']
    assert len(states) == 256
    seeds = [s['metadata']['draw_seed'] for s in states]
    assert seeds == list(range(seeds[0], seeds[0] + 256))
    assert len(set(s['state_uid'] for s in states)) == 256
    assert len(set(s['rng_namespace'] for s in states)) == 256
    assert not {s['state_uid'] for s in states}.intersection(
        s['state_uid'] for s in earlier['states'])
    config = LocalFrameConfig(**protocol['environment']['config']['config'])
    position, velocity, goal, global_angle = [], [], [], []
    for state, draw in zip(states, seeds):
        instance = sample_instance(draw, 'development', config)
        expected = dict(positions=instance.positions.tolist(),
                        velocities=instance.velocities.tolist(),
                        goals=instance.goals.tolist(), timestep=0)
        assert canonical(state['physical']) == canonical(expected)
        assert state['metadata']['source'] == 'unchanged native global development generator'
        position.append(instance.positions)
        velocity.append(instance.velocities)
        goal.append(instance.goals)
        global_angle.append(instance.global_angle)
    position, velocity, goal, global_angle = map(np.asarray,
                                                 (position, velocity, goal, global_angle))
    positions_radius = np.linalg.norm(position, axis=-1)
    goals_radius = np.linalg.norm(goal, axis=-1)
    speed = np.linalg.norm(velocity, axis=-1)
    audit = dict(state_count=256, independent_global_generator_draws=True,
                 generator_split='development', seed_first=seeds[0], seed_last=seeds[-1],
                 contiguous_unfiltered_seeds=True, all_states_regenerated_exactly=True,
                 no_overlap_with_earlier_608=True, no_state_or_seed_duplicates=True,
                 original_environment_unchanged=True, original_controller_unchanged=True,
                 original_eta_bank_unchanged=True,
                 start_radius=summary(positions_radius), goal_radius=summary(goals_radius),
                 initial_speed=summary(speed), global_angle=summary(global_angle),
                 protocol_sha256=sha(OUT / 'protocol.json'),
                 generator_sha256=sha(ROOT / 'ring_exchange/environment.py'))
    (OUT / 'sampling_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
