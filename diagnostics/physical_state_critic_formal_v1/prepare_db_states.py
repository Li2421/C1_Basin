"""Extend the unchanged DB accepted global initial-state sampler outcome-blind."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from double_bottleneck.environment import Config
from double_bottleneck.scenario import INITIAL_REGIMES, build_layout
from diagnostics.double_bottleneck_initial_state_coverage.tools.generate_pools import _sample_spec, _solve_spec, _spec_json
from diagnostics.physical_state_critic_formal_v1.prepare_physical_states import pack, read, write_frozen

COUNTS = {'train': (171, 171, 170), 'validation': (11, 11, 10),
          'test': (43, 43, 42)}
ROOT_SEED = 'formal_q_db_global_v1|948600000'


def seed(split, regime):
    digest = hashlib.sha256(f'{ROOT_SEED}|{split}|{regime}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little')


def worker(split, regime_index):
    regime = INITIAL_REGIMES[regime_index]
    goal = COUNTS[split][regime_index]
    dest = HERE / f'db_states_{split}_{regime_index}.json'
    if dest.exists():
        result = read(dest)
        assert len(result['states']) == goal
        return result['audit']
    rng = np.random.default_rng(seed(split, regime))
    goals = build_layout(Config()).goals.tolist()
    accepted, rejected = [], []
    k = 0
    started = time.monotonic()
    while len(accepted) < goal:
        spec = _sample_spec(split, regime, k, rng)
        result = _solve_spec((split, 'train' if split == 'train' else 'val', spec))
        if result['accepted']:
            physical = dict(positions=spec.positions.tolist(),
                            velocities=spec.initial_velocities.tolist(),
                            goals=goals, timestep=0)
            accepted.append(pack('double_bottleneck', split, k, physical,
                                 f'formal_q_v1_db_{split}_{regime}_{k}',
                                 'unchanged_accepted_global_initial_state_sampler',
                                 {'regime':regime, 'original_sampler_spec':_spec_json(spec)}))
        else:
            rejected.append(dict(candidate_index=k, reason=result['reason']))
        k += 1
        if k % 20 == 0:
            print(json.dumps(dict(split=split, regime=regime,
                                  accepted=len(accepted), candidates=k,
                                  seconds=time.monotonic()-started)), flush=True)
        assert k <= 4096, 'Original maximum draw count exceeded'
    audit = dict(split=split, regime=regime, required=goal, accepted=len(accepted),
                 candidates_drawn=k, rejections=rejected, rng_seed=seed(split, regime),
                 sampler_source='diagnostics/double_bottleneck_initial_state_coverage/tools/generate_pools.py',
                 expert_acceptance_unchanged=True, outcome_or_critic_filtering=False,
                 elapsed_seconds=time.monotonic()-started)
    write_frozen(dest, dict(schema='formal_q_db_sampler_v1', states=accepted,
                            audit=audit,
                            sampler_sha256=hashlib.sha256(Path(sys.modules[_sample_spec.__module__].__file__).read_bytes()).hexdigest()))
    return audit


def collect():
    chunks = [read(HERE / f'db_states_{split}_{regime_index}.json')
              for split in ('train', 'validation', 'test')
              for regime_index in range(3)]
    states = [s for part in chunks for s in part['states']]
    counts = {split: sum(s['split'] == split for s in states)
              for split in ('train', 'validation', 'test')}
    assert counts == {'train':512,'validation':32,'test':128}
    assert len(states) == len({s['physical_hash'] for s in states}) == 672
    write_frozen(HERE / 'states_double_bottleneck.json',
                 dict(schema='formal_q_physical_states_v1',scene='double_bottleneck',
                      counts=counts,states=states,
                      sampler_audits=[x['audit'] for x in chunks],
                      all_states_selected_outcome_blind=True,
                      sample_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(dict(scene='double_bottleneck', counts=counts,
                          unique_physical_states=len(states))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('worker', 'collect'))
    parser.add_argument('--split', choices=tuple(COUNTS))
    parser.add_argument('--regime-index', type=int)
    args = parser.parse_args()
    if args.action == 'worker':
        assert args.split and args.regime_index in range(3)
        print(json.dumps(worker(args.split, args.regime_index)), flush=True)
    else:
        collect()
