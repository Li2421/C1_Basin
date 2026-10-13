"""Outcome-blind physical-state sampling for the formal Q-only experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from shared_rollout_db.src.rollout_db import canonical


def read(path):
    return json.loads(Path(path).read_text())


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def write_frozen(path, obj):
    path = Path(path)
    if path.exists():
        assert read(path) == obj, f'frozen state file changed: {path}'
    else:
        path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def pack(scene, split, draw_seed, physical, alias, origin, metadata=None):
    assert physical['timestep'] == 0
    return dict(scene=scene, split=split, alias=alias, draw_seed=draw_seed,
                physical=physical, physical_hash=digest(physical),
                rng_namespace=int(hashlib.sha256(alias.encode()).hexdigest()[:8], 16),
                origin=origin, metadata=metadata or {})


def toy():
    old = read(ROOT / 'diagnostics/orthoflow3_tt_ff_matched_learnability_v1/protocol.json')
    prior = [s for s in old['states'] if s['scenario'] == 'toy_give_way']
    out = []
    for split, count, root in [('train', 512, 948100000),
                               ('validation', 32, 948200000),
                               ('test', 128, 948300000)]:
        existing = [] if split == 'test' else [s for s in prior if s['metadata']['split'] == split]
        for s in existing:
            out.append(pack('toy_give_way', split, s['metadata'].get('draw_seed'),
                            s['physical'], s['uid'], 'exact_prior_matched_state',
                            {'prior_state_uid': s['state_uid']}))
        for j in range(count - len(existing)):
            seed = root + j
            rng = np.random.default_rng(seed)
            x = rng.uniform(.55, 1.05, 2)
            y = rng.uniform(-.025, .025, 2)
            pos = np.array([[-x[0], y[0]], [x[1], y[1]]], np.float32).astype(float)
            physical = dict(positions=pos.tolist(), velocities=[[0., 0.], [0., 0.]],
                            goals=[[1.09, 0.], [-1.09, 0.]], timestep=0)
            out.append(pack('toy_give_way', split, seed, physical,
                            f'formal_q_v1_toy_{split}_{seed}',
                            'unchanged_WIDE_IC_uniform'))
    return out


def ring():
    from ring_exchange.environment import sample_instance, LocalFrameConfig
    old = read(ROOT / 'diagnostics/state_nuisance_rootcause_v1/real_state_expansion/protocol.json')
    cfg = LocalFrameConfig(**old['environment']['config']['config'])
    out = []
    for split, count in [('train', 512), ('validation', 32)]:
        existing = [s for s in old['states'] if s['metadata']['split'] == split]
        assert len(existing) == count
        for s in existing:
            out.append(pack('ring_exchange', split, s['metadata'].get('draw_seed'),
                            s['physical'], s['uid'], 'exact_prior_512_state_expansion',
                            {'prior_state_uid': s['state_uid']}))
    for j in range(128):
        seed = 948400000 + j
        ins = sample_instance(seed, 'development', cfg)
        physical = dict(positions=ins.positions.tolist(), velocities=ins.velocities.tolist(),
                        goals=ins.goals.tolist(), timestep=0)
        out.append(pack('ring_exchange', 'test', seed, physical,
                        f'formal_q_v1_ring_test_{seed}',
                        'unchanged_native_development_sampler'))
    return out


def four():
    from four_way_intersection.environment import Config
    from four_way_intersection.scenario import sample_initial_state, build_layout
    manifest = read(ROOT / 'diagnostics/four_way_intersection_stage1/base_u_v13_broad_global_dataset/manifest.json')
    cfg = Config(**manifest['scenario_config'])
    goals = build_layout(cfg).goals.tolist()
    out = []
    for split, count, native_split in [('train', 512, 'train'),
                                       ('validation', 32, 'development'),
                                       ('test', 128, 'test')]:
        for j in range(count):
            positions, velocities, metadata = sample_initial_state(
                cfg, native_split, j, base_seed=948500000)
            physical = dict(positions=positions.tolist(), velocities=velocities.tolist(),
                            goals=goals, timestep=0)
            out.append(pack('four_way_intersection', split, j, physical,
                            f'formal_q_v1_four_{split}_{j}',
                            'unchanged_native_global_sampler', metadata))
    return out


def main(scene):
    states = {'toy_give_way': toy, 'ring_exchange': ring,
              'four_way_intersection': four}[scene]()
    counts = {split: sum(s['split'] == split for s in states)
              for split in ('train', 'validation', 'test')}
    assert counts == {'train': 512, 'validation': 32, 'test': 128}
    assert len(states) == len({s['physical_hash'] for s in states}) == 672
    if scene == 'ring_exchange':
        prior = read(ROOT / 'diagnostics/state_nuisance_rootcause_v1/heldout_adaptive_selection_v1/protocol.json')
        old_hashes = {digest(s['physical']) for s in prior['states']}
        assert not old_hashes & {s['physical_hash'] for s in states if s['split'] == 'test'}
    if scene == 'toy_give_way':
        old = read(ROOT / 'diagnostics/orthoflow3_tt_ff_matched_learnability_v1/protocol.json')
        old_hashes = {digest(s['physical']) for s in old['states'] if s['scenario'] == scene}
        assert not old_hashes & {s['physical_hash'] for s in states if s['split'] == 'test'}
    write_frozen(HERE / f'states_{scene}.json', dict(schema='formal_q_physical_states_v1',
                 scene=scene, counts=counts, states=states, all_states_selected_outcome_blind=True,
                 test_prior_overlap=0, sample_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(dict(scene=scene, counts=counts, unique_physical_states=len(states))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', choices=('toy_give_way', 'ring_exchange', 'four_way_intersection'), required=True)
    args = parser.parse_args()
    main(args.scene)
