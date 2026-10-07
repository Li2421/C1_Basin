#!/usr/bin/env python3
"""Freeze source-isolated true-t0 orders and reconstruct exact deployment states."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path('/home/zhihan/research/Basin_C1')
SYSROOT = Path('/home/zhihan/research/02_C1_Toy_GiveWay')
HERE = ROOT/'diagnostics/orthoflow3_true_t0_point_learning_v1'
FRESH = ROOT/'diagnostics/orthoflow3_basin_margin_learning_v1/fresh_wide_manifest.json'
PRIOR = ROOT/'diagnostics/orthoflow3_t0_pact_training_readiness_v1'
T0 = ROOT/'diagnostics/orthoflow3_t0_basin_structure_v1'
COMP = ROOT/'diagnostics/orthoflow3_t0_basin_completion_v1'
BASIS = ROOT/'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py'
SOURCE = ROOT/'diagnostics/orthoflow3_continuous_inner_ball_pilot6_v1/continuous_pilot6.py'
FUTURE_ROOT = 2026092811
ORDER_SALT = 'ORTHOFLOW3_TRUE_T0_POINT_LEARNING_V1_ORDER'

for p in (SYSROOT, ROOT):
    sys.path.insert(0, str(p))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def load_geometry_module():
    spec = importlib.util.spec_from_file_location('point_geom_source', SOURCE)
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def split_and_rank(source_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(f'{ORDER_SALT}|{source_id}'.encode()).hexdigest()
    bucket = int(digest[:8], 16) % 5
    split = 'train' if bucket < 3 else ('val' if bucket == 3 else 'test')
    return split, digest


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    for d in ('states', 'state_runs', 'logs'):
        (HERE/d).mkdir(exist_ok=True)
    if sha(BASIS) != '51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38':
        raise RuntimeError('authoritative OrthoFlow3 hash mismatch')

    fresh = json.load(open(FRESH))
    prior_rows = list(csv.DictReader(open(PRIOR/'point_dataset_candidates.csv')))
    existing = {r['source_group'] for r in prior_rows if r['point_usable'] == 'True'}
    if len(existing) != 8:
        raise RuntimeError(f'expected eight frozen usable source groups, got {len(existing)}')

    orders = {'train': [], 'val': [], 'test': []}
    for ep in fresh['episodes']:
        if ep['source_id'] in existing:
            continue
        split, digest = split_and_rank(ep['source_id'])
        orders[split].append((digest, ep))
    for split in orders:
        orders[split].sort(key=lambda x: (x[0], x[1]['source_id']))
    if min(map(len, orders.values())) < 8:
        raise RuntimeError('deterministic split unexpectedly too small')

    frozen = {
        'schema': 'orthoflow3_true_t0_point_learning_v1_frozen_orders',
        'frozen_before_new_eta_outcomes': True,
        'assignment': 'sha256(salt|source_id) modulo 5: 0-2 train, 3 val, 4 test; within-split lexical digest order',
        'salt': ORDER_SALT,
        'existing_counts': {'train': 5, 'val': 1, 'test': 2},
        'needed_additions': {'train': 19, 'val': 7, 'test': 6},
        'orders': {},
    }
    for split, vals in orders.items():
        frozen['orders'][split] = [
            {'rank': i, 'digest': d, 'source_id': ep['source_id'],
             'episode_index': int(ep['episode_index']), 'rollout_id': int(ep['rollout_id'])}
            for i, (d, ep) in enumerate(vals)
        ]
    dump(HERE/'frozen_state_orders.json', frozen)

    # Reconstruct every eligible t0 state now, without querying any eta outcome.
    from diagnostics.gphi_training_dataset_v2.build_states import save_full
    from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
    from diagnostics.single_segment_recovery_training_v1.state_machine import FiniteHistoryView
    from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
    from single_integrator.cbf import CBFConfig, barrier_constraints
    from single_integrator.environment import Config, GiveWayEnv, bounded_nominal
    from single_integrator.evaluate import load_policy

    integ = json.load(open(ROOT/'diagnostics/gphi_fixed_d_eta_predictor_v1/integrity_audit.json'))
    config = Config(**integ['environment']); cbf = CBFConfig(**integ['cbf'])
    policy, _ = load_policy(SYSROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')
    sample = jax.jit(lambda obs, key: policy.sample_actions(obs[None], seed=key)[0])
    flow_seed = int(fresh['flow_randomness']['root_seed'])
    ep_by_source = {x['source_id']: x for x in fresh['episodes']}
    features = []; states = []
    for split in ('train', 'val', 'test'):
        for item in frozen['orders'][split]:
            ep = ep_by_source[item['source_id']]
            sid = f'T0_POINT_{split}_{item["rank"]:03d}_ep{int(ep["episode_index"]):04d}'
            env = GiveWayEnv(config); env.reset(np.asarray(ep['initial_positions'], float))
            state_path = HERE/'states'/f'{sid}.npz'
            if not state_path.exists():
                save_full(state_path, env)
            rng = int(ep['rollout_id'])
            base = jax.random.fold_in(jax.random.PRNGKey(flow_seed), rng)
            current_key = jax.random.fold_in(base, 0)
            act = np.asarray(sample(jnp.asarray(env.observation(), dtype=jnp.float32), current_key))
            flow = bounded_nominal(act, config.max_speed)
            A, lower, _ = barrier_constraints(env.snapshot(), cbf)
            safe, _, _, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
            h, _ = StartupAwareFeatureBuilder().build(
                FiniteHistoryView(env), {'u_flow': flow, 'u_safe': safe}, config, cbf)
            h = np.asarray(h, dtype=np.float64)
            if h.shape != (214,):
                raise RuntimeError((sid, h.shape))
            feature_index = len(features); features.append(h)
            states.append({
                'state_id': sid, 'source_group': ep['source_id'], 'source_trajectory': ep['source_id'],
                'split': split, 'split_rank': int(item['rank']), 'absolute_step': 0, 'true_t0': True,
                'initial_positions': ep['initial_positions'], 'fresh_episode_index': int(ep['episode_index']),
                'fresh_rollout_id': rng, 'state_file': str(state_path), 'state_sha256': sha(state_path),
                'feature_index': feature_index, 'feature_sha256': hashlib.sha256(h.tobytes()).hexdigest(),
                'flow_seed': flow_seed, 'rng_namespace': rng,
                'h_conditioning_identifier': f'{sid}__flow{flow_seed}__episode{rng}',
                'current_flow_key_semantics': f'fold_in(fold_in(PRNGKey({flow_seed}),{rng}),0)',
                'future_root_seed': FUTURE_ROOT,
            })
    np.savez_compressed(HERE/'conditioning_features.npz', features=np.asarray(features))
    dump(HERE/'eligible_state_manifest.json', {'selected_states': states})

    # Freeze the exact 64-point global Sobol sequence and E_bridge geometry.
    mod = load_geometry_module()
    _, affine, scale, _, _, _, _, _, eq = mod.geometry()
    cloud = mod.sobol_bridge(eq, affine, scale, 64)
    write_csv(HERE/'global_eta_sobol64.csv', cloud)
    for name in ('ebridge_definition.json', 'ebridge_halfspaces.csv', 'eta_normalization.json'):
        shutil.copy2(T0/name, HERE/name)

    protocol = '''# OrthoFlow3 true-t0 robust eta dataset and point learning v1

The source-isolated TRAIN/VAL/TEST candidate orders are frozen before any new eta outcome. Each state is a genuine episode-start state with its exact h0/current-Flow identity. One global deterministic E_bridge Sobol sequence is shared by all states. The per-state search is capped at 32 candidates plus zero/common anchors, then one optional 32-candidate extension; only 8/8 candidates (or at most two 7/8 candidates after full-search failure) are promoted to Q64. Exact B63 means >=63/64. Target selection uses only local robustness and geometric interiority, never J_def or a learned score. Eta is selected once at t0 and remains fixed while OrthoFlow3 bases update every physical step.
'''
    (HERE/'protocol.md').write_text(protocol)
    dump(HERE/'conditioning_semantics.json', {
        'h0': 'exact h(z0,xi0), current Flow fixed by flow_seed/rng_namespace/step0',
        'future_flow': f'state-token-folded root {FUTURE_ROOT} and future_index',
        'eta': 'selected once at true t0 and persistent', 'basis': 'OrthoFlow3 recomputed each physical step',
        'first_and_second_safety_projection': True,
    })
    dump(HERE/'cost_preflight.json', {
        'first_wave_states': 32, 'worst_case_continuations_per_state': 34*8 + 4*56 + 32*8 + 4*56 + 2*56 + 4*6*8,
        'expected_typical_continuations_per_state': 34*8 + 4*56 + 4*6*8,
        'execution': 'six one-shard state workers when server remains idle',
        'adaptive_extension_only_after_no_B63': True,
    })
    print(json.dumps({'eligible_new_states': len(states), 'order_sizes': {k: len(v) for k,v in orders.items()},
                      'features_shape': list(np.asarray(features).shape), 'sobol_sha256': sha(HERE/'global_eta_sobol64.csv')}, indent=2))


if __name__ == '__main__':
    main()
