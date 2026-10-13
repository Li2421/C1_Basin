"""Observer-only replay of exact cached LOSO continuations for complete videos.

Historical Q16 labels are never regenerated or overwritten. Each selected seed is
replayed only because the old cache did not retain per-step positions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2')

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
TOY_ROOT = Path('/home/zhihan/research/02_C1_Toy_GiveWay')
sys.path.insert(0, str(TOY_ROOT))
sys.path.insert(0, str(ROOT))

from new_benchmark_common.batch_videos import load_trace, sha256
from shared_rollout_db.src.rollout_db import canonical


def capture_step(env, items):
    original = env.step
    items['positions'] = [env.positions.copy()]
    items['goals'] = env.goals.copy()
    items['walls'] = np.asarray(getattr(env, 'walls', np.empty((0, 2, 2))), float).reshape(-1, 2, 2)
    if hasattr(env, 'reset'):
        original_reset = env.reset
        def observed_reset(*args, **kwargs):
            result = original_reset(*args, **kwargs)
            items['positions'] = [env.positions.copy()]
            items['goals'] = env.goals.copy()
            items['walls'] = np.asarray(getattr(env, 'walls', np.empty((0, 2, 2))), float).reshape(-1, 2, 2)
            items['clearance'].clear()
            return result
        env.reset = observed_reset

    def observed(action):
        result = original(action)
        info = result[3]
        items['positions'].append(env.positions.copy())
        if 'swept_obstacle_distance' in info:
            vals = [np.min(info['swept_obstacle_distance']), np.min(info['swept_outer_distance']),
                    info['min_swept_agent_distance']]
        else:
            vals = [info['min_swept_wall_distance'], info['min_swept_agent_distance']]
        items['clearance'].append(float(np.min(vals)))
        return result

    env.step = observed
    return env


def replay_new(case, runtimes):
    scene = {'four': 'four_way_intersection', 'ring': 'ring_exchange'}[case['scene']]
    rt = runtimes[scene]
    state = rt.states[case['state_index']]
    assert state['uid'] == case['state_uid']
    chain = 'hard_safety_q16' if case['panel'] == 'safety_baseline' else 'orthoflow3'
    assert rt.controllers[chain]['uid'] == case['controller_uid']
    items = {'clearance': []}
    factory = rt.make_env
    rt.make_env = lambda: capture_step(factory(), items)
    try:
        outcome = rt.rollout(state, np.asarray(case['eta'], float), case['future_index'], chain, save_trace=True)
    finally:
        rt.make_env = factory
    assert len(items['positions']) == outcome['episode_length'] + 1
    if 'trace' in outcome:
        np.testing.assert_allclose(np.asarray(items['positions']), np.asarray(outcome['trace']['positions']), atol=0, rtol=0)
    return outcome, items, rt.config.to_dict(), str(rt.spec['checkpoint']), rt.checkpoint_sha


def replay_double(case, runtimes):
    from diagnostics.orthoflow3_generator_critic_frozen_test_v1 import run_frozen_test as frozen
    rt = runtimes['double_bottleneck']
    state = rt.states[case['state_index']]
    assert state['uid'] == case['state_uid']
    chain = 'hard_safety_q16' if case['panel'] == 'safety_baseline' else 'orthoflow3'
    assert rt.controllers[chain]['uid'] == case['controller_uid']
    items = {'clearance': []}
    original = frozen.db_old._initialize_env

    def observed(*args, **kwargs):
        return capture_step(original(*args, **kwargs), items)

    frozen.db_old._initialize_env = observed
    try:
        outcome = rt.rollout(state, np.asarray(case['eta'], float), case['future_index'], chain)
    finally:
        frozen.db_old._initialize_env = original
    config = rt.datasets[state['metadata']['set']].config
    assert len(items['positions']) == outcome['episode_length'] + 1
    return outcome, items, config, str(frozen.db_old.CHECKPOINT), rt.checkpoint_sha


def replay_toy(case, policy_cache):
    from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
    from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
    from shared_control.basis_families import get_basis_family
    from single_integrator.cbf import CBFConfig, barrier_constraints
    from single_integrator.environment import Config, GiveWayEnv, bounded_nominal
    from single_integrator.evaluate import load_policy
    from diagnostics.orthoflow3_mode_free_generator_critic_hard_cohort_v1.run_rollouts import label

    jax.config.update('jax_enable_x64', True)
    if not policy_cache:
        wide = json.loads((ROOT/'diagnostics/gphi_wide_ic_cadence_v1/frozen_benchmark_manifest.json').read_text())
        ckpt = TOY_ROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl'
        policy, provenance = load_policy(ckpt)
        assert provenance['evaluation_environment'] == wide['environment']
        policy_cache.update(policy=policy, config=Config(**wide['environment']), cbf=CBFConfig(**wide['cbf']),
                            checkpoint=str(ckpt), checkpoint_sha=sha256(ckpt),
                            sample=jax.jit(lambda obs, key: policy.sample_actions(obs[None], seed=key)[0]),
                            states={s['state_uid']: s for s in json.loads((ROOT/'diagnostics/orthoflow3_c1_generalization_v1/toy_replication/frozen_proposals.json').read_text())['states']},
                            features=np.load(ROOT/'diagnostics/orthoflow3_c1_generalization_v1/toy_replication/cohort_features.npz')['h_raw'])
    state = policy_cache['states'][case['state_uid']]
    assert int(state['episode_index']) == case['state_index']
    config, cbf, sample = policy_cache['config'], policy_cache['cbf'], policy_cache['sample']
    env = GiveWayEnv(config)
    env.reset(np.asarray(state['initial_positions'], float))
    items = {'clearance': []}
    capture_step(env, items)
    initial_key = jax.random.fold_in(jax.random.PRNGKey(42), int(state['rollout_id']))
    future_key = jax.random.fold_in(jax.random.PRNGKey(42+case['future_index']), int(state['rollout_id']))
    basis = get_basis_family('orthoflow3')
    eta = np.asarray(case['eta'], float)
    error = None
    for step in range(config.max_steps):
        try:
            obs = np.asarray(env.observation(), np.float32)
            key = jax.random.fold_in(initial_key if step == 0 else future_key, step)
            flow = bounded_nominal(np.asarray(sample(jnp.asarray(obs), key), np.float64), config.max_speed)
            A, lower, _ = barrier_constraints(env.snapshot(), cbf)
            safe, _, _, _ = project_velocity_with_retry(flow, A, lower, config.max_speed, cbf)
            if step == 0:
                feature, _ = StartupAwareFeatureBuilder().build(env, {'u_flow': flow, 'u_safe': safe}, config, cbf)
                np.testing.assert_allclose(np.asarray(feature, np.float32), policy_cache['features'][case['state_index']], rtol=1e-6, atol=1e-7)
            correction = basis.compute(env.positions, env.goals, safe, config.max_speed).correction(eta)
            executed, _, _, _ = project_velocity_with_retry(safe+correction, A, lower, config.max_speed, cbf)
            if np.min(A @ executed.reshape(4)-lower) < -cbf.feasibility_tol:
                raise RuntimeError('Hard-safety projection violated')
            env.step(executed)
            if env.done:
                break
        except Exception as exc:
            error = {'type': type(exc).__name__, 'message': str(exc), 'step': step}
            break
    outcome, _ = label(env, error)
    return ({'outcome': outcome, 'episode_length': env.step_count,
             'collision': bool(env.summary()['wall_collision'] or env.summary()['agent_collision']),
             'numerical_failure': error is not None, 'termination': outcome,
             'execution_error': error}, items, config.to_dict(),
            policy_cache['checkpoint'], policy_cache['checkpoint_sha'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', choices=['toy','db','four','ring'])
    args = ap.parse_args()
    plan = json.loads((OUT/'selection.json').read_text())['cases']
    preflight = json.loads((OUT/'cache_preflight.json').read_text())
    assert preflight['summary']['exact_reusable'] == len(plan) and preflight['summary']['genuinely_missing'] == 0
    runtimes = None
    if args.scene != 'toy':
        # The current MACFlow source acquired optional Gap1 model branches after
        # this LOSO experiment. Load its exact historical source bytes so both
        # execution and controller fingerprint match the archived evidence.
        import types
        import new_benchmark_common.basin_dataset_v1 as basin_dataset
        import new_benchmark_common.safety_eta3 as safety_eta3
        historic_path = OUT/'macflow_historical.py'
        assert sha256(historic_path) == 'd02e61d0b5447f0da206b839ab292d5f7655ec44e6be866b15a5227d7cb06fac'
        historical_macflow = types.ModuleType('loso_macflow_historical')
        historical_macflow.__file__ = str(ROOT/'new_benchmark_common/macflow.py')
        sys.modules[historical_macflow.__name__] = historical_macflow
        exec(compile(historic_path.read_bytes(), historical_macflow.__file__, 'exec'), historical_macflow.__dict__)
        basin_dataset.load_checkpoint = historical_macflow.load_checkpoint
        safety_eta3.load_checkpoint = historical_macflow.load_checkpoint
        safety_eta3.sample_bounded_actions = historical_macflow.sample_bounded_actions
        from diagnostics.orthoflow3_k16_critic_canonicalization_v1.confirmation_a import runtimes as get_runtimes
        runtimes = dict(get_runtimes())
        for rt in runtimes.values():
            if rt.name in ('four_way_intersection', 'ring_exchange'):
                rt.macflow_hash = sha256(historic_path)
                rt.controllers['orthoflow3'] = safety_eta3.ScenarioRuntime._controller(rt, 'orthoflow3')
                rt.controllers['hard_safety_q16'] = safety_eta3.ScenarioRuntime._controller(rt, 'hard_safety_q16')
    policy_cache = {}
    trace_dir = OUT/'traces';trace_dir.mkdir(exist_ok=True)
    for case in plan:
        if args.scene and case['scene'] != args.scene:
            continue
        target = trace_dir/(case['name']+'.npz')
        if target.exists():
            meta, _ = load_trace(target)
            assert meta['rollout_id'] == case['expected']['rollout_uid']
            print(json.dumps({'name':case['name'],'status':'trace_reused'}),flush=True)
            continue
        if case['scene'] == 'toy':
            result, items, config, checkpoint, checkpoint_sha = replay_toy(case, policy_cache)
        elif case['scene'] == 'db':
            result, items, config, checkpoint, checkpoint_sha = replay_double(case, runtimes)
        else:
            result, items, config, checkpoint, checkpoint_sha = replay_new(case, runtimes)
        expected = case['expected']
        if result['outcome'] != expected['outcome'] or result['episode_length'] != expected['episode_length']:
            raise RuntimeError(f"Replay mismatch {case['name']}: {result['outcome']}/{result['episode_length']} vs {expected['outcome']}/{expected['episode_length']}")
        pos = np.asarray(items['positions'], np.float64)
        clearance = np.asarray(items['clearance'], np.float64)
        assert len(pos) == len(clearance)+1 == result['episode_length']+1
        scene = {'toy':'toy_giveway','db':'double_bottleneck','four':'four_way_intersection','ring':'ring_exchange'}[case['scene']]
        initial_hash = hashlib.sha256(canonical({'positions':pos[0].tolist(),
            'state_uid':case['state_uid']}).encode()).hexdigest()
        meta = {'batch_id':'strict_loso_video_20261011','scenario':scene,
                'display_label':f"{scene} {case['panel']} eta={np.round(case['eta'],3).tolist()}",
                'rollout_id':expected['rollout_uid'], 'controller':case['controller_uid'],
                'checkpoint':checkpoint, 'checkpoint_sha256':checkpoint_sha,
                'seed':case['future_index'], 'source_record':expected['original_source_file'],
                'state_uid':case['state_uid'], 'initial_state_sha256':initial_hash,
                'eta':case['eta'], 'selected_index':case['selected_index'],
                'config':config, 'start_step':0,'episode_steps':result['episode_length'],
                'complete':True,'termination':result['outcome'],
                'collision':bool(result['collision']), 'safety_enabled':True,
                'replay_matches_cached_outcome':True,
                'replay_reason':'Historical cache lacked complete per-step trace; no Q estimate recomputed.'}
        np.savez_compressed(target,positions=pos,steps=np.arange(len(pos)),
                            goals=np.asarray(items['goals'],float), walls=np.asarray(items['walls'],float),
                            swept_clearance=clearance, metadata_json=np.array(json.dumps(meta)))
        load_trace(target)
        print(json.dumps({'name':case['name'],'outcome':result['outcome'],
                          'steps':result['episode_length'],'min_swept_clearance':float(clearance.min()),
                          'trace':str(target)}),flush=True)


if __name__ == '__main__':
    main()
