#!/usr/bin/env python3
"""Synchronized, batched online proposal-selection latency benchmark."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization

ROOT = Path('/home/zhihan/research/Basin_C1')
OLD = ROOT / 'diagnostics/orthoflow3_mode_free_generator_critic_hard_cohort_v1'
OUT = Path(__file__).resolve().parent
SYSROOT = Path('/home/zhihan/research/02_C1_Toy_GiveWay')
sys.path[:0] = [str(SYSROOT), str(ROOT)]

spec = importlib.util.spec_from_file_location('frozen_models_for_latency', OLD / 'freeze_proposals.py')
prior = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = prior
spec.loader.exec_module(prior)


def percentile_summary(samples_ms):
    x = np.asarray(samples_ms, np.float64)
    return {'mean_ms': float(x.mean()), 'p50_ms': float(np.median(x)),
            'p95_ms': float(np.percentile(x, 95)), 'p99_ms': float(np.percentile(x, 99)),
            'n': int(len(x))}


def timed(fn, args_for_i, repeats, warmup=40):
    for i in range(warmup):
        jax.block_until_ready(fn(*args_for_i(i)))
    obs = []
    for i in range(repeats):
        args = args_for_i(i + warmup)
        start = time.perf_counter_ns()
        jax.block_until_ready(fn(*args))
        obs.append((time.perf_counter_ns() - start) / 1e6)
    return percentile_summary(obs)


def model_functions(K):
    training = json.loads((OLD / 'generator_training.json').read_text())
    model = prior.genlib.Generator()
    template = model.init(jax.random.PRNGKey(0), jnp.zeros((1,214), jnp.float32), 'Toy')
    gp = serialization.from_bytes(template, Path(training['selected']['checkpoint']).read_bytes())
    manifest = json.loads((prior.OLD / 'dataset_manifest.json').read_text())
    sn = manifest['state_normalization']['Toy']
    hm, hs = jnp.asarray(sn['mean'], jnp.float32), jnp.asarray(sn['std'], jnp.float32)
    ec = jnp.asarray(manifest['eta_normalization']['center'], jnp.float32)
    es = jnp.asarray(manifest['eta_normalization']['scale'], jnp.float32)
    critic = prior.ranklib.SingleCritic()
    cp = []
    for seed in (17,23,41):
        template = critic.init(jax.random.PRNGKey(seed), jnp.zeros((1,214)), jnp.zeros((1,3)))
        path = prior.CRITIC / 'models/secondary_combined/W1' / f'seed{seed}/checkpoint.msgpack'
        cp.append(serialization.from_bytes(template, path.read_bytes()))
    center = jnp.asarray(prior.genlib.CENTER)
    radius = jnp.asarray(prior.genlib.RADIUS)

    @jax.jit
    def gen(h_raw, key):
        h = ((h_raw-hm)/hs)[None]
        raw = model.apply(gp, h, 'Toy')
        mu, sigma = prior.genlib.dist_params(raw)
        noise = jax.random.normal(key, (K,3), dtype=jnp.float32)
        return center + radius*jnp.tanh(mu + sigma*noise)

    @jax.jit
    def score(h_raw, eta):
        h = jnp.broadcast_to((h_raw-hm)/hs, (K,214))
        z = (eta-ec)/es
        scores = [critic.apply(p,h,z) for p in cp]
        return jnp.mean(jnp.stack(scores), axis=0)

    @jax.jit
    def select(scores, eta):
        idx = jnp.argmax(scores)
        return idx, eta[idx]

    @jax.jit
    def full(h_raw, key):
        eta = gen(h_raw, key)
        scores = score(h_raw, eta)
        return select(scores, eta)

    return gen, score, select, full


def state_and_base_timing(hraw, repeats):
    from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
    from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
    from single_integrator.cbf import CBFConfig, barrier_constraints
    from single_integrator.environment import Config, GiveWayEnv, bounded_nominal
    from single_integrator.evaluate import load_policy

    wide = json.loads((ROOT / 'diagnostics/gphi_wide_ic_cadence_v1/frozen_benchmark_manifest.json').read_text())
    config, cbf = Config(**wide['environment']), CBFConfig(**wide['cbf'])
    policy, _ = load_policy(SYSROOT / 'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')
    sample = jax.jit(lambda obs, key: policy.sample_actions(obs[None], seed=key)[0])
    envs, flows, safes, obs, keys = [], [], [], [], []
    for state in wide['episodes'][:16]:
        env = GiveWayEnv(config)
        env.reset(np.asarray(state['initial_positions'], np.float64))
        key = jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(42), int(state['rollout_id'])),0)
        o = np.asarray(env.observation(), np.float32)
        f = bounded_nominal(np.asarray(sample(jnp.asarray(o),key),np.float64), config.max_speed)
        A, lower, _ = barrier_constraints(env.snapshot(),cbf)
        s, _, _, _ = project_velocity_with_retry(f,A,lower,config.max_speed,cbf)
        envs.append(env); flows.append(f); safes.append(s); obs.append(o); keys.append(key)
    builder = StartupAwareFeatureBuilder()
    def feature(i):
        return builder.build(envs[i%len(envs)], {'u_flow':flows[i%len(envs)],'u_safe':safes[i%len(envs)]},config,cbf)[0]
    def base(i):
        j = i%len(envs)
        f = bounded_nominal(np.asarray(sample(jnp.asarray(obs[j]), keys[j]),np.float64), config.max_speed)
        A, lower, _ = barrier_constraints(envs[j].snapshot(),cbf)
        return project_velocity_with_retry(f,A,lower,config.max_speed,cbf)[0]
    for i in range(20): feature(i); base(i)
    fs, bs = [], []
    for i in range(repeats):
        start = time.perf_counter_ns(); feature(i); fs.append((time.perf_counter_ns()-start)/1e6)
        start = time.perf_counter_ns(); base(i); bs.append((time.perf_counter_ns()-start)/1e6)
    return percentile_summary(fs), percentile_summary(bs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=1000)
    parser.add_argument('--sequential-repeats', type=int, default=100)
    parser.add_argument('--output', type=Path, default=OUT / 'latency_benchmark.json')
    args = parser.parse_args()
    if args.repeats < 1000: raise RuntimeError('At least 1000 batched inference repeats required')
    h = np.load(OLD / 'cohort_features.npz')['h_raw']
    h_input = [h[i%len(h)] for i in range(args.repeats+100)]
    keys = np.asarray(jax.random.split(jax.random.PRNGKey(20261001), args.repeats+100))
    result = {'device_platform': jax.default_backend(), 'devices': [str(d) for d in jax.devices()],
              'machine': platform.machine(), 'processor': platform.processor(),
              'precision': 'float32 neural inference, JAX; float64 physical safety projection',
              'batching': 'single JIT/vectorized K proposals and three-critic ensemble, synchronized after each call',
              'repeats': args.repeats, 'sequential_diagnostic_repeats': args.sequential_repeats,
              'K': {}}
    for K in (1,2,4,8,16):
        gen, score, select, full = model_functions(K)
        eta = np.asarray(gen(h_input[0],keys[0]))
        scores = np.asarray(score(h_input[0],eta))
        row = {'generator': timed(gen, lambda i: (h_input[i],keys[i]),args.repeats),
               'critic': timed(score,lambda i:(h_input[i],eta),args.repeats),
               'selection': timed(select,lambda i:(scores,eta),args.repeats),
               'end_to_end': timed(full,lambda i:(h_input[i],keys[i]),args.repeats)}
        # Diagnostic only: force K separate generator + critic calls and select in Python.
        gen1, score1, _, _ = model_functions(1)
        for i in range(5):
            for j in range(K):
                ep = np.asarray(gen1(h_input[i],keys[(i*K+j)%len(keys)]))
                jax.block_until_ready(score1(h_input[i],ep))
        seq = []
        for i in range(args.sequential_repeats):
            start=time.perf_counter_ns()
            vals=[]
            for j in range(K):
                ep=gen1(h_input[i],keys[(i*K+j)%len(keys)])
                vals.append(score1(h_input[i],ep))
            jax.block_until_ready(jnp.stack(vals))
            seq.append((time.perf_counter_ns()-start)/1e6)
        row['sequential_K_diagnostic'] = percentile_summary(seq)
        row['theoretical_update_hz_from_p50'] = 1000/row['end_to_end']['p50_ms']
        row['theoretical_update_hz_from_p95'] = 1000/row['end_to_end']['p95_ms']
        result['K'][str(K)] = row
        print(json.dumps({'K': K, 'batched_p50_ms': row['end_to_end']['p50_ms'],
                          'batched_p95_ms': row['end_to_end']['p95_ms']}),flush=True)
    result['state_encoding'], result['base_safety_step'] = state_and_base_timing(h,args.repeats)
    for K,row in result['K'].items():
        row['hypothetical_loop_p50_ms'] = result['base_safety_step']['p50_ms'] + row['end_to_end']['p50_ms']
        row['hypothetical_loop_p95_ms'] = result['base_safety_step']['p95_ms'] + row['end_to_end']['p95_ms']
        row['hypothetical_loop_hz_from_p50'] = 1000/row['hypothetical_loop_p50_ms']
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'written':str(args.output),'backend':result['device_platform']}))


if __name__=='__main__': main()
