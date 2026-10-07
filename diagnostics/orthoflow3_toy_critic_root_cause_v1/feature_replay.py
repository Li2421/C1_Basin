#!/usr/bin/env python3
"""No-rollout replay of frozen true-t0 critic features."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
jax.config.update('jax_enable_x64', True)

ROOT=Path('/home/zhihan/research/Basin_C1')
SYSROOT=Path('/home/zhihan/research/02_C1_Toy_GiveWay')
sys.path[:0]=[str(SYSROOT),str(ROOT)]
D=ROOT/'diagnostics'
OUT=Path(__file__).resolve().parent

def main():
    from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
    from diagnostics.gphi_training_dataset_v2.finalize_dataset import restore_full
    from diagnostics.single_segment_recovery_training_v1.state_machine import FiniteHistoryView
    from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
    from single_integrator.cbf import CBFConfig,barrier_constraints
    from single_integrator.environment import Config,GiveWayEnv,bounded_nominal
    from single_integrator.evaluate import load_policy
    cfg=json.loads((D/'gphi_fixed_d_eta_predictor_v1/integrity_audit.json').read_text())
    envcfg=Config(**cfg['environment']);cbf=CBFConfig(**cfg['cbf'])
    policy,_=load_policy(SYSROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')
    sample=jax.jit(jax.vmap(lambda obs,key:policy.sample_actions(obs[None],seed=key)[0]))
    states=json.loads((D/'orthoflow3_structured_continuous_q_data_v1/toy_state_split.json').read_text())['states']
    original=np.load(D/'orthoflow3_shared_eta_codebook_v1/state_features.npz')['features']
    builder=StartupAwareFeatureBuilder()
    failures=[];deltas=[]
    for s in states:
        env=restore_full(Path(s['state_file']),envcfg)
        key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(s['flow_seed']),s['rng_namespace']),0)
        obs=jnp.asarray(env.observation(),dtype=jnp.float32)[None]
        flow=bounded_nominal(np.asarray(sample(obs,jnp.asarray(np.stack([np.asarray(key)]))))[0],envcfg.max_speed)
        A,lower,_=barrier_constraints(env.snapshot(),cbf)
        safe,_,_,_=project_velocity_with_retry(flow,A,lower,envcfg.max_speed,cbf)
        x,_=builder.build(FiniteHistoryView(env),{'u_flow':flow,'u_safe':safe},envcfg,cbf)
        delta=float(np.max(abs(x-original[s['dataset_index']])))
        deltas.append(delta)
        if delta>1e-5:failures.append({'state_id':s['state_id'],'max_abs':delta})
    result={'states_replayed':len(states),'max_abs_feature_difference':max(deltas),
            'median_abs_max_feature_difference':float(np.median(deltas)),'failures_gt1e-5':failures,
            'feature_builder_sha256_verified_by_wrapper':True,'new_rollout':0}
    (OUT/'feature_replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
