#!/usr/bin/env python3
"""Recompute frozen h0 features on the rollout backend for exact replay parity."""
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
import jax,jax.numpy as jnp
import numpy as np
jax.config.update('jax_enable_x64', True)
jax.config.update('jax_platform_name', 'gpu')

ROOT=Path('/home/zhihan/research/Basin_C1'); SYSROOT=Path('/home/zhihan/research/02_C1_Toy_GiveWay')
HERE=ROOT/'diagnostics/orthoflow3_true_t0_point_learning_v1'
for p in (SYSROOT,ROOT): sys.path.insert(0,str(p))

from diagnostics.gphi_training_dataset_v2.finalize_dataset import restore_full
from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
from diagnostics.single_segment_recovery_training_v1.state_machine import FiniteHistoryView
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
from single_integrator.cbf import CBFConfig,barrier_constraints
from single_integrator.environment import Config,bounded_nominal
from single_integrator.evaluate import load_policy

def main():
 mpath=HERE/'eligible_state_manifest.json'; manifest=json.load(open(mpath)); states=manifest['selected_states']
 integ=json.load(open(ROOT/'diagnostics/gphi_fixed_d_eta_predictor_v1/integrity_audit.json')); config=Config(**integ['environment']); cbf=CBFConfig(**integ['cbf'])
 policy,_=load_policy(SYSROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')
 # Match the rollout Oracle exactly: a jitted vmap, even for a one-state batch.
 sample=jax.jit(jax.vmap(lambda obs,key:policy.sample_actions(obs[None],seed=key)[0]))
 features=[]
 for i,st in enumerate(states):
  env=restore_full(Path(st['state_file']),config); base=jax.random.fold_in(jax.random.PRNGKey(int(st['flow_seed'])),int(st['rng_namespace'])); key=jax.random.fold_in(base,0)
  obs=jnp.asarray(env.observation(),dtype=jnp.float32)[None]
  act=np.asarray(sample(obs,jnp.asarray(np.stack([np.asarray(key)]))))[0]
  flow=bounded_nominal(act,config.max_speed); A,lower,_=barrier_constraints(env.snapshot(),cbf)
  safe,_,_,_=project_velocity_with_retry(flow,A,lower,config.max_speed,cbf); h,_=StartupAwareFeatureBuilder().build(FiniteHistoryView(env),{'u_flow':flow,'u_safe':safe},config,cbf); h=np.asarray(h,dtype=np.float64)
  if h.shape!=(214,): raise RuntimeError((st['state_id'],h.shape))
  features.append(h); st['feature_index']=i; st['feature_sha256']=hashlib.sha256(h.tobytes()).hexdigest()
 np.savez_compressed(HERE/'conditioning_features.npz',features=np.asarray(features)); mpath.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'device':str(jax.devices()[0]),'states':len(states),'shape':list(np.asarray(features).shape)},indent=2))
if __name__=='__main__':main()
