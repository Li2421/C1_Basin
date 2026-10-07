#!/usr/bin/env python3
import json,sys
from pathlib import Path
import jax,jax.numpy as jnp,numpy as np
jax.config.update('jax_enable_x64',True);jax.config.update('jax_platform_name','gpu')
ROOT=Path('/home/zhihan/research/Basin_C1'); SYSROOT=Path('/home/zhihan/research/02_C1_Toy_GiveWay'); HERE=ROOT/'diagnostics/orthoflow3_true_t0_point_learning_v1'
for p in (SYSROOT,ROOT):sys.path.insert(0,str(p))
from diagnostics.gphi_training_dataset_v2.finalize_dataset import restore_full
from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
from diagnostics.single_segment_recovery_training_v1.state_machine import FiniteHistoryView
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
from single_integrator.cbf import CBFConfig,barrier_constraints
from single_integrator.environment import Config,bounded_nominal
from single_integrator.evaluate import load_policy
st=json.load(open(HERE/'eligible_state_manifest.json'))['selected_states'][0]; feat=np.load(HERE/'conditioning_features.npz')['features'][0]
integ=json.load(open(ROOT/'diagnostics/gphi_fixed_d_eta_predictor_v1/integrity_audit.json')); cfg=Config(**integ['environment']);cbf=CBFConfig(**integ['cbf']);policy,_=load_policy(SYSROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')
sample=jax.jit(jax.vmap(lambda obs,key:policy.sample_actions(obs[None],seed=key)[0])); env=restore_full(Path(st['state_file']),cfg); base=jax.random.fold_in(jax.random.PRNGKey(int(st['flow_seed'])),int(st['rng_namespace'])); key=np.asarray(jax.random.fold_in(base,0)); obs=np.asarray(env.observation(),np.float32)
for n in (1,16,32,64):
 acts=np.asarray(sample(jnp.asarray(np.repeat(obs[None],n,axis=0)),jnp.asarray(np.repeat(key[None],n,axis=0)))); act=acts[0]; flow=bounded_nominal(act,cfg.max_speed);A,lower,_=barrier_constraints(env.snapshot(),cbf);safe,_,_,_=project_velocity_with_retry(flow,A,lower,cfg.max_speed,cbf);h,_=StartupAwareFeatureBuilder().build(FiniteHistoryView(env),{'u_flow':flow,'u_safe':safe},cfg,cbf);h=np.asarray(h,np.float64);print(n,'act',act.tolist(),'maxerr_saved',np.max(np.abs(h-feat)),'hfirst',h[:8].tolist())
# Mirror the patched rollout exactly: restore 64 envs first, then call batch-one 64 times.
envs=[restore_full(Path(st['state_file']),cfg) for _ in range(64)]; obs64=np.stack([np.asarray(e.observation(),np.float32) for e in envs]); keys=[key for _ in envs]
acts=np.concatenate([np.asarray(sample(jnp.asarray(obs64[j:j+1]),jnp.asarray(np.stack(keys[j:j+1])))) for j in range(64)],axis=0)
for idx in (0,63):
 e=envs[idx]; flow=bounded_nominal(acts[idx],cfg.max_speed);A,lower,_=barrier_constraints(e.snapshot(),cbf);safe,_,_,_=project_velocity_with_retry(flow,A,lower,cfg.max_speed,cbf);h,_=StartupAwareFeatureBuilder().build(FiniteHistoryView(e),{'u_flow':flow,'u_safe':safe},cfg,cbf);h=np.asarray(h,np.float64);print('mirror',idx,'act',acts[idx].tolist(),'maxerr_saved',np.max(np.abs(h-feat)))
