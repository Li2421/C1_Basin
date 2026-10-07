#!/usr/bin/env python3
"""Matched Safety and frozen one-shot OrthoFlow3 Direct-eta WIDE rollout."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,platform,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
import jax,jax.numpy as jnp
import flax.linen as nn
from flax import serialization

ROOT=Path('/home/zhihan/research/Basin_C1'); SYSROOT=Path('/home/zhihan/research/02_C1_Toy_GiveWay'); HERE=ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1'
for p in (SYSROOT,ROOT): sys.path.insert(0,str(p))
LOW=np.array([0.,-.5,0.],np.float32);HIGH=np.array([1.25,.5,.75],np.float32)
class G(nn.Module):
 @nn.compact
 def __call__(self,x):
  x=nn.silu(nn.Dense(128)(x));x=nn.silu(nn.Dense(128)(x));return jnp.asarray(LOW)+nn.sigmoid(nn.Dense(3)(x))*jnp.asarray(HIGH-LOW)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def outcome(env,error):
 if error:return 'other',error['type']
 s=env.summary()
 if s['wall_collision']:return 'collision','wall_collision'
 if s['agent_collision']:return 'collision','agent_collision'
 if s['success']:return 'success','success'
 if s['deadlock']:return 'deadlock','strict_deadlock'
 return 'timeout','timeout'

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,required=True);a=ap.parse_args()
 jax.config.update('jax_enable_x64',True);jax.config.update('jax_platform_name','gpu')
 from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
 from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
 from shared_control.basis_families import get_basis_family
 from single_integrator.cbf import CBFConfig,barrier_constraints
 from single_integrator.environment import Config,GiveWayEnv,bounded_nominal
 from single_integrator.evaluate import load_policy
 manifest=json.loads((HERE/'fresh_wide_manifest.json').read_text());sel=json.loads((HERE/'selected_checkpoint.json').read_text());norm=json.loads((HERE/'normalization.json').read_text())
 if sha(sel['checkpoint'])!=sel['checkpoint_sha256'] or manifest['checkpoint_sha256']!=sel['checkpoint_sha256']:raise RuntimeError('checkpoint mismatch')
 model=G();template=model.init(jax.random.PRNGKey(0),jnp.zeros((1,214)));params=serialization.from_bytes(template,Path(sel['checkpoint']).read_bytes());hm=np.asarray(norm['h_mean']);hs=np.asarray(norm['h_std'])
 config=Config(**manifest['environment']);cbf=CBFConfig(**manifest['cbf']);policy,prov=load_policy(SYSROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')
 if prov['evaluation_environment']!=manifest['environment']:raise RuntimeError('Flow environment mismatch')
 sample=jax.jit(lambda obs,key:policy.sample_actions(obs[None],seed=key)[0]);basis=get_basis_family('orthoflow3');root=int(manifest['flow_randomness']['root_seed'])
 episodes=[e for i,e in enumerate(manifest['episodes']) if i%a.shards==a.shard];outdir=HERE/'raw/fresh_wide';outdir.mkdir(parents=True,exist_ok=True);path=outdir/f'shard{a.shard}.jsonl';rows=[] if not path.exists() else [json.loads(x) for x in path.read_text().splitlines() if x.strip()];seen={(r['episode_index'],r['controller']) for r in rows};steps=0;started=time.monotonic()
 for ep in episodes:
  episode_key=jax.random.fold_in(jax.random.PRNGKey(np.uint32(root)),int(ep['rollout_id']))
  for controller in ('safety','direct_eta'):
   if (ep['episode_index'],controller) in seen:continue
   env=GiveWayEnv(config);env.reset(np.asarray(ep['initial_positions'],float));eta=None;feature=None;jdef=0.;error=None;retries1=retries2=invalid=0;minres=np.inf;maxspeed=-np.inf
   for step in range(config.max_steps):
    try:
     obs=np.asarray(env.observation(),np.float32);key=jax.random.fold_in(episode_key,step);flow=bounded_nominal(np.asarray(sample(jnp.asarray(obs),key),float),config.max_speed);A,lower,_=barrier_constraints(env.snapshot(),cbf);safe,_,r1,_=project_velocity_with_retry(flow,A,lower,config.max_speed,cbf);retries1+=int(r1)
     if step==0:
      feature,_=StartupAwareFeatureBuilder().build(env,{'u_flow':flow,'u_safe':safe},config,cbf)
      eta=np.zeros(3) if controller=='safety' else np.asarray(model.apply(params,jnp.asarray(((feature-hm)/hs).astype(np.float32))[None])[0],float)
     correction=np.zeros_like(safe) if controller=='safety' else basis.compute(env.positions,env.goals,safe,config.max_speed).correction(eta)
     executed,_,r2,_=project_velocity_with_retry(safe+correction,A,lower,config.max_speed,cbf);retries2+=int(r2);res=float(np.min(A@executed.reshape(4)-lower));spd=float(np.max(np.linalg.norm(executed,axis=-1)-config.max_speed));minres=min(minres,res);maxspeed=max(maxspeed,spd)
     if not np.isfinite(executed).all() or res < -cbf.feasibility_tol or spd > cbf.speed_tol:invalid+=1;raise RuntimeError(('invalid projected action',res,spd))
     jdef+=config.dt*float(np.sum((executed-safe)**2));env.step(executed);steps+=1
     if env.done:break
    except Exception as exc:error={'type':type(exc).__name__,'message':str(exc),'step':step};break
   label,failure=outcome(env,error);summ=env.summary();row={'episode_index':ep['episode_index'],'rollout_id':ep['rollout_id'],'source_id':ep['source_id'],'controller':controller,'initial_positions':ep['initial_positions'],'outcome':label,'failure_type':failure,'success':label=='success','deadlock':label=='deadlock','timeout':label=='timeout','collision':label=='collision','wall_collision':bool(summ['wall_collision']),'agent_collision':bool(summ['agent_collision']),'other_failure':label=='other','episode_steps':int(env.step_count),'J_def':jdef,'eta':eta.tolist(),'eta_norm':float(np.linalg.norm(eta)),'eta_clipped':False,'eta_query_count':0 if controller=='safety' else 1,'feature_sha256':hashlib.sha256(np.asarray(feature,dtype=np.float64).tobytes()).hexdigest(),'second_projection':True,'first_projection_retries':retries1,'second_projection_retries':retries2,'minimum_linear_residual':None if not np.isfinite(minres) else minres,'maximum_speed_excess':None if not np.isfinite(maxspeed) else maxspeed,'invalid_actions':invalid,'execution_error':error,'checkpoint_sha256':None if controller=='safety' else sel['checkpoint_sha256'],'manifest_sha256':sha(HERE/'fresh_wide_manifest.json')};rows.append(row);path.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows));print(json.dumps({'episode':ep['episode_index'],'controller':controller,'outcome':label,'steps':env.step_count}),flush=True)
 runtime={'shard':a.shard,'shards':a.shards,'controller_rollouts':len(rows),'physical_steps':steps,'wall_seconds':time.monotonic()-started,'outcomes':dict(Counter(r['outcome'] for r in rows)),'devices':[str(x) for x in jax.devices()],'host':platform.node(),'cpu_request':os.environ.get('SLURM_CPUS_PER_TASK'),'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES')};(outdir/f'shard{a.shard}_runtime.json').write_text(json.dumps(runtime,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
