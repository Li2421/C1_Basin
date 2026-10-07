#!/usr/bin/env python3
"""Matched fresh WIDE Safety/baseline/guided evaluation; Q is not loaded."""
import argparse,hashlib,json,os,platform,sys,time
from collections import Counter
from pathlib import Path
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');SYSROOT=Path('/home/zhihan/research/02_C1_Toy_GiveWay');HERE=ROOT/'diagnostics/orthoflow3_q_guided_direct_eta_v1';BASE=ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1';sys.path[:0]=[str(ROOT),str(SYSROOT)];LOW=np.array([0.,-.5,0.],np.float32);HIGH=np.array([1.25,.5,.75],np.float32)
class G(nn.Module):
 @nn.compact
 def __call__(self,x):x=nn.silu(nn.Dense(128)(x));x=nn.silu(nn.Dense(128)(x));return jnp.asarray(LOW)+nn.sigmoid(nn.Dense(3)(x))*jnp.asarray(HIGH-LOW)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def label(env,error):
 if error:return 'other',error['type']
 s=env.summary()
 if s['wall_collision']:return 'collision','wall_collision'
 if s['agent_collision']:return 'collision','agent_collision'
 if s['success']:return 'success','success'
 if s['deadlock']:return 'deadlock','strict_deadlock'
 return 'timeout','timeout'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,required=True);a=ap.parse_args();jax.config.update('jax_enable_x64',True);jax.config.update('jax_platform_name','gpu')
 from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
 from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
 from shared_control.basis_families import get_basis_family
 from single_integrator.cbf import CBFConfig,barrier_constraints
 from single_integrator.environment import Config,GiveWayEnv,bounded_nominal
 from single_integrator.evaluate import load_policy
 man=json.loads((HERE/'fresh_wide_manifest.json').read_text());norm=json.loads((BASE/'normalization.json').read_text());hm=np.asarray(norm['h_mean']);hs=np.asarray(norm['h_std']);bs=json.loads((BASE/'selected_checkpoint.json').read_text());gs=json.loads((HERE/'selected_checkpoint.json').read_text());gm=G();tmp=gm.init(jax.random.PRNGKey(0),jnp.zeros((1,214)));models={'baseline':serialization.from_bytes(tmp,Path(bs['checkpoint']).read_bytes()),'guided':serialization.from_bytes(tmp,Path(gs['checkpoint']).read_bytes())}
 if sha(bs['checkpoint'])!=man['baseline_checkpoint_sha256'] or sha(gs['checkpoint'])!=man['guided_checkpoint_sha256']:raise RuntimeError('checkpoint mismatch')
 cfg=Config(**man['environment']);cbf=CBFConfig(**man['cbf']);policy,prov=load_policy(SYSROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl');sample=jax.jit(lambda obs,key:policy.sample_actions(obs[None],seed=key)[0]);basis=get_basis_family('orthoflow3');root=man['flow_randomness']['root_seed'];episodes=[e for i,e in enumerate(man['episodes']) if i%a.shards==a.shard];outdir=HERE/'raw/fresh_wide';outdir.mkdir(parents=True,exist_ok=True);path=outdir/f'shard{a.shard}.jsonl';rows=[] if not path.exists() else [json.loads(x) for x in path.read_text().splitlines() if x.strip()];seen={(r['episode_index'],r['controller']) for r in rows};steps=0;started=time.monotonic()
 for ep in episodes:
  ek=jax.random.fold_in(jax.random.PRNGKey(np.uint32(root)),ep['rollout_id'])
  for controller in ('safety','baseline','guided'):
   if (ep['episode_index'],controller) in seen:continue
   env=GiveWayEnv(cfg);env.reset(np.asarray(ep['initial_positions'],float));eta=None;feat=None;jdef=0.;error=None;invalid=naninf=projfail=r1c=r2c=0;minres=np.inf;maxspd=-np.inf
   for step in range(cfg.max_steps):
    try:
     obs=np.asarray(env.observation(),np.float32);act=np.asarray(sample(jnp.asarray(obs),jax.random.fold_in(ek,step)),float);flow=bounded_nominal(act,cfg.max_speed);A,lower,_=barrier_constraints(env.snapshot(),cbf);safe,_,r1,_=project_velocity_with_retry(flow,A,lower,cfg.max_speed,cbf);r1c+=int(r1)
     if step==0:
      feat,_=StartupAwareFeatureBuilder().build(env,{'u_flow':flow,'u_safe':safe},cfg,cbf);eta=np.zeros(3) if controller=='safety' else np.asarray(gm.apply(models[controller],jnp.asarray(((feat-hm)/hs).astype(np.float32))[None])[0],float)
     corr=np.zeros_like(safe) if controller=='safety' else basis.compute(env.positions,env.goals,safe,cfg.max_speed).correction(eta);executed,_,r2,_=project_velocity_with_retry(safe+corr,A,lower,cfg.max_speed,cbf);r2c+=int(r2);res=float(np.min(A@executed.reshape(4)-lower));spd=float(np.max(np.linalg.norm(executed,axis=-1)-cfg.max_speed));minres=min(minres,res);maxspd=max(maxspd,spd)
     if not np.isfinite(executed).all():naninf+=1;raise RuntimeError('NaN/Inf action')
     if res < -cbf.feasibility_tol or spd > cbf.speed_tol:invalid+=1;raise RuntimeError(('invalid action',res,spd))
     jdef+=cfg.dt*float(np.sum((executed-safe)**2));env.step(executed);steps+=1
     if env.done:break
    except Exception as exc:projfail+=int('project' in str(exc).lower() or 'solver' in type(exc).__name__.lower());error={'type':type(exc).__name__,'message':str(exc),'step':step};break
   out,fail=label(env,error);s=env.summary();row={'episode_index':ep['episode_index'],'source_id':ep['source_id'],'controller':controller,'outcome':out,'failure_type':fail,'success':out=='success','deadlock':out=='deadlock','timeout':out=='timeout','collision':out=='collision','wall_collision':bool(s['wall_collision']),'agent_collision':bool(s['agent_collision']),'other_failure':out=='other','episode_steps':int(env.step_count),'J_def':jdef,'eta':eta.tolist(),'eta_norm':float(np.linalg.norm(eta)),'eta_clipped':False,'feature0':np.asarray(feat,float).tolist(),'invalid_actions':invalid,'nan_inf':naninf,'projection_failures':projfail,'first_projection_retries':r1c,'second_projection_retries':r2c,'minimum_linear_residual':None if not np.isfinite(minres) else minres,'maximum_speed_excess':None if not np.isfinite(maxspd) else maxspd,'execution_error':error,'manifest_sha256':sha(HERE/'fresh_wide_manifest.json')};rows.append(row);path.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows))
  print(json.dumps({'episode':ep['episode_index'],'done':len(rows)}),flush=True)
 rt={'shard':a.shard,'shards':a.shards,'new_controller_rollouts':len(rows),'physical_steps':steps,'wall_seconds':time.monotonic()-started,'outcomes':dict(Counter(r['outcome'] for r in rows)),'devices':[str(x) for x in jax.devices()],'host':platform.node()};(outdir/f'shard{a.shard}_runtime.json').write_text(json.dumps(rt,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
