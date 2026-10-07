#!/usr/bin/env python3
"""Exact Q-conditioned fixed-eta continuation runner for frozen evaluation plans."""
from __future__ import annotations
import argparse,hashlib,json,os,platform,sys,time
from collections import Counter
from pathlib import Path
import jax,jax.numpy as jnp
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');SYSROOT=Path('/home/zhihan/research/02_C1_Toy_GiveWay');HERE=ROOT/'diagnostics/orthoflow3_basin_margin_learning_v1'
for p in (SYSROOT,ROOT):sys.path.insert(0,str(p))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ekey(e):return np.asarray(e,dtype=np.float64).tobytes().hex()
def token(s):return int(hashlib.sha256(s.encode()).hexdigest()[:8],16)
def lines(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def outcome(e,err):
 if err:return 'other_numerical'
 s=e.summary()
 if s['wall_collision'] or s['agent_collision']:return 'collision'
 if s['success']:return 'success'
 if s['deadlock']:return 'safe_deadlock'
 return 'timeout'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();jax.config.update('jax_enable_x64',True);jax.config.update('jax_platform_name','gpu')
 from diagnostics.gphi_training_dataset_v2.finalize_dataset import restore_full
 from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
 from diagnostics.single_segment_recovery_training_v1.state_machine import FiniteHistoryView
 from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
 from shared_control.basis_families import get_basis_family
 from single_integrator.cbf import CBFConfig,barrier_constraints
 from single_integrator.environment import Config,bounded_nominal
 from single_integrator.evaluate import load_policy
 sm={x['state_id']:x for x in json.load(open(HERE/'evaluation_state_manifest.json'))['states']};features=np.load(HERE/'learning_features.npz')['features'];future_root=int(json.load(open(HERE/'evaluation_state_manifest.json'))['future_root_seed'])
 integ=json.load(open(ROOT/'diagnostics/gphi_fixed_d_eta_predictor_v1/integrity_audit.json'));config=Config(**integ['environment']);cbf=CBFConfig(**integ['cbf']);basis=get_basis_family('orthoflow3')
 policy,_=load_policy(SYSROOT/'baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl');sample=jax.jit(jax.vmap(lambda obs,key:policy.sample_actions(obs[None],seed=key)[0]))
 plan=Path(a.plan);out=HERE/'raw'/a.out;out.mkdir(parents=True,exist_ok=True);record=out/(plan.stem+'.jsonl');prior=lines(record) if record.exists() else [];known={(x['state_id'],x['controller'],ekey(x['eta']),int(x['future_index'])) for x in prior};tasks=[x for x in lines(plan) if (x['state_id'],x['controller'],ekey(x['eta']),int(x['future_index'])) not in known]
 started=time.monotonic();steps=0;maxerr=0.;allrows=list(prior)
 for begin in range(0,len(tasks),32):
  chunk=tasks[begin:begin+32];envs=[];etas=[];cur=[];fut=[];errs=[None]*len(chunk);first=[None]*len(chunk);jdef=np.zeros(len(chunk))
  for t in chunk:
   s=sm[t['state_id']];p=Path(s['state_file']);
   if sha(p)!=s['state_sha256']:raise RuntimeError(('state hash',t['state_id']))
   e=restore_full(p,config);base=jax.random.fold_in(jax.random.PRNGKey(int(s['flow_seed'])),int(s['rng_namespace']));cur.append(np.asarray(jax.random.fold_in(base,int(s['absolute_step']))));root=jax.random.fold_in(jax.random.PRNGKey(future_root),token(t['state_id']));fut.append(np.asarray(jax.random.fold_in(root,int(t['future_index']))));envs.append(e);etas.append(np.asarray(t['eta'],float))
  while any(not e.done and errs[i] is None for i,e in enumerate(envs)):
   active=[i for i,e in enumerate(envs) if not e.done and errs[i] is None];obs=np.stack([envs[i].observation() for i in active]);keys=[]
   for i in active:
    s=sm[chunk[i]['state_id']];step=int(envs[i].step_count);keys.append(cur[i] if step==int(s['absolute_step']) else np.asarray(jax.random.fold_in(jnp.asarray(fut[i]),step)))
   acts=np.asarray(sample(jnp.asarray(obs,dtype=jnp.float32),jnp.asarray(np.stack(keys))))
   for loc,i in enumerate(active):
    e=envs[i];s=sm[chunk[i]['state_id']]
    try:
     flow=bounded_nominal(acts[loc],config.max_speed);A,lower,_=barrier_constraints(e.snapshot(),cbf);safe,st1,re1,_=project_velocity_with_retry(flow,A,lower,config.max_speed,cbf);fields=basis.compute(e.positions,e.goals,safe,config.max_speed);exe,st2,re2,_=project_velocity_with_retry(safe+fields.correction(etas[i]),A,lower,config.max_speed,cbf)
     if first[i] is None:
      h,_=StartupAwareFeatureBuilder().build(FiniteHistoryView(e),{'u_flow':flow,'u_safe':safe},config,cbf);er=float(np.max(np.abs(h-features[int(s['dataset_index'])])));maxerr=max(maxerr,er)
      if er>1e-10:raise RuntimeError(('feature replay',er))
      first[i]={'feature_sha256':hashlib.sha256(np.asarray(h,dtype=np.float64).tobytes()).hexdigest(),'first_projection_status':str(st1),'second_projection_status':str(st2),'first_projection_retry':bool(re1),'second_projection_retry':bool(re2)}
     if not np.isfinite(exe).all():raise RuntimeError('invalid nonfinite action')
     jdef[i]+=config.dt*float(np.sum((exe-safe)**2));e.step(exe);steps+=1
    except Exception as ex:errs[i]={'type':type(ex).__name__,'message':str(ex),'step':int(e.step_count)}
  batch=[]
  for i,(t,e) in enumerate(zip(chunk,envs,strict=True)):
   lab=outcome(e,errs[i]);s=sm[t['state_id']];summ=e.summary();batch.append({**t,'outcome':lab,'success':lab=='success','deadlock':lab=='safe_deadlock','timeout':lab=='timeout','collision':lab=='collision','wall_collision':bool(summ['wall_collision']),'agent_collision':bool(summ['agent_collision']),'continuation_steps':int(e.step_count-int(s['absolute_step'])),'terminal_step':int(e.step_count),'J_def':float(jdef[i]),'execution_error':errs[i],'first_step':first[i],'source_group':s['source_group'],'anchor_family':s['anchor_family']})
  allrows+=batch
  with record.open('a') as f:
   for x in batch:f.write(json.dumps(x,sort_keys=True)+'\n')
  if begin==0 or begin+32>=len(tasks) or (begin//32+1)%10==0:print(json.dumps({'done':min(begin+32,len(tasks)),'total':len(tasks),'steps':steps,'max_feature_error':maxerr}),flush=True)
 summary={'plan':str(plan),'new_continuations':len(tasks),'physical_steps':steps,'wall_time_seconds':time.monotonic()-started,'max_feature_replay_error':maxerr,'outcomes':dict(Counter(x['outcome'] for x in allrows)),'devices':[str(x) for x in jax.devices()],'cpu_request':os.environ.get('SLURM_CPUS_PER_TASK'),'host':platform.node()};(out/(plan.stem+'_runtime.json')).write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
