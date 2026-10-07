"""Collect one learner transition then validate the frozen source eta densely."""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np

ROOT=Path('/home/zhihan/research/Basin_C1'); HERE=ROOT/'diagnostics/gphi_dagger_k1_diagnostic_v1'
CAP=ROOT/'diagnostics/strict_deadlock_success_basin_capacity_v1'; PILOT=ROOT/'diagnostics/gphi_closed_loop_pilot_v1'
CHECKPOINT=ROOT/'diagnostics/gphi_strict_deadlock_coverage_retrain_v1/best_strict_deadlock_coverage_checkpoint.npz'
FLOW=Path('/home/zhihan/research/02_C1_Toy_GiveWay/baseline_309_314/checkpoints/seed0/ckpt_0025000.pkl')
sys.path.insert(0,str(PILOT)); sys.path.insert(0,str(ROOT/'diagnostics/gphi_strict_deadlock_burst_length_v1'))
import run_bursts as base
from pilot_common import DeterministicGphi, sha256
from diagnostics.cl_fhcb.closed_loop import DiagnosticCorrector, DiagnosticPhi
from diagnostics.gphi_training_dataset_startup_complete_v1.startup_feature_builder import StartupAwareFeatureBuilder
from diagnostics.gphi_training_dataset_v1.build_states import restore_full, save_full
from diagnostics.success_basin_multimodality.exact_projector import project_velocity_with_retry
from single_integrator.cbf import CBFConfig, barrier_constraints
from single_integrator.environment import Config, bounded_nominal
from single_integrator.evaluate import load_policy

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--shard-index',type=int,required=True); ap.add_argument('--shard-count',type=int,default=4); ap.add_argument('--batch',type=int,default=32); ap.add_argument('--cohort',choices=('historical','fresh6'),default='historical'); a=ap.parse_args()
 jax.config.update('jax_enable_x64',True); jax.config.update('jax_platform_name','gpu')
 src=json.load(open(HERE/'source_manifest.json')); cohort_states=src['training_sources'] if a.cohort=='historical' else src['fresh6_heldout_sources']; states=[s for i,s in enumerate(cohort_states) if i%a.shard_count==a.shard_index]
 suffix='' if a.cohort=='historical' else '_fresh6'; raw_dir=HERE/f'raw{suffix}_k1'; state_dir=HERE/f'states{suffix}_k1'; tuple_dir=HERE/f'tuples{suffix}_k1'
 for d in (raw_dir,state_dir,tuple_dir): d.mkdir(exist_ok=True)
 cap=json.load(open(CAP/'strict_deadlock_manifest.json')); cfg=Config(**cap['environment']); cbf=CBFConfig(**cap['cbf'])
 model=DeterministicGphi(CHECKPOINT); policy,prov=load_policy(FLOW)
 if not prov or prov['evaluation_environment']!=cap['environment']: raise RuntimeError('Flow mismatch')
 sample=jax.jit(jax.vmap(lambda o,k:policy.sample_actions(o[None],seed=k)[0])); fold=jax.jit(jax.vmap(jax.random.fold_in))
 eta=src['frozen_eta_by_state']; started=time.monotonic(); nr=steps=0
 for pos,state in enumerate(states):
  out=raw_dir/f"{state['state_id']}.jsonl"
  if out.is_file() and len(out.read_text().splitlines())==64:
   print(json.dumps({'skip':state['state_id']}),flush=True); continue
  rows=[]
  for begin in range(0,64,a.batch):
   seeds=list(range(95310001+begin,95310001+min(begin+a.batch,64)))
   envs=[restore_full(Path(state['state_file']),cfg) for _ in seeds]
   teachers=[DiagnosticCorrector(DiagnosticPhi(*eta[state['state_id']])) for _ in seeds]
   keys=np.asarray([np.asarray(jax.random.fold_in(jax.random.PRNGKey(seed),state['rng_namespace']),dtype=np.uint32) for seed in seeds])
   obs=np.stack([e.observation() for e in envs]); abs0=np.asarray([e.step_count for e in envs],dtype=np.uint32)
   actions=np.asarray(sample(jnp.asarray(obs),fold(jnp.asarray(keys),jnp.asarray(abs0))))
   features0=[]; safe0=[]; proj0=[]
   for e,act in zip(envs,actions):
    flow=bounded_nominal(act,cfg.max_speed); A,lo,_=barrier_constraints(e.snapshot(),cbf); safe,_,_,_=project_velocity_with_retry(flow,A,lo,cfg.max_speed,cbf)
    feat,_=StartupAwareFeatureBuilder().build(base.FiniteHistoryView(e),{'u_flow':flow,'u_safe':safe},cfg,cbf)
    features0.append(feat); safe0.append(safe); proj0.append((A,lo))
   pred=model(np.stack(features0)).reshape(len(seeds),2,2)
   for i,e in enumerate(envs):
    A,lo=proj0[i]; action,_,_,_=project_velocity_with_retry(safe0[i]+pred[i],A,lo,cfg.max_speed,cbf); e.step(action)
   # Save the learner-visited z1 before applying the teacher.
   for seed,e in zip(seeds,envs):
    sid=f"k1__{state['state_id']}__flow{seed}"; save_full(state_dir/f'{sid}.npz',e)
   # Compute the k1 feature/teacher target, then continue the same eta densely.
   obs1=np.stack([e.observation() for e in envs]); abs1=np.asarray([e.step_count for e in envs],dtype=np.uint32)
   actions1=np.asarray(sample(jnp.asarray(obs1),fold(jnp.asarray(keys),jnp.asarray(abs1))))
   tuple_data=[]
   for i,(seed,e,teacher) in enumerate(zip(seeds,envs,teachers)):
    flow=bounded_nominal(actions1[i],cfg.max_speed); A,lo,_=barrier_constraints(e.snapshot(),cbf); safe,_,_,_=project_velocity_with_retry(flow,A,lo,cfg.max_speed,cbf)
    feature,structured=StartupAwareFeatureBuilder().build(base.FiniteHistoryView(e),{'u_flow':flow,'u_safe':safe},cfg,cbf)
    raw=teacher(np.asarray(obs1[i],dtype=float),safe,cfg.max_speed); exec_action,_,_,_=project_velocity_with_retry(safe+raw,A,lo,cfg.max_speed,cbf); target=(exec_action-safe).reshape(4)
    tuple_data.append((flow,safe,exec_action,target,feature,structured))
   # Batched teacher continuation, beginning with the already-computed k1 action.
   jdef=np.zeros(len(envs)); local_steps=np.zeros(len(envs),dtype=int)
   for i,e in enumerate(envs):
    safe=tuple_data[i][1]; action=tuple_data[i][2]; e.step(action); jdef[i]+=cfg.dt*float(np.sum((action-safe)**2)); local_steps[i]+=1
   while any(not e.done for e in envs):
    ob=np.zeros((len(envs),2,10),dtype=np.float32); ab=np.zeros(len(envs),dtype=np.uint32)
    for i,e in enumerate(envs):
     if not e.done: ob[i]=e.observation(); ab[i]=e.step_count
    acts=np.asarray(sample(jnp.asarray(ob),fold(jnp.asarray(keys),jnp.asarray(ab))))
    for i,(e,teacher) in enumerate(zip(envs,teachers)):
     if e.done: continue
     flow=bounded_nominal(acts[i],cfg.max_speed); A,lo,_=barrier_constraints(e.snapshot(),cbf); safe,_,_,_=project_velocity_with_retry(flow,A,lo,cfg.max_speed,cbf)
     raw=teacher(np.asarray(ob[i],dtype=float),safe,cfg.max_speed); action,_,_,_=project_velocity_with_retry(safe+raw,A,lo,cfg.max_speed,cbf)
     e.step(action); jdef[i]+=cfg.dt*float(np.sum((action-safe)**2)); local_steps[i]+=1
   for i,(seed,e) in enumerate(zip(seeds,envs)):
    sid=f"k1__{state['state_id']}__flow{seed}"; flow,safe,action,target,feature,st=tuple_data[i]
    tuple_path=tuple_dir/f'{sid}.npz'
    np.savez_compressed(tuple_path,feature=feature,target=target,target_action=action.reshape(4),observation=st['observation'],positions=st['positions'],velocities=st['velocities'],goal_relative=st['goal_relative'],relative_position=st['relative_position'],relative_velocity=st['relative_velocity'],u_flow=st['u_flow'],u_safe=st['u_safe'],B_goal=st['B_goal'],B_rel=st['B_rel'],goal_error_history=st['goal_error_history'],recent_progress=st['recent_progress'],first_projection_delta=st['projection_delta'],first_projection_linear_residuals=st['linear_residuals'])
    summary=e.summary(); outcome='collision' if summary['wall_collision'] or summary['agent_collision'] else 'success' if summary['success'] else 'deadlock' if summary['deadlock'] else 'timeout'
    state_path=state_dir/f'{sid}.npz'
    rows.append({'case_id':state['case_id'],'source_state_id':state['state_id'],'state_id':sid,'flow_seed':seed,'query_step':state['query_step']+1,'outcome':outcome,'teacher_valid':outcome=='success','continuation_steps':int(local_steps[i]),'J_def':float(jdef[i]),'state_file':str(state_path.relative_to(HERE)),'state_sha256':sha256(state_path),'tuple_file':str(tuple_path.relative_to(HERE)),'tuple_sha256':sha256(tuple_path),'target_norm':float(np.linalg.norm(target)),'record_complete':True})
    nr+=1; steps+=int(local_steps[i])
  tmp=out.with_suffix('.tmp'); tmp.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows)); os.replace(tmp,out)
  print(json.dumps({'completed':state['state_id'],'valid':sum(r['teacher_valid'] for r in rows),'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
 (HERE/f'runtime_collect{suffix}_shard{a.shard_index}.json').write_text(json.dumps({'cohort':a.cohort,'shard':a.shard_index,'states':len(states),'new_tuples':nr,'physical_steps':steps,'elapsed_s':time.monotonic()-started,'device':[str(x) for x in jax.devices()]},indent=2)+'\n')
if __name__=='__main__': main()
