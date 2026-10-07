#!/usr/bin/env python3
"""Frozen joint-4A DB rollout runner with persistent OrthoFlow3 eta."""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,hashlib,json,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
HERE=Path(__file__).parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
import jax
from diagnostics.double_bottleneck_eta_basis_redesign.tools import run_rollouts as old

def run_one_with_jdef(policy,dataset,episode,job,scale):
  """Byte-for-byte frozen rollout loop plus the canonical J_def accumulator.

  J_def = dt * sum_t ||u_exec-u_safe||^2.  This adds an observer only; the
  environment, both projections, correction, RNG keys, and stopping rules are
  the authoritative DB implementation imported as ``old``.
  """
  config=old.Config(**dataset.config);env=old._initialize_env(config,episode);projector=old.CertifiedHardSafetyFilter();theta=np.asarray(job['theta'],float)
  key=jax.random.fold_in(jax.random.PRNGKey(int(job['seed'])),int(job['rollout_id']))
  raw_norms=[];executable_norms=[];removal_norms=[];removal_ratios=[];cosines=[];more_half=[];first_status=Counter();second_status=Counter();min_wall=float(env.distances()[0].min());min_pair=float(env.distances()[1].min());positions=[env.positions.copy()];final_info=None;retry_events=[];started=time.perf_counter();jdef=0.
  for step in range(config.max_steps):
   raw_flow=np.asarray(policy.sample_actions(env.observation()[None],jax.random.fold_in(key,step))[0],float);u_flow=old._radial_bound64(raw_flow,config.max_speed);snapshot=env.snapshot();first,retry=old.project_with_retries(projector,snapshot,u_flow,step=step,stage='first_projection')
   if retry is not None:retry_events.append(retry)
   if first is None:return old.numerical_failure_result(job,env,raw_norms,retry_events,min_wall,min_pair,started)
   u_safe=np.asarray(first.velocity,float);g=old.correction(job['representation'],theta,env.positions,env.goals,u_safe,config.max_speed,scale);second,retry=old.project_with_retries(projector,snapshot,u_safe+g,step=step,stage='second_projection')
   if retry is not None:retry_events.append(retry)
   if second is None:return old.numerical_failure_result(job,env,raw_norms,retry_events,min_wall,min_pair,started)
   u_exec=np.asarray(second.velocity,float);executable=u_exec-u_safe;raw_norm=float(np.linalg.norm(g));executable_norm=float(np.linalg.norm(executable));removal_norm=float(np.linalg.norm(u_exec-(u_safe+g)));jdef+=float(config.dt*np.sum(executable*executable));raw_norms.append(raw_norm);executable_norms.append(executable_norm);removal_norms.append(removal_norm);removal_ratios.append(removal_norm/max(raw_norm,1e-12));cosines.append(float(np.sum(g*executable)/max(raw_norm*executable_norm,1e-12)));more_half.append(raw_norm>1e-12 and removal_norm>.5*raw_norm);first_status[str(first.status)]+=1;second_status[str(second.status)]+=1
   _,_,done,info=env.step(u_exec);positions.append(env.positions.copy());min_wall=min(min_wall,float(info['min_swept_wall_distance']));min_pair=min(min_pair,float(info['min_swept_agent_distance']));final_info=info
   if done:break
  termination=str(final_info['termination']);wall=bool(final_info['wall_collision']);agent=bool(final_info['agent_collision'])
  return {**{k:job[k] for k in ('job_id','stage','representation','parameter_id','eta_index','sample_type','theta','episode_id','population','set','family_id','regime','seed','rollout_id','baseline_outcome','center_eta_index','local_index') if k in job},'success':termination=='success','termination':termination,'outcome':old.outcome_name(termination,wall,agent),'episode_steps':len(raw_norms),'J_def':jdef,'wall_collision':wall,'agent_collision':agent,'minimum_wall_clearance':min_wall,'minimum_agent_clearance':min_pair,'final_goal_errors':np.linalg.norm(env.goals-env.positions,axis=-1).tolist(),'coordination_mode':old.infer_coordination_mode(np.asarray(positions),env.goals,config),'correction':{'raw_norm':old.summary(raw_norms),'executable_norm':old.summary(executable_norms),'projection_removal_norm':old.summary(removal_norms),'removal_ratio':old.summary(removal_ratios),'raw_executable_cosine':old.summary(cosines),'more_than_half_removed_fraction':float(np.mean(more_half)),'first_status_counts':dict(first_status),'second_status_counts':dict(second_status)},'wall_seconds':time.perf_counter()-started,'scientific_outcome_valid':True,'solver_retry_occurred':bool(retry_events),'solver_retry_events':retry_events}

def load_states():
 x=json.load(open(HERE/'db_state_split.json'))
 states={s['state_id']:s for s in x['states']}
 if (HERE/'fresh_state_manifest.json').exists():
  for s in json.load(open(HERE/'fresh_state_manifest.json'))['states']:states[s['state_id']]=s
 # Existing four geometry states are external transform-development states.
 for s in json.load(open(HERE.parent/'orthoflow3_general_basin_geometry_v1/double_bottleneck_state_panel.json')):states[s['state_id']]=s
 return states
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan',required=True);ap.add_argument('--shard',type=int,required=True);args=ap.parse_args()
 states=load_states();tasks=[json.loads(x) for x in open(HERE/'plans'/args.plan/f'shard{args.shard}.jsonl')]
 if not tasks:
  (HERE/'raw').mkdir(exist_ok=True)
  (HERE/'raw'/f'{args.plan}_{args.shard}_runtime.json').write_text(json.dumps({'plan':args.plan,'shard':args.shard,'new_continuations':0,'physical_steps':0,'wall_seconds':0.,'empty_plan':True},indent=2)+'\n')
  return
 for p,h in old.EXPECTED.items():assert old.sha(p)==h,(p,'integrity')
 paths=sorted({states[t['state_id']]['dataset'] for t in tasks});datasets={p:old.FlowBC4ADataset(p,'all') for p in paths}
 first=datasets[paths[0]];policy,_=old.load_checkpoint(old.CHECKPOINT,first.environment_fingerprint)
 out=HERE/'raw';out.mkdir(exist_ok=True);dest=out/f'{args.plan}_{args.shard}.jsonl';known={}
 for p in out.glob('*.jsonl'):
  for line in open(p):
   r=json.loads(line);known[(r['state_id'],r['controller'],r.get('mode_id'),r['future_index'])]=r
 started=time.time();new=0;steps=0
 for t in tasks:
  key=(t['state_id'],t['controller'],t.get('mode_id'),t['future_index'])
  if key in known:continue
  s=states[t['state_id']];dataset=datasets[s['dataset']];episode=dataset.by_family[s['family_id']][0];ns=s['rng_namespace']
  future=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(s['future_root']),ns),t['future_index'])
  current=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(s['initial_flow_root']),ns),0)
  class FixedCurrent:
   def __init__(self):self.step=0;self.h=None;self.conditioning=None
   def sample_actions(self,obs,unused):
    stepkey=current if self.step==0 else jax.random.fold_in(future,self.step);action=policy.sample_actions(obs,stepkey)
    if self.step==0:
     o=np.asarray(obs,dtype=np.float64);a=np.asarray(action,dtype=np.float64);self.h=hashlib.sha256(o.tobytes()+a.tobytes()).hexdigest()
     self.conditioning={'state_id':s['state_id'],'observation':o.tolist(),'current_flow_action':a.tolist(),'h_feature':np.r_[o.ravel(),a.ravel()].tolist(),'h_sha256':self.h,'initial_positions':episode.initial_positions.tolist(),'initial_velocities':episode.initial_velocities.tolist(),'episode_path':str(episode.path),'source_group':s['source_group'],'config':dataset.config}
    self.step+=1;return action
  wrapper=FixedCurrent();job={'job_id':t['probe_id'],'stage':t['phase'],'representation':'P1-OrthoFlow3','theta':t['eta'],'seed':s['future_root'],'rollout_id':ns,'episode_id':s['state_id'],'family_id':s['family_id']}
  r=run_one_with_jdef(wrapper,dataset,episode,job,3.303687238760696) if t['phase']!='transform_dev_pilot' else old.run_one(wrapper,dataset,episode,job,3.303687238760696)
  cp=out/f"conditioning_{s['state_id']}.json"
  try:
   with cp.open('x') as f:f.write(json.dumps(wrapper.conditioning,indent=2)+'\n')
  except FileExistsError:
   assert json.load(open(cp))['h_sha256']==wrapper.h
  r.update(t,h_conditioning_identifier=wrapper.h,source_group=s['source_group'],scenario='DoubleBottleneck_4A',future_root_seed=s['future_root'],current_root_seed=s['initial_flow_root'])
  with open(dest,'a') as f:f.write(json.dumps(r)+'\n')
  new+=1;steps+=int(r.get('episode_steps',0));print(json.dumps({k:r.get(k) for k in ('state_id','controller','mode_id','future_index','outcome','wall_seconds','scientific_outcome_valid')}),flush=True)
  if not r['scientific_outcome_valid']:raise RuntimeError('Projection execution failure; do not interpret')
 (HERE/'raw'/f'{args.plan}_{args.shard}_runtime.json').write_text(json.dumps({'plan':args.plan,'shard':args.shard,'new_continuations':new,'physical_steps':steps,'wall_seconds':time.time()-started},indent=2)+'\n')
if __name__=='__main__':main()
