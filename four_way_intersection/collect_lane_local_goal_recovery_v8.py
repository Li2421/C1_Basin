"""v8 development-only lane-local general + goal-recovery acquisition.

Goal-near/overshoot selection is a physical state predicate over position and
goal displacement.  It is audit metadata only, never a policy feature or a
coordination/order label.
"""
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path
import jax,numpy as np
from new_benchmark_common.dataset import RecoveryAudit,TRAJECTORY_SCHEMA,_digest,_jsonable
from new_benchmark_common.macflow import load_checkpoint,sample_bounded_actions
from .environment import Config
from .lane_local import FourWayLaneLocalEnv,local_to_world,representation_fingerprint,world_to_local
from .lane_local_protocol import FourWayLaneLocalScenario

def _initial(root,row):
 with np.load(root/row['file'],allow_pickle=False) as d:return json.loads(str(d['initial_state_json'].item()))
def _bounded(agent,obs,key,limit):
 u=np.asarray(sample_bounded_actions(agent,obs[None],key)[0],dtype=np.float64);n=np.linalg.norm(u,axis=-1,keepdims=True);return u*np.minimum(1.,(limit-1e-8)/np.maximum(n,1e-12))
def _write(path,c,initial,audit,rid,kind):
 states=np.asarray(c.states,dtype=np.float64);obs=np.asarray(c.observations,dtype=np.float32);act=np.asarray(c.actions,dtype=np.float32);digest=_digest(states,obs,act)
 meta=dict(c.metadata);meta.update(success=True,terminal_reason=c.terminal_reason,rollout_id=rid,split='train',source='uniform_recovery',trajectory_digest=digest,recovery_anchor_kind=kind)
 path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,schema=np.asarray(TRAJECTORY_SCHEMA),states=states,observations=obs,actions=act,initial_state_json=np.asarray(json.dumps(_jsonable(initial),sort_keys=True)),metadata_json=np.asarray(json.dumps(_jsonable(meta),sort_keys=True)),recovery_audit_json=np.asarray(json.dumps(_jsonable(audit.__dict__),sort_keys=True)))
 return {'file':str(path.relative_to(path.parents[2])),'rollout_id':rid,'split':'train','source':'uniform_recovery','trajectory_digest':digest,'length':len(act)}

def collect(source,output,checkpoint,*,seed=8125,uniform_anchors=30,goal_anchors=12):
 source,output,checkpoint=Path(source),Path(output),Path(checkpoint)
 if output.exists():raise FileExistsError(output)
 m=json.loads((source/'manifest.json').read_text());cfg=Config(**{k:v for k,v in m['scenario_config'].items() if k in Config.__dataclass_fields__});fp=representation_fingerprint(cfg)
 agent,_=load_checkpoint(checkpoint,expected_environment_fingerprint=fp);scenario=FourWayLaneLocalScenario(cfg)
 shutil.copytree(source,output) # Copy provenance only; test files are never loaded below.
 dev=[r for r in m['files'] if r['split']=='dev' and r['source']=='nominal'];adds=[];provenance=[];serial=attempted=accepted=invalid=failed=transitions=0
 for case,row in enumerate(dev):
  initial=_initial(source,row);env=FourWayLaneLocalEnv(cfg);env.reset(np.asarray(initial['positions']),np.asarray(initial['velocities']));snaps=[env.augmented_state()];terminal='timeout';key=jax.random.PRNGKey(seed+case)
  for step in range(cfg.max_steps):
   local=_bounded(agent,env.observation(),jax.random.fold_in(key,step),cfg.max_speed);_,_,done,info=env.step(local_to_world(local));terminal=str(info['termination'])
   if done:break
   snaps.append(env.augmented_state())
  uniform=np.unique(np.linspace(0,len(snaps)-1,min(uniform_anchors,len(snaps)),dtype=int))
  goal_candidates=[]
  for t,snap in enumerate(snaps):
   p=np.asarray(snap['positions']);rem=env.goals-p;err=np.linalg.norm(rem,axis=1);local_rem=world_to_local(rem)
   # Include near-goal waiting/settling states and post-goal forward overshoot.
   if bool(np.any(err<=.55) or np.any(local_rem[:,0]<-cfg.goal_tolerance)):goal_candidates.append(t)
  dense=np.unique(np.linspace(0,len(goal_candidates)-1,min(goal_anchors,len(goal_candidates)),dtype=int)) if goal_candidates else np.empty(0,dtype=int)
  candidates=[(int(t),'uniform_full_rollout') for t in uniform]+[(int(goal_candidates[i]),'goal_near_or_overshoot') for i in dense]
  # An overlap is represented once, with physical goal label preferred.
  selected={t:k for t,k in candidates}
  local_rows=[]
  for time,kind in sorted(selected.items()):
   snap=snaps[time];state={'positions':np.asarray(snap['positions'],dtype=np.float64),'velocities':np.asarray(snap['last_applied_velocity'],dtype=np.float64),'split':'dev','recovery':True,'source_time':time}
   attempted+=1;perturb_seed=int(np.random.SeedSequence([seed,case,time,0 if kind=='uniform_full_rollout' else 1]).generate_state(1)[0]);perturbed=scenario.perturb_state(state,np.random.default_rng(perturb_seed))
   if not scenario.valid_state(perturbed):invalid+=1;continue
   c=scenario.expert(perturbed,np.random.default_rng(perturb_seed))
   if not c.success:failed+=1;continue
   audit=RecoveryAudit(source_rollout_id=f'dev_policy_lane_v7_{case:03d}',source_split='dev',source_time=time,collision_type=None,collision_identity=None,distance_to_collision=None,perturbation_seed=perturb_seed,expert_success=True)
   rid=f'train_lane_v8_recovery_{serial:06d}';added=_write(output/'rollouts'/'train'/f'{rid}.npz',c,perturbed,audit,rid,kind);adds.append(added);serial+=1;accepted+=1;transitions+=int(added['length']);local_rows.append({'time':time,'kind':kind,'accepted':True})
  provenance.append({'source_rollout_id':f'dev_policy_lane_v7_{case:03d}','nominal_rollout_id':row['rollout_id'],'terminal':terminal,'valid_preterminal_states':len(snaps),'uniform_times':uniform.tolist(),'goal_candidate_count':len(goal_candidates),'selected_goal_times':[int(goal_candidates[i]) for i in dense],'accepted':local_rows})
 changed=json.loads((output/'manifest.json').read_text());changed['files'].extend(adds);changed['counts']['source']['uniform_recovery']+=accepted;changed['counts']['split']['train']+=accepted
 report={'policy_checkpoint':str(checkpoint),'source':'development nominal lane-local policy trajectories only','destination_split':'train','source_split':'dev','uniform_anchors_per_rollout':uniform_anchors,'goal_near_or_overshoot_anchors_per_rollout':goal_anchors,'perturbation_distribution':{'position_std':scenario.perturb_position_std,'velocity_std':scenario.perturb_velocity_std},'attempted':attempted,'accepted':accepted,'accepted_transitions':transitions,'invalid_perturbation':invalid,'expert_failed_skipped':failed,'source_rollouts':provenance,'test_opened':False}
 changed['extra_report']['lane_local_goal_recovery_v8']=report;(output/'manifest.json').write_text(json.dumps(changed,indent=2,sort_keys=True)+'\n');final={**report,'base_source':str(source),'output':str(output),'output_manifest_sha256':hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()};(output/'lane_local_goal_recovery_report.json').write_text(json.dumps(final,indent=2,sort_keys=True)+'\n');return final
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');p.add_argument('checkpoint');p.add_argument('--seed',type=int,default=8125);a=p.parse_args();print(json.dumps(collect(a.source,a.output,a.checkpoint,seed=a.seed),indent=2))
