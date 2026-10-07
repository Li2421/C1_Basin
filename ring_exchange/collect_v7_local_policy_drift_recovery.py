"""v7 local-frame counterpart of dev-policy full-trajectory drift recovery."""
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path
import jax,numpy as np
from new_benchmark_common.dataset import RecoveryAudit,TRAJECTORY_SCHEMA,_digest,_jsonable
from new_benchmark_common.macflow import load_checkpoint,sample_bounded_actions
from .environment import LocalFrameConfig,RingExchangeEnv
from .local_frame import local_actions_to_world,local_observation
from .protocol_v6 import RingExchangeLocalFrameScenario

def _write(path,c,initial,rid,audit):
 s=np.asarray(c.states);o=np.asarray(c.observations,dtype=np.float32);a=np.asarray(c.actions,dtype=np.float32);d=_digest(s,o,a)
 m=dict(c.metadata);m.update(success=True,terminal_reason=c.terminal_reason,rollout_id=rid,split='train',source='uniform_recovery',trajectory_digest=d)
 path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,schema=np.asarray(TRAJECTORY_SCHEMA),states=s,observations=o,actions=a,
  initial_state_json=np.asarray(json.dumps(_jsonable(initial),sort_keys=True)),metadata_json=np.asarray(json.dumps(_jsonable(m),sort_keys=True)),recovery_audit_json=np.asarray(json.dumps(_jsonable(audit.__dict__),sort_keys=True)))
 return {'file':str(path.relative_to(path.parents[2])),'rollout_id':rid,'split':'train','source':'uniform_recovery','trajectory_digest':d,'length':int(len(a))}

def collect(source,output,checkpoint,*,seed=7125,anchors=30):
 source,output,checkpoint=Path(source),Path(output),Path(checkpoint)
 if output.exists():raise FileExistsError(output)
 manifest=json.loads((source/'manifest.json').read_text());shutil.copytree(source,output);cfg=LocalFrameConfig()
 agent,_=load_checkpoint(checkpoint,expected_environment_fingerprint=manifest['environment_fingerprint']);scenario=RingExchangeLocalFrameScenario(perturb_position_std=.035,perturb_velocity_std=.025)
 dev=[r for r in manifest['files'] if r['split']=='dev' and r['source']=='nominal'];records=[];rows=[];attempted=accepted=invalid=failed=serial=0
 for ix,row in enumerate(dev):
  with np.load(source/row['file'],allow_pickle=False) as d:initial=json.loads(str(d['initial_state_json'].item()))
  env=RingExchangeEnv(cfg);env.reset(initial['positions'],velocities=initial['velocities'],goals=initial['goals']);snapshots=[env.augmented_state()];term='timeout'
  for step in range(cfg.max_steps):
   key=jax.random.fold_in(jax.random.PRNGKey(seed+ix),step);obs=local_observation(env.positions,env.velocities,env.goals,cfg)
   local=np.asarray(sample_bounded_actions(agent,obs[None],key)[0],dtype=np.float64);world=local_actions_to_world(local,env.positions)
   n=np.linalg.norm(world,axis=-1,keepdims=True);world*=np.minimum(1.,(cfg.max_speed-1e-8)/np.maximum(n,1e-12));_,_,done,info=env.step(world);term=str(info['termination'])
   if done:break
   snapshots.append(env.augmented_state())
  chosen=np.unique(np.linspace(0,len(snapshots)-1,min(anchors,len(snapshots)),dtype=int));made=0
  for time in chosen:
   q=snapshots[int(time)];state={'positions':np.asarray(q['positions']),'velocities':np.asarray(q['last_applied_velocity']),'goals':np.asarray(q['goals']),'split':'dev','recovery':True,'source_time':int(time)}
   attempted+=1;ps=int(np.random.SeedSequence([seed,ix,int(time)]).generate_state(1)[0]);pert=scenario.perturb_state(state,np.random.default_rng(ps))
   if not scenario.valid_state(pert):invalid+=1;continue
   c=scenario.expert(pert,np.random.default_rng(ps))
   if not c.success:failed+=1;continue
   audit=RecoveryAudit(f'dev_policy_v7_local_drift_{ix:03d}','dev',int(time),None,None,None,ps,True);rid=f'train_uniform_drift_v7_local_{serial:06d}'
   records.append(_write(output/'rollouts'/'train'/f'{rid}.npz',c,pert,rid,audit));serial+=1;accepted+=1;made+=1
  rows.append({'source_rollout_id':f'dev_policy_v7_local_drift_{ix:03d}','nominal_rollout_id':row['rollout_id'],'terminal':term,'valid_policy_states':len(snapshots),'uniform_anchor_count':len(chosen),'accepted':made})
 manifest=json.loads((output/'manifest.json').read_text());manifest['files'].extend(records);manifest['counts']['source']['uniform_recovery']+=accepted;manifest['counts']['split']['train']+=accepted
 report={'base_source':str(source),'policy_checkpoint':str(checkpoint),'representation':'radial_tangential_local_v7','source':'development_policy_full_trajectory_uniform_anchors_only','anchors_per_rollout':anchors,'attempted':attempted,'accepted':accepted,'invalid_perturbation':invalid,'expert_failed_skipped':failed,'rollouts':rows,'test_opened':False}
 manifest['extra_report']['v7_local_policy_drift_uniform_recovery']=report;(output/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');report={**report,'output':str(output),'output_manifest_sha256':hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()};(output/'v7_local_policy_drift_uniform_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');p.add_argument('checkpoint');p.add_argument('--anchors',type=int,default=30);a=p.parse_args();print(json.dumps(collect(a.source,a.output,a.checkpoint,anchors=a.anchors),indent=2))
