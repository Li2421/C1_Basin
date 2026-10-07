"""v11 development-policy recovery with separately auditable agent precursors.

Timeout anchors are generic ``uniform_recovery``.  Only collision rollouts
contribute ``targeted_agent`` samples, and then exclusively from valid states
before contact.  No test trajectory is loaded.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
import jax
import numpy as np
from new_benchmark_common.dataset import RecoveryAudit, TRAJECTORY_SCHEMA, _digest, _jsonable
from new_benchmark_common.macflow import load_checkpoint, sample_bounded_actions
from .environment import Config, FourWayIntersectionEnv
from .lane_local import world_to_local
from .protocol import FourWayScenario

def _initial(root, row):
    with np.load(Path(root)/row['file'],allow_pickle=False) as d:return json.loads(str(d['initial_state_json'].item()))
def _action(agent,obs,key,limit):
    u=np.asarray(sample_bounded_actions(agent,obs[None],key)[0],dtype=np.float64); n=np.linalg.norm(u,axis=-1,keepdims=True)
    # Match the frozen development evaluator exactly; this avoids a tiny
    # bound-rounding perturbation accumulating into a different stochastic
    # policy rollout before source states are selected.
    return u*np.minimum(1.,(limit-1e-10)/np.maximum(n,1e-12))
def _write(path,c,initial,audit,rid,source,kind):
    states=np.asarray(c.states,dtype=np.float64);obs=np.asarray(c.observations,dtype=np.float32);act=np.asarray(c.actions,dtype=np.float32);digest=_digest(states,obs,act)
    meta=dict(c.metadata);meta.update(success=True,terminal_reason=c.terminal_reason,rollout_id=rid,split='train',source=source,trajectory_digest=digest,recovery_anchor_kind=kind)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,schema=np.asarray(TRAJECTORY_SCHEMA),states=states,observations=obs,actions=act,initial_state_json=np.asarray(json.dumps(_jsonable(initial),sort_keys=True)),metadata_json=np.asarray(json.dumps(_jsonable(meta),sort_keys=True)),recovery_audit_json=np.asarray(json.dumps(_jsonable(audit.__dict__),sort_keys=True)))
    return {'file':str(path.relative_to(path.parents[2])),'rollout_id':rid,'split':'train','source':source,'trajectory_digest':digest,'length':len(act)}

def collect(source,output,checkpoint,*,seed=11123,late_anchors=50,goal_anchors=24,collision_offsets=(3,6,10,16,24)):
    source,output,checkpoint=map(Path,(source,output,checkpoint))
    if output.exists():raise FileExistsError(output)
    m=json.loads((source/'manifest.json').read_text());cfg=Config(**{k:v for k,v in m['scenario_config'].items() if k in Config.__dataclass_fields__})
    agent,_=load_checkpoint(checkpoint,expected_environment_fingerprint=cfg.fingerprint);scenario=FourWayScenario(cfg)
    shutil.copytree(source,output) # opaque frozen-test provenance; test files are not read below.
    dev=[r for r in m['files'] if r['split']=='dev' and r['source']=='nominal'];adds=[];rollouts=[];serial=attempted=accepted=invalid=failed=transitions=0; source_counts={'uniform_recovery':0,'targeted_agent':0}
    for case,row in enumerate(dev):
        initial=_initial(source,row);env=FourWayIntersectionEnv(cfg);env.reset(np.asarray(initial['positions']),np.asarray(initial['velocities']));snaps=[env.augmented_state()];key=jax.random.PRNGKey(seed+case);terminal='timeout';last_info={}
        for time in range(cfg.max_steps):
            _,_,done,info=env.step(_action(agent,env.observation(),jax.random.fold_in(key,time),cfg.max_speed));terminal=str(info['termination']);last_info=info
            if done:break
            snaps.append(env.augmented_state())
        candidates=[]
        if terminal=='timeout':
            pool=np.arange(int(.60*(len(snaps)-1)),len(snaps),dtype=int); late=pool[np.unique(np.linspace(0,len(pool)-1,min(late_anchors,len(pool)),dtype=int))]
            goals=[]
            for t,snap in enumerate(snaps):
                rem=env.goals-np.asarray(snap['positions']);near=np.linalg.norm(rem,axis=1)<=.65;beyond=world_to_local(rem)[:,0]<-cfg.goal_tolerance
                if bool(np.any(near|beyond)):goals.append(t)
            gi=np.unique(np.linspace(0,len(goals)-1,min(goal_anchors,len(goals)),dtype=int)) if goals else np.empty(0,dtype=int)
            candidates=[(int(t),'uniform_recovery','late_timeout_uniform',None,None) for t in late]
            candidates += [(int(goals[i]),'uniform_recovery','goal_settling_or_overshoot',None,None) for i in gi]
        elif terminal=='collision' and bool(last_info.get('agent_collision')):
            pair_index=int(np.argmin(np.asarray(last_info['swept_agent_distances'])));pair=tuple(int(x) for x in env.pair_indices[pair_index]);identity=f'{pair[0]}-{pair[1]}';distance=float(last_info['swept_agent_distances'][pair_index]);end=len(snaps)
            # snapshots exclude the invalid/contact state: offsets are strictly pre-collision.
            for offset in collision_offsets:
                t=end-int(offset)
                if t>=0:candidates.append((t,'targeted_agent','agent_precollision',identity,{'offset':int(offset),'distance':distance}))
        selected={t:(source_name,kind,identity,extra) for t,source_name,kind,identity,extra in candidates}
        accepted_rows=[]
        for time,(source_name,kind,identity,extra) in sorted(selected.items()):
            snap=snaps[time];state={'positions':np.asarray(snap['positions'],dtype=np.float64),'velocities':np.asarray(snap['last_applied_velocity'],dtype=np.float64),'split':'dev','recovery':True,'source_time':time}
            attempted+=1;perturb_seed=int(np.random.SeedSequence([seed,case,time,0 if source_name=='uniform_recovery' else 1]).generate_state(1)[0]);perturbed=scenario.perturb_state(state,np.random.default_rng(perturb_seed))
            if not scenario.valid_state(perturbed):invalid+=1;continue
            c=scenario.expert(perturbed,np.random.default_rng(perturb_seed))
            if not c.success:failed+=1;continue
            audit=RecoveryAudit(source_rollout_id=f'dev_policy_world_v10_{case:03d}',source_split='dev',source_time=time,collision_type='agent' if source_name=='targeted_agent' else None,collision_identity=identity,distance_to_collision=None if extra is None else extra['distance'],perturbation_seed=perturb_seed,expert_success=True)
            rid=f'train_world_v11_{source_name}_{serial:06d}';entry=_write(output/'rollouts'/'train'/f'{rid}.npz',c,perturbed,audit,rid,source_name,kind);adds.append(entry);serial+=1;accepted+=1;transitions+=int(entry['length']);source_counts[source_name]+=1;accepted_rows.append({'time':time,'source':source_name,'kind':kind,'collision_identity':identity,**(extra or {})})
        rollouts.append({'source_rollout_id':f'dev_policy_world_v10_{case:03d}','terminal':terminal,'valid_preterminal_states':len(snaps),'candidate_count':len(selected),'accepted':accepted_rows})
    changed=json.loads((output/'manifest.json').read_text());changed['files'].extend(adds)
    for name,count in source_counts.items():changed['counts']['source'][name]+=count
    changed['counts']['split']['train']+=accepted
    report={'policy_checkpoint':str(checkpoint),'source':'development nominal v10 policy trajectories only','destination_split':'train','source_split':'dev','timeout_anchor_policy':{'late_fraction_start':.60,'late_anchors':late_anchors,'goal_anchors':goal_anchors},'agent_precollision_offsets':list(collision_offsets),'perturbation_distribution':{'position_std':scenario.perturb_position_std,'velocity_std':scenario.perturb_velocity_std},'attempted':attempted,'accepted':accepted,'accepted_by_source':source_counts,'accepted_transitions':transitions,'invalid_perturbation':invalid,'expert_failed_skipped':failed,'source_rollouts':rollouts,'test_opened':False}
    changed['extra_report']['world_timeout_agent_recovery_v11']=report;(output/'manifest.json').write_text(json.dumps(changed,indent=2,sort_keys=True)+'\n')
    final={**report,'base_source':str(source),'output':str(output),'output_manifest_sha256':hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest()};(output/'world_timeout_agent_recovery_report.json').write_text(json.dumps(final,indent=2,sort_keys=True)+'\n');return final
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');p.add_argument('checkpoint');p.add_argument('--seed',type=int,default=11123);a=p.parse_args();print(json.dumps(collect(a.source,a.output,a.checkpoint,seed=a.seed),indent=2))
