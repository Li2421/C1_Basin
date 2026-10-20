"""Exact terminal/unknown reuse when the selected trigger never activates."""
from common import *
from trigger import Trigger
from continuing_rhr import execute as original_execute
from types import SimpleNamespace

def inactive(raw,rule,tolerance):
 assert sha(raw['trace_path'])==raw['trace_sha256'];z=np.load(raw['trace_path']);positions=z['positions'];velocities=z['velocities'];goals=z['goals'];t=Trigger(rule);total=len(positions) if raw['numerical_failure'] else len(positions)-1
 for step in range(total):
  env=SimpleNamespace(step_count=step,positions=positions[step],velocities=velocities[step],goals=goals,config=SimpleNamespace(goal_tolerance=tolerance))
  if t.update(env).any():return False,step
 return True,total

def attach_references(p,rows):
 states=read(R32/'EVAL_STATES.json');rc=read(R32/'main_CASES.json');hc=read(R35/'horizon_CASES.json');nc=read(R33/'main_CASES.json');gc=read(HERE/'geometry_controls_CASES.json');indices={}
 for label,cases,root in [('state',rc,R32/'main/raw'),('horizon',hc,R35/'horizon/raw'),('novel',nc,R33/'main/raw'),('geometry',gc,HERE/'geometry_controls/raw')]:
  for c in cases:
   if c['arm']!='safety_baseline':continue
   if label=='geometry' and c['cohort']!='same_state':continue
   key=(c['base_controller_name'],c['state_uid'],c['seed']) if label in ['state','novel','geometry'] else (c['base_controller_name'],c['state_index'],c['seed'],c['H']);indices[label,key]=root/f"{c['rollout_uid']}.json"
 proofs=[]
 for c in rows:
  label=c['subunit'];name=c['base_controller_name'];i=c['state_index'];q=c['seed']
  if label in ['D1','D2']:key=('state',(name,states[i if label=='D1' else 64+i]['state_uid'],q))
  elif label.startswith('D3'):key=('horizon',(name,i,q,c['H']))
  elif label.startswith('D4'):key=('geometry',(name,c['state_uid'],q))
  else:key=('novel',(name,c['state_uid'],q))
  path=indices[key];raw=read(path);s=next(s for s in p['states']if s['state_uid']==c['state_uid']);z=np.load(raw['trace_path']);assert all(np.array_equal(z[k][0] if k!='goals' else z[k],s['physical'][k])for k in ['positions','velocities','goals']);assert raw['case']['seed']==q
  tolerance=read(R33/'main_protocol.json')['native_manifest']['scenario_config']['goal_tolerance'] if c['kind']!='Flow' else __import__('factor_runtime').config_for(name,c['H'],c['r']).goal_tolerance;ok,steps=inactive(raw,c['rhr_rule'],tolerance);proof=dict(rollout_uid=c['rollout_uid'],source_raw=str(path),source_raw_sha256=sha(path),source_trace_sha256=raw['trace_sha256'],trigger_never_active=ok,checked_pre_action_steps=steps,tolerance=tolerance)
  if ok:c['exact_inactive_source']=proof
  proofs.append(proof)
 write(HERE/'D_INACTIVE_REUSE_DESIGN.json',dict(utc=utc(),records=len(rows),exact_reusable=sum(x['trigger_never_active']for x in proofs),proofs=proofs,new_evaluation_physics_started=False));seal('D_INACTIVE_REUSE_DESIGN.receipt.json',[HERE/'D_INACTIVE_REUSE_DESIGN.json',HERE/'rhr_reuse.py'],before_evaluation_submission=True)

def execute(c,p,cache,phase):
 if not c.get('exact_inactive_source'):return original_execute(c,p,cache,phase)
 ref=c['exact_inactive_source'];old=read(ref['source_raw']);assert sha(ref['source_raw'])==ref['source_raw_sha256'];ok,steps=inactive(old,c['rhr_rule'],ref['tolerance']);assert ok and steps==ref['checked_pre_action_steps'];z=np.load(old['trace_path']);values={k:z[k]for k in z.files};T=len(z['positions'])-1;values['rhr_active']=np.zeros((T,4),bool);dest=HERE/phase/'traces'/f"{c['rollout_uid']}.npz";dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**values);r=dict(old);s=next(s for s in p['states']if s['state_uid']==c['state_uid']);r.update(experiment_uid=EXP,scenario_uid=s['scenario_uid'],case=c,**{k:c[k]for k in ['state_uid','eta_uid','controller_uid','seed_key','rollout_uid']},trace_path=str(dest),trace_sha256=sha(dest),new_physical_steps=0,physical_steps=T,exact_inactive_reuse=ref,execution_code_sha256=sha(__file__),jax_enable_x64=False);return clean(r)
