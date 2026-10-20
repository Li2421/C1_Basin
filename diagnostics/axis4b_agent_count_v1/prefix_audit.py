"""Verify exact unreached-switch evidence derivations independently of DB matching."""
from common import *
from exact_reuse import compatible

def main():
 counts={};saved=0;unknown=0
 for phase in ['source','target']:
  if not (HERE/f'{phase}_CASES.json').exists():continue
  p=read(HERE/f'{phase}_protocol.json');count=0
  for c in read(HERE/f'{phase}_CASES.json'):
   q=read(HERE/phase/'raw'/f"{c['rollout_uid']}.json");ref=q.get('prefix_source')
   if ref is None:continue
   d=c['shared_prefix_source_case'];path=HERE/d['phase']/'raw'/f"{d['rollout_uid']}.json";old=read(path)
   assert Path(ref['raw_path']).resolve()==path.resolve() and sha(path)==ref['raw_sha256']
   assert compatible(old,c) and old['rollout_uid']==ref['source_rollout_uid']==d['rollout_uid']
   assert old['controller_uid']==ref['source_controller_uid'] and old['experiment_uid']==ref['source_experiment_uid']
   assert old['case']['arm']==ref['source_arm'] and old['outcome']==ref['source_outcome']
   assert c['switch_step']==ref['unreached_switch_step']
   v=p['profiles'][c['profile_key']]['payload'];ov=p['profiles'][old['case']['profile_key']]['payload']
   for key in ['N','controller_class','flow_checkpoint_sha256','evaluation_config','orthoflow3_sha256','basis_implementation_sha256','safety_projection_sha256','safety_wrapper_sha256','rng','execution_source_sha256','jax_enable_x64']:assert v[key]==ov[key],key
   for key in ['trace_path','trace_sha256','outcome','success','collision','timeout','deadlock','numerical_failure','error','first_action','physical_steps','episode_length','actual_config','final_goal_fraction','minimum_swept_clearance','execution_code_sha256','future_root','evidence_grade']:assert q[key]==old[key],key
   assert q['trace_path']==ref['trace_path'] and sha(q['trace_path'])==q['trace_sha256']==ref['trace_sha256']
   assert q['new_physical_steps']==0 and q['deterministic_reused_steps']==old['physical_steps'] and q['deterministic_source'] is None
   assert q['label_derivation_code_sha256']==sha(HERE/'exact_reuse.py') and q['execution_kind']=='EXACT_PRE_SWITCH_EPISODE_DERIVATION'
   count+=1;saved+=q['physical_steps'];unknown+=q['numerical_failure']
  counts[phase]=count
 write(HERE/'EXACT_PREFIX_REUSE_AUDIT.json',dict(utc=utc(),passed=True,derived_episodes=counts,avoided_duplicate_steps=saved,preserved_unknown_episodes=unknown,numerical_runtime_unchanged=True,scope='Whole episode only; no suffix resume; unknown strictly before switch. P2 future1 copies are checked separately by original review.'))
if __name__=='__main__':main()
