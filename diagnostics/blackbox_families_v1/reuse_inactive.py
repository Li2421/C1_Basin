"""Exact reuse only after proving both rule trajectories inactive."""
from common import *
from trigger import Trigger
from types import SimpleNamespace
from functools import lru_cache
@lru_cache(None)
def manifest(phase):return read(HERE/f'{phase}_CASES.json')

def inactive(rule,positions,velocities,goals,tolerance,failed):
 if rule is None:return True
 trigger=Trigger(rule);end=len(positions) if failed else len(positions)-1
 for t in range(end):
  env=SimpleNamespace(step_count=t,positions=positions[t],velocities=velocities[t],goals=goals,config=SimpleNamespace(goal_tolerance=tolerance))
  if trigger.update(env).any():return False
 return True

def try_reuse(c,p,phase):
 if not c.get('rhr_rule'):return None
 key=(c['base_controller_name'],c['state_uid'],c['seed'],c['placement'])
 # Candidate paths come from this frozen manifest only; no unrelated directory scan.
 cases=manifest(phase);matches=[x for x in cases if x['rollout_uid']!=c['rollout_uid'] and (x['base_controller_name'],x['state_uid'],x['seed'],x['placement'])==key and (x.get('rhr_rule') or x['arm']=='safety_baseline')]
 for source in matches:
  path=HERE/phase/'raw'/f"{source['rollout_uid']}.json"
  if not path.exists():continue
  r=read(path)
  # Do not chain provenance; use a directly simulated source.
  if r.get('exact_inactive_reuse'):continue
  with np.load(r['trace_path']) as z:pos=z['positions'];vel=z['velocities'];goals=z['goals']
  tol=p['native_manifest']['scenario_config']['goal_tolerance'];failed=bool(r['numerical_failure'])
  if not inactive(source.get('rhr_rule'),pos,vel,goals,tol,failed):continue
  if not inactive(c['rhr_rule'],pos,vel,goals,tol,failed):continue
  assert sha(r['trace_path'])==r['trace_sha256']
  proof=dict(source_raw=str(path),source_raw_sha256=sha(path),source_trace_sha256=r['trace_sha256'],source_rule=source.get('rhr_rule'),target_rule=c['rhr_rule'],source_and_target_inactive_on_every_action_and_failure_attempt=True,same_controller_state_seed_placement=True,physical_steps_repeated=0)
  r.update(case=c,**{k:c[k] for k in ['rollout_uid','state_uid','eta_uid','controller_uid','seed_key']},new_physical_steps=0,exact_inactive_reuse=proof,execution_code_sha256=sha(__file__))
  return r
 return None
