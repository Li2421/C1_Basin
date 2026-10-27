"""Posthoc taxonomy, gain attribution and hover diagnostics for revised stage two."""
from experiment import *
from atomic_write import write
from reclassify_instances_20261021 import GRID,LABELS,small_diagnosis
from analysis_standard_supplement_20261022 import classify as category
from statistics import summarize,FIXED
from derived_metrics import weighted_rates
from result_inventory import qualification
import numpy as np,collections

def report(results,dynamics,sources):
 byuid={s['rollout_uid']:s for s in sources};groups={};hover_cache={};allrows=[]
 for row in dynamics:
  stage_result=results.get('optional_dynamics_'+row['scene'],{})
  if stage_result.get('status')!='complete' or stage_result.get('diagnosis_status')!='complete':continue
  raw=read(byuid[row['rollout_uid']]['raw']);c=raw['case'];tol=c['scenario_payload']['config']['goal_tolerance'];dist=np.asarray(raw['final_goal_distances']);missing=dist>tol
  hovering=int(bool(raw['timeout']) and bool(missing.any()) and bool(np.all(dist[missing]<=3*tol)))
  allrows.append(dict(row,near_goal_wandering=hovering));hover_cache[row['rollout_uid']]=hovering
  if row['method']!='safety':continue
  key=(row['scene'],row['controller'])
  if key not in groups:groups[key]=dict(scene=row['scene'],layer='paper_dynamics',controller=row['controller'],type=row['type'],episodes=0,successes=0,unknowns=0,counts={g:collections.Counter() for g in GRID},at_least_one_4a_prime=False)
  g=groups[key];g['episodes']+=1;g['successes']+=row['success'];g['unknowns']+=row['unknown'];g['at_least_one_4a_prime']|=bool(row['deadlock_episode'])
  dp=byuid[row['rollout_uid']]['diagnostic']
  if not dp:raise RuntimeError('Stage-two baseline diagnostic missing')
  diag=small_diagnosis(Path(dp).read_bytes())
  if not raw['success']:
   grids={(x['window_seconds'],x['cone_tolerance']):x['local'] for x in diag['grid']}
   if not grids and row['unknown']:grids={g:['unknown']*row['N'] for g in GRID}
   assert set(grids)==set(GRID)
   for point,cats in grids.items():g['counts'][point].update(cat for cat,failed in zip(cats,missing) if failed)
 instances=[]
 for key,g in sorted(groups.items()):
  counts=g.pop('counts');lo=g['successes']/g['episodes'];hi=(g['successes']+g['unknowns'])/g['episodes'];primary=category(True,hi,counts[(5.,.05)])
  sensitivity=[dict(window_seconds=W,cone_tolerance=tau,classification=category(True,hi,counts[(W,tau)]),unarrived_agent_episode_categories=dict(counts[(W,tau)])) for W,tau in GRID]
  instances.append(dict(**g,qualified=True,success_lower=lo,success_upper=hi,classification=primary,remove_in_revised_exclusion=primary==LABELS[2],robot_deadlock_evidence_eligible=primary!=LABELS[2],state_related_failure_definite=lo>.05 and hi<.95,primary_unarrived_agent_episode_categories=dict(counts[(5.,.05)]),sensitivity=sensitivity,sensitivity_changes=any(x['classification']!=primary for x in sensitivity),other_layer_success='N/A: no centralized stage-two execution requested'))
 discrimination=[];splits={};hovers={}
 for scene in NAMES:
  rr=[r for r in instances if r['scene']==scene];rows=[r for r in allrows if r['scene']==scene]
  failures={r['controller'] for r in rr if r['remove_in_revised_exclusion']};globals={r['controller'] for r in rr if r['classification']==LABELS[0]}
  discrimination.append(dict(scene=scene,layer='paper_dynamics',qualified_instances=len(rr),controller_level_failure_instances=len(failures),state_related_failure_instances=sum(r['state_related_failure_definite'] for r in rr),safety_layer_global_failure_instances=len(globals),wall_trap_instances=sum(r['classification']==LABELS[1] for r in rr)))
  split={}
  for label,predicate in [('all_qualified',lambda c:True),('controller_failure',lambda c:c in failures),('remaining_instances',lambda c:c not in failures),('safety_layer_global_failure',lambda c:c in globals)]:
   subset=[r for r in rows if predicate(r['controller'])]
   split[label]=dict(fixed_full=summarize(subset,[('paper_dynamics',m) for m in FIXED],[scene]),gain_interpretation='修正推动controller自身停住的实例；不计为解开死锁' if label=='controller_failure' else '有限窗口诊断候选；timeout不是死锁')
  splits[scene]=split;hovers[scene]=weighted_rates(rows,('near_goal_wandering',))
 write(HERE/'INSTANCE_CLASSES_DYNAMICS_SUPPLEMENT_20261022.json',dict(utc=utc(),instances=instances,discrimination=discrimination,sensitivity_changes=[r for r in instances if r['sensitivity_changes']],primary=dict(window_seconds=5.,cone_tolerance=.05),new_rollouts=0,centralized_stage_two='N/A: not requested',original_stage_two_nonzero='N/A: no original nonzero stage-two executions; eta0 unchanged'))
 for scene in NAMES:
  stage='optional_dynamics_'+scene
  if stage in results:results[stage].update(instance_class_gain_splits=splits[scene],near_goal_wandering=hovers[scene],semantics='修订修正语义（事后开发证据）；eta0沿用原执行语义')
 # Remaining optional variants use the same definition; nominal background has no safety correction.
 from optional_results import collect
 for stage,entry in results.items():
  if stage.startswith(('optional_full_circle_','optional_local_progress_')):
   rows,ss,_=collect(stage);source={s['rollout_uid']:s for s in ss};augmented=[]
   for row in rows:
    raw=read(source[row['rollout_uid']]['raw']);tol=raw['case']['scenario_payload']['config']['goal_tolerance'];dist=np.asarray(raw['final_goal_distances']);missing=dist>tol
    if row['controller'] not in {r['controller'] for r in qualification(row['scene']) if r['qualified']}:continue
    augmented.append(dict(row,near_goal_wandering=int(bool(raw['timeout']) and bool(missing.any()) and bool(np.all(dist[missing]<=3*tol)))))
   entry['near_goal_wandering']=weighted_rates(augmented,('near_goal_wandering',))
 return dict(stage_two_instance_classes='INSTANCE_CLASSES_DYNAMICS_SUPPLEMENT_20261022.json',revised_nonzero_execution=True,original_second_stage_nonzero='N/A: not run',centralized_second_stage='N/A: not requested')
