"""Combined original/revised main evidence; immutable original summaries are reused."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from statistics_bootstrap_memo_reader_20261022 import FIXED,LAYERS,collect,controller_census,summarize,crossfit_oracle
from reclassify_instances_20261021 import small_diagnosis
from result_inventory import find_raw,qualification
from derived_metrics import weighted_rates
import numpy as np,collections,csv,copy

SEMANTICS={'original':'原修正语义（预注册）','revised':'修订修正语义（看过预注册结果后提出的事后修订，开发证据）'}

def revised_records(original):
 rows=[dict(r) for r in original if r['method']=='safety'];sources=[]
 stages=[f'{l}_revised_fixed_{s}' for l in LAYERS for s in NAMES]
 stages += ['learning_eval_revised_20261021_'+s for s in NAMES if (HERE/f'learning_eval_revised_20261021_{s}_CASES.json').exists()]
 stages += ['learning_eval_revised_20261021_trainfixed_'+s for s in NAMES]
 for stage in stages:
  for c in read(HERE/f'{stage}_CASES.json'):
   path=find_raw(c['rollout_uid'],stage,c['state']['scene']);raw=read(path);dpath=HERE/'DIAGNOSTICS'/f"{c['rollout_uid']}.json";diag=small_diagnosis(dpath.read_bytes());ctl=c['controller_payload']['nominal'];s=c['state'];cats=diag['local']
   rows.append(dict(layer=c.get('safety_layer','distributed'),scene=s['scene'],N=s['N'],state=s['index'],controller=ctl['controller'],type=ctl['type_code'],method=c['method'],seed=c['seed'],success=raw['success'],upper=min(1,raw['success']+raw['numerical_failure']),collision=raw['collision'],timeout=raw['timeout'],unknown=raw['numerical_failure'],deadlock_episode=int('robot_filter_equilibrium_deadlock' in cats),wall_episode=int('wall_filter_equilibrium' in cats),deadlock_agents=cats.count('robot_filter_equilibrium_deadlock'),wall_agents=cats.count('wall_filter_equilibrium'),categories=cats,canonical_joint_episode=int('filter_equilibrium_deadlock' in diag.get('canonical_joint',[])),actual_joint_episode=int('filter_equilibrium_deadlock' in diag.get('actual_product_joint',[])),rollout_uid=c['rollout_uid']))
   sources.append(dict(rollout_uid=c['rollout_uid'],logical_stage=stage,raw=str(path),raw_sha256=sha(path),diagnostic=str(dpath),diagnostic_sha256=sha(dpath)))
 return rows,sources

def diagnostic_counts(summary,rows):
 out=copy.deepcopy(summary)
 if out.get('status')!='complete':return out
 ctls=set(out['controllers']);states={tuple(s) for s in out['state_clusters']}
 for key,counter in out['pooled_counts'].items():
  import ast
  arm=ast.literal_eval(key);rr=[r for r in rows if r['controller'] in ctls and (r['scene'],r['N'],r['state']) in states and (r['layer'],r['method'])==arm]
  for k in ('success','collision','timeout','unknown'):assert sum(r[k] for r in rr)==counter[k],(key,k)
  assert len(rr)==counter['episodes']
  for k in ('deadlock_episode','wall_episode','deadlock_agents','wall_agents','canonical_joint_episode','actual_joint_episode'):counter[k]=sum(r[k] for r in rr)
 out['outcome_statistics_exact_reuse']='COMPLETED_SCENE_RESULTS.json; original outcome estimates,5000 bootstrap draws and comparisons unchanged; diagnostic counts added from saved diagnostics'
 return out

def matched(rows,scene,layer='distributed'):
 keys={(r['N'],r['state']) for r in rows if r['scene']==scene and r['layer']==layer and r['method']=='learned_scene'}
 return [r for r in rows if r['scene']==scene and (r['N'],r['state']) in keys]

def class_splits(rows,scene,classes):
 output={}
 for layer in LAYERS:
  group=[r for r in rows if r['scene']==scene and r['layer']==layer]
  failures={r['controller'] for r in classes if r['scene']==scene and r['layer']==layer and r['remove_in_revised_exclusion']}
  globalfails={r['controller'] for r in classes if r['scene']==scene and r['layer']==layer and r['classification']=='安全层导致的全局失败'}
  byscope={}
  for label,predicate in [('all_qualified',lambda c:True),('controller_failure',lambda c:c in failures),('remaining_instances',lambda c:c not in failures),('safety_layer_global_failure',lambda c:c in globalfails)]:
   selected=[r for r in group if predicate(r['controller'])]
   fixed=summarize(selected,[(layer,m) for m in FIXED],[scene])
   subset=matched(selected,scene,layer)
   learned=summarize(subset,[(layer,m) for m in (*FIXED,'learned_scene')],[scene]) if subset else dict(status='N/A',reason='No learned result in this layer/semantics/scope')
   byscope[label]=dict(fixed_full=fixed,learned_evaluation=learned,gain_interpretation='修正推动controller自身停住的实例；不计为解开死锁' if label=='controller_failure' else '有限窗口诊断候选；不能把timeout等同死锁',class_filter_layer=layer)
  output[layer]=byscope
 return output

def wandering_rates(rows,sources):
 source={s['rollout_uid']:s for s in sources};cache={};out=[]
 for row in rows:
  rid=row['rollout_uid']
  if rid not in cache:
   entry=source[rid];raw=read(entry['raw']);cfg=raw['case']['scenario_payload']['config'];dist=np.asarray(raw.get('final_goal_distances',[]),float);tol=cfg['goal_tolerance']
   if len(dist)!=row['N']:
    with np.load(raw['trace_path']) as z:dist=np.linalg.norm(z['positions'][-1]-z['goals'],axis=-1)
   unarrived=dist>tol
   cache[rid]=int(bool(raw['timeout']) and bool(unarrived.any()) and bool(np.all(dist[unarrived]<=3*tol)))
  out.append(dict(row,near_goal_wandering=cache[rid]))
 q={s:{r['controller'] for r in qualification(s) if r['qualified']} for s in NAMES}
 qualified=[r for r in out if r['controller'] in q[r['scene']]]
 return {s:weighted_rates([r for r in qualified if r['scene']==s],('near_goal_wandering',)) for s in NAMES}

def save_or_assert(name,value):
 path=HERE/name
 if path.exists():assert canonical(read(path))==canonical(value),name
 else:write(path,value)

def complete_original_landed_label_diagnoses():
 from correction_after_distributed_fixed_20261021_quota18_now_20261022 import analyze
 originals=[];coverage=[]
 for scene in NAMES:
  original='learning_labels_'+scene;stage='landed_original_labels_20261021_'+scene
  planned=read(HERE/f'{original}_CASES.json');landed=[]
  for c in planned:
   try:find_raw(c['rollout_uid'],original,scene)
   except FileNotFoundError:
    assert scene=='lit_circle_five',c['rollout_uid']
    continue
   landed.append(c)
  save_or_assert(stage+'_CASES.json',landed);originals.append(stage)
  coverage.append(dict(scene=scene,planned=len(planned),landed=len(landed),unwritten_not_resumed=len(planned)-len(landed)))
 analyze(originals)
 write(HERE/'ORIGINAL_LANDED_LABEL_DIAGNOSIS_COVERAGE_20261021.json',dict(utc=utc(),coverage=coverage,new_rollouts=0,unwritten_not_unknown=True,scope='Only exact landed original labels; no old evaluation or recovery of stopped requests'))

def main():
 prereg();verify_receipt('CORRECTION_MAIN_RESULTS_READY_20261021.receipt.json');assert not (HERE/'MIGRATION_NUMBERS.receipt.json').exists()
 for proof in ('COMPLETED_SCENE_NUMBERS.receipt.json','COMPLETED_FIXED_CROSSFIT_NUMBERS.receipt.json','COMPLETED_DIAGNOSIS_NUMBERS.receipt.json','INSTANCE_CLASSES_SUPPLEMENT_20261022.receipt.json'):verify_receipt(proof)
 complete_original_landed_label_diagnoses()
 old,sources=collect();new,new_sources=revised_records(old);census=controller_census(old);classes=read(HERE/'INSTANCE_CLASSES_SUPPLEMENT_20261022.json')['instances']
 assert census==read(HERE/'COMPLETED_DIAGNOSIS_RESULTS.json')['controller_census']
 # Before reusing saved estimates, verify every previously sealed raw hash.
 current={s['rollout_uid']:s['raw_sha256'] for s in sources}
 for s in read(HERE/'COMPLETED_SCENE_RESULT_SOURCES.json'):assert current[s['rollout_uid']]==s['raw_sha256'],s['rollout_uid']
 previous=read(HERE/'COMPLETED_SCENE_RESULTS.json')['results'];crossfit=read(HERE/'COMPLETED_FIXED_CROSSFIT.json')['results'];results={}
 arms=[(l,m) for l in LAYERS for m in FIXED]
 # Native old threshold census is retained for exact historical comparability only.
 for scene in NAMES:
  print('Summarizing',scene,flush=True)
  oldentry=previous[scene]
  fixed=diagnostic_counts(oldentry['fixed_full'],old)
  learned=diagnostic_counts(oldentry['learned_evaluation'],old)
  if scene=='lit_circle_five':learned=dict(status='N/A',reason='用户决定停止：修正语义修订；已落盘旧标签保留，不执行旧语义评估')
  entry=dict(fixed_full=fixed,excluding_controller_failures=summarize(old,arms,[scene],True,census),learned_evaluation=learned,crossfit_fixed_oracle=crossfit[scene],semantics=SEMANTICS['original'],legacy_exclusion_note='Historical threshold-only exclusion; superseded by instance_class_gain_splits')
  entry['revised']=dict(fixed_full=summarize(new,arms,[scene]),learned_evaluation=summarize(matched(new,scene),[('distributed',m) for m in (*FIXED,'learned_scene')],[scene]),semantics=SEMANTICS['revised'],central_scene_learning='N/A: not requested for revised semantics')
  entry['instance_class_gain_splits']={sem:class_splits(rows,scene,classes) for sem,rows in [('original',old),('revised',new)]}
  results[scene]=entry
 results['four_scene_type_equal']=summarize(old,arms,list(NAMES),census=census)
 results['four_scene_excluding_controller_failures']=summarize(old,arms,list(NAMES),True,census)
 results['four_scene_revised_type_equal']=summarize(new,arms,list(NAMES))
 oldlearned=[s for s in NAMES if matched(old,s)]
 oldsub=[r for s in oldlearned for r in matched(old,s)]
 results['supported_scene_distributed_learned_evaluation']=summarize(oldsub,[('distributed',m) for m in (*FIXED,'learned_scene')],oldlearned)
 results['four_scene_revised_distributed_learned_evaluation']=summarize([r for s in NAMES for r in matched(new,s)],[('distributed',m) for m in (*FIXED,'learned_scene')],list(NAMES))
 hover={sem:wandering_rates(rows,sources+new_sources) for sem,rows in [('original',old),('revised',new)]}
 write(HERE/'MIGRATION_RESULTS.json',dict(utc=utc(),evidence='DEVELOPMENT',results=results,semantics=SEMANTICS,near_goal_wandering=hover,instance_class_authority='INSTANCE_CLASSES_SUPPLEMENT_20261022.json',old_circle_label_stop=read(HERE/'OLD_CIRCLE_LEARNING_STOPPED_20261021.json')))
 save_or_assert('CONTROLLER_LEVEL_FAILURES.json',census)
 write(HERE/'MIGRATION_RESULT_SOURCES.json',sources);write(HERE/'REVISED_MIGRATION_RESULT_SOURCES_20261021.json',new_sources)
 write(HERE/'COMPACT_MAIN_RESULTS.json',old);write(HERE/'COMPACT_REVISED_MAIN_RESULTS_20261021.json',new)
 fields=('scene','layer','controller','type','N','state','seed','success','collision','timeout','unknown','deadlock_episode','wall_episode','deadlock_agents','wall_agents','rollout_uid')
 with (HERE/'SAFETY_BASELINE.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in old if r['method']=='safety')
 flat=[]
 for scene in NAMES:
  for sem,entry in [('original',results[scene]),('revised',results[scene]['revised'])]:
   for scope in ('fixed_full','learned_evaluation'):
    flat.extend(dict(scene=scene if scope=='fixed_full' else scene+'_learned_evaluation',semantics=sem,**r) for r in entry[scope].get('statistics',[]) if scope=='fixed_full' or r['method']=='learned_scene')
 fields=('scene','type','layer','method','success_rate','ci95','success_upper','upper_ci95','controllers','semantics')
 with (HERE/'MIGRATION_RESULTS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:canonical(r[k]) if isinstance(r[k],(list,dict)) else r[k] for k in fields} for r in flat)
 names=['MIGRATION_RESULTS.json','MIGRATION_RESULTS.csv','CONTROLLER_LEVEL_FAILURES.json','MIGRATION_RESULT_SOURCES.json','REVISED_MIGRATION_RESULT_SOURCES_20261021.json','COMPACT_MAIN_RESULTS.json','COMPACT_REVISED_MAIN_RESULTS_20261021.json','SAFETY_BASELINE.csv','statistics_correction_revised_20261021_quota18_now_20261022.py','statistics.py','INSTANCE_CLASSES_SUPPLEMENT_20261022.receipt.json']
 seal('MIGRATION_NUMBERS.receipt.json',[HERE/n for n in names],bootstrap_iterations=5000,before_optional_dynamics_and_charts=True,original_outcome_estimates_exact_reuse=True,revised_semantics_posthoc=True,primary_exclusion='Only revised taxonomy controller failure,per layer')
if __name__=='__main__':raise RuntimeError('Import-only helper; use statistics_parallel_correction_20261023_quota18_now_20261022.py')
