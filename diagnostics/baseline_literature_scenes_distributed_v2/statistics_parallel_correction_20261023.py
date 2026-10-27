"""Parallel independent native summaries, with unchanged5000-draw paired inference."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from statistics_correction_header_20261023 import collect,revised_records,controller_census,complete_original_landed_label_diagnoses,save_or_assert,SEMANTICS
from statistics_parallel_tasks_20261023 import task_manifest
from coordinator_correction_release_20261023 import status,jobs_by_stage,stage_done
from resources_corrections_release_20261023 import snapshot,require_capacity
from infrastructure_recovery_correction_20261021 import classify
import subprocess,time,fcntl,csv

INPUT_PROOF='PARALLEL_NUMERIC_INPUTS_20261023.receipt.json'
CODE_PROOF='CORRECTION_PARALLEL_NUMERICS_READY_20261023.receipt.json'

def prepare_inputs():
 if (HERE/INPUT_PROOF).exists():verify_receipt(INPUT_PROOF);return
 prereg()
 for proof in ('REVISED_DISTRIBUTED_FIXED_COMPLETE_20261021.receipt.json','REVISED_CENTRAL_FIXED_COMPLETE_20261021.receipt.json','COMPLETED_SCENE_NUMBERS.receipt.json','COMPLETED_DIAGNOSIS_NUMBERS.receipt.json','COMPLETED_FIXED_CROSSFIT_NUMBERS.receipt.json','INSTANCE_CLASSES_REVISED.receipt.json'):verify_receipt(proof)
 for layer in ('distributed','central'):
  for scene in NAMES:assert stage_done(f'{layer}_revised_fixed_{scene}'),(layer,scene)
 for scene in NAMES:
  stage='learning_eval_revised_20261021_'+scene
  assert stage_done(stage) or (HERE/f'LEARNING_GAP_REVISED_20261021_{scene}.json').exists(),scene
 complete_original_landed_label_diagnoses()
 old,sources=collect();new,new_sources=revised_records(old);census=controller_census(old)
 assert census==read(HERE/'COMPLETED_DIAGNOSIS_RESULTS.json')['controller_census']
 current={s['rollout_uid']:s['raw_sha256'] for s in sources}
 for s in read(HERE/'COMPLETED_SCENE_RESULT_SOURCES.json'):assert current[s['rollout_uid']]==s['raw_sha256'],s['rollout_uid']
 for name,value in [('COMPACT_MAIN_RESULTS.json',old),('COMPACT_REVISED_MAIN_RESULTS_20261021.json',new),('CONTROLLER_LEVEL_FAILURES.json',census),('MIGRATION_RESULT_SOURCES.json',sources),('REVISED_MIGRATION_RESULT_SOURCES_20261021.json',new_sources)]:save_or_assert(name,value)
 seal(INPUT_PROOF,[HERE/n for n in ('COMPACT_MAIN_RESULTS.json','COMPACT_REVISED_MAIN_RESULTS_20261021.json','CONTROLLER_LEVEL_FAILURES.json','MIGRATION_RESULT_SOURCES.json','REVISED_MIGRATION_RESULT_SOURCES_20261021.json','ORIGINAL_LANDED_LABEL_DIAGNOSIS_COVERAGE_20261021.json','INSTANCE_CLASSES_REVISED.receipt.json',CODE_PROOF)],original_raw_hashes_verified=True,original_census_exact=True,immutable_shared_statistical_inputs=True,new_rollouts=0)

def submit(key,task,memory_gb):
 with (HERE/'SUBMISSION.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);before=snapshot()
  if not before['free_cpu_shares']:return None
  job=subprocess.check_output(['sbatch','--parsable',f'--mem={memory_gb}G',f'--export=ALL,BLV2_NUMERIC_TASK={key}',str(HERE/'statistics_parallel_20261023.sbatch')],text=True).strip()
  write(HERE/'submissions'/f'{time.time_ns()}_{job}.json',dict(utc=utc(),job_id=job,stage='analysis_'+key,numeric_task=key,numeric_definition=task,memory_gb=memory_gb,resource_snapshot=before,posthoc_only=True))
  return job

def run_tasks():
 manifest=task_manifest();folder=HERE/'parallel_numeric_fragments_20261023';folder.mkdir(exist_ok=True);reviewed=set();memory={k:4 for k in manifest}
 while True:
  state=snapshot();mapping=jobs_by_stage(state);live={j['job_id'] for j in state['jobs']}
  complete={key for key in manifest if (folder/(key+'.receipt.json')).exists()}
  for path in (HERE/'submissions').glob('*.json'):
   row=read(path);key=row.get('numeric_task');job=row['job_id']
   if key not in manifest or job in live or job in reviewed or key in complete:continue
   review=classify('analysis_'+key,job);reviewed.add(job)
   write(HERE/'parallel_numeric_reviews_20261023'/f'{job}.json',review)
   if not review['infrastructure']:
    write(HERE/'PARALLEL_NUMERIC_REVIEW_NEEDED_20261023.json',dict(utc=utc(),task=key,review=review));raise RuntimeError(('Native statistical task needs review',key,job))
   if review['slurm']['state']=='OUT_OF_MEMORY':memory[key]=max(memory[key],min(96,2*int(row['memory_gb'])))
  pending=[key for key in manifest if key not in complete]
  active={stage.removeprefix('analysis_') for stage in mapping.values() if stage.startswith('analysis_numeric_')}
  if not pending and not active:break
  for key in pending:
   if key in active:continue
   if not snapshot()['free_cpu_shares']:break
   job=submit(key,manifest[key],memory[key])
   if job is None:break
   active.add(key)
  status('revised_main_parallel_statistics',completed_native_tasks=len(complete),total_native_tasks=len(manifest),bootstrap_iterations=5000,science_unchanged=True)
  time.sleep(10)
 for key in manifest:verify_receipt(str((folder/(key+'.receipt.json')).relative_to(HERE)))
 return {key:read(folder/(key+'.json'))['result'] for key in manifest}

def assemble(parts):
 manifest=task_manifest();results={};hover={};bytask={canonical(t):parts[k] for k,t in manifest.items()}
 for scene in NAMES:
  entry=bytask[canonical(dict(kind='scene',scene=scene))]
  entry['instance_class_gain_splits']={sem:{layer:bytask[canonical(dict(kind='split',scene=scene,semantics=sem,layer=layer))] for layer in ('central','distributed')} for sem in ('original','revised')}
  results[scene]=entry
 for kind,key in [('old_fixed_pool','four_scene_type_equal'),('old_excluded_pool','four_scene_excluding_controller_failures'),('new_fixed_pool','four_scene_revised_type_equal'),('old_learned_pool','supported_scene_distributed_learned_evaluation')]:results[key]=bytask[canonical(dict(kind=kind))]
 newpool=bytask[canonical(dict(kind='new_learned_pool'))];newkey='four_scene_revised_distributed_learned_evaluation' if len(newpool.get('scope',[]))==4 else 'supported_scene_revised_distributed_learned_evaluation';results[newkey]=newpool
 for sem,kind in [('original','old_hover'),('revised','new_hover')]:hover[sem]=bytask[canonical(dict(kind=kind))]
 write(HERE/'MIGRATION_RESULTS.json',dict(utc=utc(),evidence='DEVELOPMENT',results=results,semantics=SEMANTICS,near_goal_wandering=hover,instance_class_authority='INSTANCE_CLASSES_REVISED.json',old_circle_label_stop=read(HERE/'OLD_CIRCLE_LEARNING_STOPPED_20261021.json'),computation='Independent native summaries run in parallel; original5000 draws,RNG,weighting and formulas unchanged'))
 old=read(HERE/'COMPACT_MAIN_RESULTS.json');fields=('scene','layer','controller','type','N','state','seed','success','collision','timeout','unknown','deadlock_episode','wall_episode','deadlock_agents','wall_agents','rollout_uid')
 with (HERE/'SAFETY_BASELINE.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in old if r['method']=='safety')
 flat=[]
 for scene in NAMES:
  for sem,entry in [('original',results[scene]),('revised',results[scene]['revised'])]:
   for scope in ('fixed_full','learned_evaluation'):flat.extend(dict(scene=scene if scope=='fixed_full' else scene+'_learned_evaluation',semantics=sem,**r) for r in entry[scope].get('statistics',[]) if scope=='fixed_full' or r['method']=='learned_scene')
 for key,sem in [('four_scene_type_equal','original'),('four_scene_excluding_controller_failures','original_legacy_exclusion'),('four_scene_revised_type_equal','revised'),('supported_scene_distributed_learned_evaluation','original'),(newkey,'revised')]:flat.extend(dict(scene=key,semantics=sem,**r) for r in results[key].get('statistics',[]))
 fields=('scene','type','layer','method','success_rate','ci95','success_upper','upper_ci95','controllers','semantics')
 with (HERE/'MIGRATION_RESULTS.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:canonical(r[k]) if isinstance(r[k],(list,dict)) else r[k] for k in fields} for r in flat)
 names=['MIGRATION_RESULTS.json','MIGRATION_RESULTS.csv','CONTROLLER_LEVEL_FAILURES.json','MIGRATION_RESULT_SOURCES.json','REVISED_MIGRATION_RESULT_SOURCES_20261021.json','COMPACT_MAIN_RESULTS.json','COMPACT_REVISED_MAIN_RESULTS_20261021.json','SAFETY_BASELINE.csv',INPUT_PROOF,CODE_PROOF,'statistics_parallel_correction_20261023.py','statistics_parallel_tasks_20261023.py','statistics.py','INSTANCE_CLASSES_REVISED.receipt.json']
 fragments=[HERE/'parallel_numeric_fragments_20261023'/(key+'.receipt.json') for key in manifest]
 seal('MIGRATION_NUMBERS.receipt.json',[*[HERE/n for n in names],*fragments],bootstrap_iterations=5000,before_optional_dynamics_and_charts=True,original_outcome_estimates_exact_reuse=True,revised_semantics_posthoc=True,primary_exclusion='Only revised taxonomy controller failure,per layer',independent_native_calls_parallelized=True,mathematical_kernel_unchanged=True)

def main():
 with (HERE/'PARALLEL_NUMERIC_DRIVER_20261023.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_receipt(CODE_PROOF)
  if (HERE/'MIGRATION_NUMBERS.receipt.json').exists():verify_receipt('MIGRATION_NUMBERS.receipt.json');return
  status('revised_main_statistical_input_preparation');prepare_inputs();parts=run_tasks();assemble(parts)
if __name__=='__main__':main()
