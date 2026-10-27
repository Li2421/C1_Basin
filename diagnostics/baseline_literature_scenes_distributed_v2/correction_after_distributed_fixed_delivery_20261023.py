"""Strict user-ordered correction continuation; all scheduling uses the dated quota."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from coordinator_correction_configured_20261023 import action,status,stage_ready,stage_done,jobs_by_stage,prepare_fixed,run_stages
from resources_corrections_release_20261023 import snapshot
from infrastructure_recovery_correction_20261021 import recover
import fine_dispatch_correction_release_20261023 as fine
import subprocess,time,fcntl

def wait_ready(name,phase):
 while not (HERE/name).exists():status(phase,required_implementation_receipt=name);time.sleep(10)
 verify_receipt(name)

def yield_analyses():
 while True:
  state=snapshot();mapping=jobs_by_stage(state);jobs={j['job_id']:j for j in state['jobs']}
  targets=[j for j,s in mapping.items() if s.startswith('analysis_')]
  if not targets:return
  for job in targets:
   args=['scancel',job] if jobs[job]['state']=='PENDING' else ['scancel','--batch','--signal=USR1',job]
   subprocess.run(args,capture_output=True,text=True,check=True)
  status('saved_analysis_yielding_to_next_required_phase',jobs=targets);time.sleep(10)

def learning(scenes):
 from scene_learning_scheduler_revised_release_20261023 import run
 verify_receipt('REVISED_LEARNING_IMPLEMENTATION_20261021.receipt.json');verify_receipt('CPU_RELEASE_CODE_20261023.receipt.json')
 for scene in scenes:
  stage='learning_labels_revised_20261021_'+scene
  if not (HERE/f'{stage}_PREFLIGHT.receipt.json').exists():action('revised_labels_plan_'+scene,'learning_design_revised_20261021.py','plan',scene)
  if not stage_ready(stage):action('revised_labels_queue_'+scene,'work_queue.py',stage)
  fine.prepare(stage)
 return run(scenes)

def analyze(originals):
 from analysis_jobs_current import prepare
 from analysis_submit_release_20261023 import submit
 stages=[prepare(s) for s in originals]
 while True:
  state=snapshot();mapping=jobs_by_stage(state)
  for s in stages:
   recover(s,set(mapping))
   if (HERE/s/'NEEDS_IMPLEMENTATION_REVIEW.json').exists():raise RuntimeError(('Required saved diagnosis needs review',s))
  pending=[s for s in stages if not stage_done(s)]
  if not pending and not any(s in stages for s in mapping.values()):return
  available={s:sum(not (HERE/s/'claims'/r['group']).exists() for r in read(HERE/s/'GROUP_INDEX.json')) for s in pending}
  counts={s:sum(v==s for v in mapping.values()) for s in pending}
  for _ in range(min(state['free_cpu_shares'],sum(available.values()))):
   target=min((s for s in pending if available[s]),key=lambda s:counts[s]);submit(target);counts[target]+=1;available[target]-=1
  status('revised_main_saved_trace_diagnosis',pending=pending,stage_worker_counts=counts);time.sleep(10)

def main():
 verify_receipt('CORRECTION_FINAL_DELIVERY_READY_20261023.receipt.json')
 verify_receipt('CORRECTION_CONFIGURED_CONTINUATION_READY_20261023.receipt.json')
 with (HERE/'CORRECTION_CONTINUATION.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for proof in ('CORRECTION_CONTINUATION_READY_20261021.receipt.json','REVISED_DISTRIBUTED_FIXED_COMPLETE_20261021.receipt.json','CORRECTION_REVISION.receipt.json','REVISED_LEARNING_IMPLEMENTATION_20261021.receipt.json','CPU_RELEASE_CODE_20261023.receipt.json'):verify_receipt(proof)
  write(HERE/'CORRECTION_CONTINUATION_PID_20261021.json',dict(utc=utc(),pid=os.getpid()))
  action('merge_before_revised_learning','db_merge.py')
  evaluations=learning(['lit_narrow_passage','lit_crowded_random','lit_doorway'])
  yield_analyses()
  central=[prepare_fixed('central',scene) for scene in NAMES]
  run_stages(central,'revised_central_fixed')
  action('merge_before_revised_circle_learning','db_merge.py')
  if not (HERE/'REVISED_CENTRAL_FIXED_COMPLETE_20261021.receipt.json').exists():seal('REVISED_CENTRAL_FIXED_COMPLETE_20261021.receipt.json',[HERE/s/'GROUP_INDEX.json' for s in central],stages=central,logical_requests=68064)
  evaluations+=learning(['lit_circle_five'])
  yield_analyses()
  fixed=[f'{layer}_revised_fixed_{s}' for layer in ('distributed','central') for s in NAMES]
  analyze(fixed+evaluations)
  action('merge_revised_main','db_merge.py')
  wait_ready('CORRECTION_PARALLEL_NUMERICS_READY_20261023.receipt.json','revised_main_complete_waiting_combined_statistics_implementation')
  if not (HERE/'MIGRATION_NUMBERS.receipt.json').exists():action('combined_original_revised_numbers','statistics_parallel_correction_20261023.py')
  wait_ready('CORRECTION_FINAL_DELIVERY_READY_20261023.receipt.json','combined_main_numbers_sealed_waiting_revised_optional_implementation')
  action('revised_optional_and_final_delivery','optional_delivery_correction_delivery_20261023.py')
if __name__=='__main__':main()
