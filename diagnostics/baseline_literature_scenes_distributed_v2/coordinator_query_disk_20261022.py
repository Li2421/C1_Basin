"""User-ordered corrected stage sequence, beginning with all distributed fixed arms."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from resources_corrections_release_20261023 import snapshot
import fine_dispatch_query_disk_20261022 as fine
from coordinator_flat_static import stage_ready,stage_done,jobs_by_stage
import time,subprocess,fcntl

def action(label,*args):
 path=HERE/'coordinator_actions'/f'correction_{label}_{time.time_ns()}.log';path.parent.mkdir(exist_ok=True)
 with path.open('w') as out:result=subprocess.run([sys.executable,'-B',str(HERE/args[0]),*args[1:]],cwd=ROOT,stdout=out,stderr=subprocess.STDOUT)
 if result.returncode:
  write(HERE/'COORDINATOR_ACTION_ERROR.json',dict(utc=utc(),label=label,arguments=args,log=str(path),returncode=result.returncode,corrected_pipeline=True));raise RuntimeError(str(path))

def status(phase,**extra):
 old=read(HERE/'STATUS.json');old.update(utc=utc(),phase=phase,resources=snapshot(),correction=extra,correction_prompt='docs/prompt_baseline_literature_scenes_distributed_v2_corrections_20261021.md');write(HERE/'STATUS.json',old)
 write(HERE/'CORRECTION_SCHEDULER_DUTY_20261021.json',dict(utc=utc(),phase=phase,resources=old['resources'],**extra))
 return old

def prepare_fixed(layer,scene):
 stage=f'{layer}_revised_fixed_{scene}'
 if not (HERE/f'{stage}_PREFLIGHT.receipt.json').exists():action('fixed_plan_'+stage,'revised_case_factory_20261021.py',layer,scene)
 if not stage_ready(stage):action('fixed_queue_'+stage,'work_queue.py',stage)
 fine.prepare(stage);return stage

def run_stages(stages,phase):
 diagnostics={}
 for stage in stages:fine.prepare(stage)
 while True:
  state=snapshot();mapping=jobs_by_stage(state)
  for stage in stages:
   fine.recover(stage,set(mapping));fine.sync_original(stage)
   if (HERE/stage/'NEEDS_IMPLEMENTATION_REVIEW.json').exists():raise RuntimeError(('Review revised worker',stage))
  pending=[s for s in stages if not stage_done(s)]
  if not pending and not any(s in stages for s in mapping.values()):return
  available={s:fine.available(s) for s in pending};counts={s:sum(v==s for v in mapping.values()) for s in pending}
  if any(available.values()):
   for job,other_stage in mapping.items():
    if other_stage.startswith('analysis_'):subprocess.run(['scancel','--batch','--signal=USR1',job],capture_output=True,text=True)
  for _ in range(min(state['free_cpu_shares'],sum(available.values()))):
   target=min((s for s in pending if available[s]),key=lambda s:counts[s]);fine.submit(target);counts[target]+=1;available[target]-=1
  # Only fill fixed tails when there is no unclaimed research request.
  current=snapshot();live=jobs_by_stage(current)
  if current['free_cpu_shares'] and not any(fine.available(stage) for stage in pending):
   from analysis_jobs_current import prepare as prepare_analysis
   from diagnosis_backfill_release_20261023 import submit as submit_backfill
   from infrastructure_recovery_correction_20261021 import recover
   completed=[f'{layer}_revised_fixed_{scene}' for layer in ('distributed','central') for scene in NAMES if stage_done(f'{layer}_revised_fixed_{scene}')]
   for original in completed:
    if original not in diagnostics:diagnostics[original]=prepare_analysis(original)
   choices=list(diagnostics.values())
   for stage in choices:recover(stage,set(live))
   remaining={stage:sum(not (HERE/stage/'claims'/row['group']).exists() for row in read(HERE/stage/'GROUP_INDEX.json')) for stage in choices}
   analysis_counts={stage:sum(value==stage for value in live.values()) for stage in choices}
   for _ in range(min(current['free_cpu_shares'],sum(remaining.values()))):
    target=min((stage for stage in choices if remaining[stage]),key=lambda stage:analysis_counts[stage]);submit_backfill(target);remaining[target]-=1;analysis_counts[target]+=1
  status(phase,pending=pending,stage_worker_counts=counts,old_circle_workers_draining=sum(s=='learning_labels_lit_circle_five' for s in mapping.values()),next_sequence='D learned narrow/crowd/door -> C fixed4 -> circle revised learning -> optional -> report/audit',no_old_circle_new_submissions=True)
  time.sleep(10)

def main():
 verify_receipt('CBS_QUERY_DISK_EQUIVALENCE_20261022.receipt.json');verify_receipt('CBS_QUERY_DISK_PIPELINE_CODE_20261022.receipt.json')
 verify_receipt('CBS_START_SIGNATURE_EQUIVALENCE_20261022.receipt.json');verify_receipt('CBS_START_SIGNATURE_PIPELINE_CODE_20261022.receipt.json')
 verify_receipt('CORRECTION_CONFIGURED_CONTINUATION_READY_20261023.receipt.json')
 verify_receipt('CPU_RELEASE_CODE_20261023.receipt.json')
 verify_receipt('CORRECTION_RECOVERY_ADAPTER_20261021.receipt.json')
 with (HERE/'CORRECTION_COORDINATOR.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  verify_receipt('CORRECTION_FIXED_PIPELINE_IMPLEMENTATION_20261021.receipt.json');verify_receipt('CORRECTION_REVISION.receipt.json');verify_receipt('PREREGISTRATION_ADDENDUM_20261021.receipt.json')
  write(HERE/'COORDINATOR_PID.json',dict(pid=os.getpid(),utc=utc(),entrypoint='coordinator_query_disk_20261022.py'))
  status('revised_distributed_fixed_preflight')
  action('merge_landed_old_semantics','db_merge.py')
  stages=[prepare_fixed('distributed',scene) for scene in NAMES]
  run_stages(stages,'revised_distributed_fixed')
  action('merge_revised_distributed_fixed','db_merge.py')
  if not (HERE/'REVISED_DISTRIBUTED_FIXED_COMPLETE_20261021.receipt.json').exists():seal('REVISED_DISTRIBUTED_FIXED_COMPLETE_20261021.receipt.json',[HERE/(s+'/GROUP_INDEX.json') for s in stages],all_original_groups_complete=True,stages=stages,logical_requests=68064)
  while True:
   entry=read(HERE/'ACTIVE_ENTRYPOINT.json');proof=entry.get('downstream_receipt');target=entry['downstream']
   assert Path(target).name==target and target.endswith('.py'),target
   if proof and (HERE/proof).exists():
    assert Path(proof).name==proof and proof.endswith('.receipt.json'),proof
    ready=verify_receipt(proof);assert str(HERE/target) in ready['files'],target
    break
   status('revised_Dfixed_complete_awaiting_continuation_implementation',required_next='D learning3 -> Cfixed4 -> circle learning last',required_receipt=proof);time.sleep(10)
  action('correction_after_Dfixed',target)
if __name__=='__main__':main()
