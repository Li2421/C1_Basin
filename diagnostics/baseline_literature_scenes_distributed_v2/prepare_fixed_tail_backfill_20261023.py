"""Use otherwise idle fixed-stage tail slots for required saved-trace diagnoses."""
from experiment import *
from atomic_write import write
import ast,difflib
os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
files=[Path(__file__)];sources=[]
def output(old,new,after):
 before=(HERE/old).read_text();ast.parse(after);target=HERE/new;assert not target.exists();target.write_text(after);diff=HERE/(new+'.diff');diff.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=old,tofile=new)))
 files.extend([target,diff]);sources.append(dict(source=old,source_sha256=sha(HERE/old),target=new,target_sha256=sha(target)))
old='coordinator_correction_release_20261023.py';s=(HERE/old).read_text()
s=s.replace("def run_stages(stages,phase):\n", "def run_stages(stages,phase):\n diagnostics={}\n")
a="  for _ in range(min(state['free_cpu_shares'],sum(available.values()))):"
b="""  if any(available.values()):
   for job,other_stage in mapping.items():
    if other_stage.startswith('analysis_'):subprocess.run(['scancel','--batch','--signal=USR1',job],capture_output=True,text=True)
  for _ in range(min(state['free_cpu_shares'],sum(available.values()))):"""
assert a in s;s=s.replace(a,b)
a="  status(phase,pending=pending,stage_worker_counts=counts,"
b="""  # Only fill fixed tails when there is no unclaimed research request.
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
  status(phase,pending=pending,stage_worker_counts=counts,"""
assert a in s;s=s.replace(a,b)
s=s.replace("def main():\n", "def main():\n verify_receipt('CORRECTION_CONTINUATION_TAIL_BACKFILL_READY_20261023.receipt.json')\n")
s=s.replace('coordinator_correction_release_20261023.py','coordinator_correction_tail_backfill_20261023.py')
s=s.replace('CORRECTION_CONTINUATION_READY_20261021.receipt.json','CORRECTION_CONTINUATION_TAIL_BACKFILL_READY_20261023.receipt.json')
s=s.replace('correction_after_distributed_fixed_20261021.py','correction_after_distributed_fixed_tail_backfill_20261023.py')
output(old,'coordinator_correction_tail_backfill_20261023.py',s)
old='correction_after_distributed_fixed_20261021.py';s=(HERE/old).read_text().replace('from coordinator_correction_release_20261023 import','from coordinator_correction_tail_backfill_20261023 import')
s=s.replace("def main():\n", "def main():\n verify_receipt('CORRECTION_CONTINUATION_TAIL_BACKFILL_READY_20261023.receipt.json')\n")
output(old,'correction_after_distributed_fixed_tail_backfill_20261023.py',s)
old='scheduler_watchdog_correction_release_20261023.py';s=(HERE/old).read_text().replace('coordinator_correction_release_20261023.py','coordinator_correction_tail_backfill_20261023.py').replace('correction_after_distributed_fixed_20261021.py','correction_after_distributed_fixed_tail_backfill_20261023.py')
a="verify_receipt('CPU_RELEASE_CODE_20261023.receipt.json')";assert a in s;s=s.replace(a,a+";verify_receipt('CORRECTION_CONTINUATION_TAIL_BACKFILL_READY_20261023.receipt.json')")
s=s.replace('scheduler_watchdog_correction_release_20261023.py','scheduler_watchdog_correction_tail_backfill_20261023.py')
output(old,'scheduler_watchdog_correction_tail_backfill_20261023.py',s)
write(HERE/'FIXED_TAIL_BACKFILL_SOURCE_DERIVATIONS_20261023.json',dict(utc=utc(),sources=sources,change='Scheduling only: completed fixed trace diagnosis fills free CPU only when no unclaimed main request; cooperative preemption when main requests appear. Same worker,rollout cases,projection,basis,physics,RNG,fit procedure and phase order.',scientific_jobs_submitted=0))
files += [HERE/'FIXED_TAIL_BACKFILL_SOURCE_DERIVATIONS_20261023.json',HERE/'CORRECTION_CONTINUATION_READY_20261021.receipt.json',HERE/'CPU_RELEASE_CODE_20261023.receipt.json']
seal('CORRECTION_CONTINUATION_TAIL_BACKFILL_READY_20261023.receipt.json',files,scheduling_only=True,main_rollout_priority_preserved=True,scientific_worker_unchanged=True,strict_phase_order_preserved=True)
print('Fixed-tail saved-evidence backfill sealed',flush=True)
