"""One-shot saved-evidence backfill before continuation starts; no scientific producer."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from closed_old_circle_scope_20261023 import preserved_scope,OLD_STAGE
from statistics_correction_revised_20261021 import save_or_assert
from analysis_jobs_current import prepare
from coordinator_correction_configured_20261023 import stage_done,jobs_by_stage
from resources_corrections_release_20261023 import snapshot
from diagnosis_backfill_release_20261023 import submit
import fine_dispatch_correction_release_20261023 as fine
import fcntl

def main():
 os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
 with (HERE/'OPPORTUNISTIC_OLD_LABEL_DIAGNOSES_20261023.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify_receipt('OPPORTUNISTIC_OLD_LABEL_DIAGNOSES_CODE_20261023.receipt.json')
  closed=preserved_scope();stages=[]
  for scene in NAMES:
   original='learning_labels_'+scene;name='landed_original_labels_20261021_'+scene;cases=read(HERE/f'{original}_CASES.json')
   if original==OLD_STAGE:cases=[c for c in cases if c['rollout_uid'] in closed];assert len(cases)==2044
   save_or_assert(name+'_CASES.json',cases);stages.append(prepare(name))
  submitted=[];reason='No safe opportunistic slot'
  # Root currently has no unclaimed research requests and all its ready fixed
  # diagnosis queues are complete. Reserve2 slots during this one-shot handoff.
  state=snapshot();budget=max(0,state['free_cpu_shares']-2);counts={s:0 for s in stages}
  for _ in range(budget):
   current=snapshot();phase=read(HERE/'STATUS.json')['phase']
   if (HERE/'CORRECTION_CONTINUATION_PID_20261021.json').exists() or phase!='revised_distributed_fixed':reason='Required continuation started; leave all submission priority to it';break
   circle='distributed_revised_fixed_lit_circle_five'
   if fine.available(circle):reason='Unclaimed required rollout appeared';break
   ready_fixed=[f'analysis_distributed_revised_fixed_{s}' for s in NAMES if s!='lit_circle_five']
   if not all(stage_done(s) for s in ready_fixed):reason='Parent still has ready fixed diagnoses to submit';break
   if current['free_cpu_shares']<=2:reason='Reserved transition capacity';break
   mapping=jobs_by_stage(current)
   available={s:sum(not (HERE/s/'claims'/r['group']).exists() for r in read(HERE/s/'GROUP_INDEX.json')) for s in stages}
   if not any(available.values()):reason='All prepared saved-label diagnosis groups are claimed';break
   target=min((s for s in stages if available[s]),key=lambda s:counts[s]+sum(v==s for v in mapping.values()));job=submit(target);submitted.append(dict(job=job,stage=target));counts[target]+=1;reason='Preemptible saved-label diagnosis running on otherwise idle tail CPUs'
  write(HERE/'OPPORTUNISTIC_OLD_LABEL_DIAGNOSES_20261023.json',dict(utc=utc(),submitted=submitted,reason=reason,stages=stages,closed_old_circle=2044,new_rollouts=0,new_training=0,scientific_workers_interrupted=0,single_shot_not_a_concurrent_research_producer=True))
  print('Old saved-label diagnosis backfill',len(submitted),reason,flush=True)
if __name__=='__main__':main()
