"""Prepare required saved-trace queues during active simulations;submit nothing."""
from experiment import *
from registry import verify_receipt
from coordinator_correction_release_20261023_quota18_now_20261022 import stage_done
from analysis_jobs_current import prepare
import optional_fine_dispatch_spatial_mask_20261022 as fine
import fcntl


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    assert read(HERE/'STATUS.json')['phase']=='paper_dynamics_optional'
    originals=['optional_dynamics_'+s for s in NAMES]
    completed=[s for s in originals if stage_done(s)]
    pending=[s for s in originals if s not in completed]
    available=sum(fine.available(s) for s in pending)
    # The sole coordinator can independently prepare diagnoses on actual spare
    # cores. Overlap only while its primary queue still has ample unclaimed work.
    assert available>17,('Already at primary tail;let sole coordinator prepare',available)
    prefix='OPTIONAL_COMPLETED_DIAGNOSIS_QUEUES_20261022'
    assert not (HERE/(prefix+'.receipt.json')).exists()
    verify_receipt('DYNAMICS_DIAGNOSTICS.receipt.json')
    sources={s:sha(HERE/s/'WORK_QUEUE.receipt.json') for s in completed}
    write(HERE/(prefix+'_HISTORY.json'),dict(utc=utc(),completed_stages=completed,
          source_queue_receipts=sources,source_of_diagnosis='Existing frozen dynamics_diagnostics and optional_analysis_backfill on immutable saved trajectories',
          evidence_gap='Completed raw outcomes need the originally required saved-trace diagnostics;this helper prepares their original queues without recomputing a diagnosis or simulation.',
          new_rollouts=0,new_training=0,new_diagnoses=0,submissions=0))
    outputs=[]
    with (HERE/'SUBMISSION.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for original in completed:
            stage=prepare(original)
            verify_receipt(stage+'/WORK_QUEUE.receipt.json')
            outputs.append(dict(original=original,analysis=stage,groups=len(read(HERE/stage/'GROUP_INDEX.json'))))
            print('PREPARED',stage,flush=True)
    write(HERE/(prefix+'.json'),dict(utc=utc(),queues=outputs,new_rollouts=0,new_training=0,new_diagnoses=0,submissions=0,sole_coordinator_unchanged=True,original_analysis_worker_and_semantics_unchanged=True))
    seal(prefix+'.receipt.json',[Path(__file__),HERE/(prefix+'_HISTORY.json'),HERE/(prefix+'.json'),*[HERE/r['analysis']/'WORK_QUEUE.receipt.json' for r in outputs]],submissions=0,new_rollouts=0,new_diagnoses=0)
    with (HERE/'ITERATION_LOG.md').open('a') as f:f.write('\n'+utc()+'：管理核在原论文动力学请求仍充足且17计算worker运行期间，提前用原analysis_jobs_current.prepare封存已完成场景的保存轨迹诊断队列。诊断计算与派发仍由原sole coordinator的backfill完成；本helper无rollout、无诊断重算、无作业提交、无科学worker信号。见'+prefix+'.receipt.json。\n')
    print('PASS completed-scene diagnosis queues ready;no jobs submitted',flush=True)


if __name__=='__main__':main()
