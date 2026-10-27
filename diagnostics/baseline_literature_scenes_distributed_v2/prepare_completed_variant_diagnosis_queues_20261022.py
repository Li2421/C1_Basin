"""Prepare original required posthoc queues; never submit or run diagnoses."""
from experiment import *
from registry import verify_receipt
from coordinator_correction_release_20261023_quota18_now_20261022 import stage_done
from analysis_jobs_current import prepare
import optional_fine_dispatch_spatial_mask_20261022 as fine
import fcntl


def main():
    os.sched_setaffinity(0, {read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    assert read(HERE/'STATUS.json')['phase'] == 'optional_local_progress'
    active = ['optional_local_progress_'+s for s in NAMES]
    available = sum(fine.available(s) for s in active if not stage_done(s))
    assert available > 17, ('Leave final-tail preparation to sole coordinator', available)
    completed = ['optional_full_circle_lit_circle_five',
                 'optional_local_progress_lit_narrow_passage',
                 'optional_local_progress_lit_doorway']
    assert all(stage_done(s) for s in completed)
    prefix = 'OPTIONAL_COMPLETED_VARIANT_DIAGNOSIS_QUEUES_20261022'
    assert not (HERE/(prefix+'.receipt.json')).exists()
    verify_receipt('OPTIONAL_DELIVERY_READY.receipt.json')
    prior = {}
    for original in completed:
        dest = HERE/('analysis_'+original)/'WORK_QUEUE.receipt.json'
        prior[original] = dict(source_queue_sha256=sha(HERE/original/'WORK_QUEUE.receipt.json'),
                               original_diagnosis_queue_exists=dest.exists())
    write(HERE/(prefix+'_HISTORY.json'), dict(
        utc=utc(), completed_stages=prior,
        evidence_gap='Original required saved-trace diagnoses, not further simulations; reuse every existing diagnostic and sealed queue.',
        implementation='Unchanged analysis_jobs_current.prepare; normal sole coordinator retains all dispatch and computation.',
        primary_available_claims=available, new_rollouts=0, new_training=0,
        new_diagnoses=0, submissions=0, science_worker_signals=0))
    outputs = []
    with (HERE/'SUBMISSION.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for original in completed:
            stage = prepare(original)
            verify_receipt(stage+'/WORK_QUEUE.receipt.json')
            outputs.append(dict(original=original, analysis=stage,
                                groups=len(read(HERE/stage/'GROUP_INDEX.json'))))
            print('PREPARED', stage, flush=True)
    write(HERE/(prefix+'.json'), dict(
        utc=utc(), queues=outputs, new_rollouts=0, new_training=0,
        new_diagnoses=0, submissions=0, science_worker_signals=0,
        sole_coordinator_unchanged=True, original_analysis_semantics_unchanged=True))
    seal(prefix+'.receipt.json', [Path(__file__), HERE/'analysis_jobs_current.py',
         HERE/(prefix+'_HISTORY.json'), HERE/(prefix+'.json'),
         *[HERE/r['analysis']/'WORK_QUEUE.receipt.json' for r in outputs]],
         new_rollouts=0, new_diagnoses=0, submissions=0)
    with (HERE/'ITERATION_LOG.md').open('a') as f:
        f.write('\n'+utc()+'：在逐机器人进度修正仍有充足未认领任务期间，'
                '管理核提前用原analysis_jobs_current.prepare封存已完成圆形全责、'
                '窄通道与门口变体的保存轨迹诊断队列。复用已有队列/诊断；'
                '无rollout、训练、诊断计算、作业提交或科学worker信号，'
                '原协调器及执行语义不改。见'+prefix+'.receipt.json。\n')
    print('PASS required completed-variant diagnosis queues prepared', flush=True)


if __name__ == '__main__':
    main()
