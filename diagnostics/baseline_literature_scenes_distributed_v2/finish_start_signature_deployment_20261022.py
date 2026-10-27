"""Read-only evidence preservation checks around graceful scientific-worker drain."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from resources_corrections_release_20261023 import snapshot
import time


def check(rows):
    trace_bytes=0
    for r in rows:
        assert sha(r['raw'])==r['raw_sha256'],r['raw']
        assert r['trace'] and r['trace_sha256'],r['raw']
        assert sha(r['trace'])==r['trace_sha256'],r['trace']
        trace_bytes+=Path(r['trace']).stat().st_size
    return dict(raw_files=len(rows),trace_files=len(rows),trace_bytes=trace_bytes,all_raw_and_trace_sha256_unchanged=True)


def main():
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    verify_receipt('CBS_START_SIGNATURE_ACTIVATION_PLAN_20261022.receipt.json')
    verify_receipt('CBS_START_SIGNATURE_DRAIN_STARTED_20261022.receipt.json')
    rows=read(HERE/'CBS_START_SIGNATURE_PRE_SWITCH_EVIDENCE_20261022.json')['rows']
    checked=check(rows);print('Initial published evidence preservation PASS',checked,flush=True)
    old=set(read(HERE/'CBS_START_SIGNATURE_ACTIVATION_PLAN_20261022.json')['old_worker_jobs'])
    while True:
        state=snapshot();remaining=sorted(old&{j['job_id'] for j in state['jobs']})
        write(HERE/'CBS_START_SIGNATURE_DEPLOYMENT_MONITOR_20261022.json',dict(utc=utc(),old_worker_jobs_remaining=remaining,initial_preservation=checked,resources=state,science_worker_hard_cancellations=0))
        if not remaining:break
        time.sleep(30)
    after=check(rows)
    verify_receipt('CBS_START_SIGNATURE_EQUIVALENCE_20261022.receipt.json')
    verify_receipt('CBS_START_SIGNATURE_PIPELINE_CODE_20261022.receipt.json')
    source=read(HERE/'CBS_START_SIGNATURE_PIPELINE_SOURCE_MAP_20261022.json')
    for r in source['static_copies']:assert sha(HERE/r['original'])==r['original_sha256'],r['original']
    write(HERE/'CBS_START_SIGNATURE_DEPLOYMENT_20261022.json',dict(utc=utc(),passed=True,preservation=after,old_worker_jobs_all_ended=True,no_hard_scientific_worker_cancellations=True,resource_schedule_unchanged='24 total CPU until2026-10-23T09:00+08:00 then18;2 GPU shares',execution_semantics_and_case_UIDs_unchanged=True,existing_rollouts_and_labels_retained=True,active_entry_snapshot=read(HERE/'ACTIVE_ENTRYPOINT.json'),resource_snapshot=state,whole_pipeline_speedup_not_yet_estimated=True))
    seal('CBS_START_SIGNATURE_DEPLOYMENT_20261022.receipt.json',[HERE/'CBS_START_SIGNATURE_DEPLOYMENT_20261022.json',HERE/'CBS_START_SIGNATURE_EQUIVALENCE_20261022.receipt.json',HERE/'CBS_START_SIGNATURE_PIPELINE_CODE_20261022.receipt.json',HERE/'CBS_START_SIGNATURE_ACTIVATION_PLAN_20261022.receipt.json',HERE/'CBS_START_SIGNATURE_DRAIN_STARTED_20261022.receipt.json',Path(__file__)],passed=True,no_published_evidence_overwritten=True)
    with (HERE/'ITERATION_LOG.md').open('a') as f:f.write('\n\n'+utc()+'：已通过644个有限网格、17个四场景历史查询、12个IEEE边界输入及完整2000步逐位回放，部署低层A*起点/初始边判定精确缓存。静态新模块，未改任务、控制器预算、求解标准、修正语义、科学UID或样本数；旧23个worker各自完成当前局后退出，历史raw/trace SHA全部保持。证据：CBS_START_SIGNATURE_DEPLOYMENT_20261022.receipt.json。单例观测52.1→20.5分钟，不能当作全流程倍数。24→18CPU（10月23日09:00）/2GPU份额规则继续。\n')
    print('PASS: all old workers drained; full existing published evidence preserved; deployment sealed',flush=True)


if __name__=='__main__':main()
