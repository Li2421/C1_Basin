"""Required optional trace diagnosis on spare CPUs; no simulation or DB writes."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from resources_released import require_capacity,snapshot
from coordinator_pipelined import jobs_by_stage,stage_done
from analysis_jobs_current import prepare
from infrastructure_recovery_active import recover
import fcntl
import subprocess
import time


def submit(stage):
    verify_receipt('OPTIONAL_FINE_SCHEDULING.receipt.json')
    verify_receipt(f'{stage}/WORK_QUEUE.receipt.json')
    with (HERE/'SUBMISSION.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);before=require_capacity()
        job=subprocess.check_output(['sbatch','--parsable',f'--export=ALL,BLV2_STAGE={stage}',
            str(HERE/'optional_analysis_backfill.sbatch')],text=True).strip()
        write(HERE/'submissions'/f'{time.time_ns()}_{job}.json',dict(utc=utc(),job_id=job,
            stage=stage,resource_snapshot=before,optional_backfill=True,posthoc_only=True))
        return job


def yield_jobs(state):
    live={j['job_id'] for j in state['jobs']}
    for path in (HERE/'submissions').glob('*.json'):
        row=read(path)
        if row.get('optional_backfill') and row['job_id'] in live:
            subprocess.run(['scancel','--batch','--signal=USR1',row['job_id']],capture_output=True,text=True)


def fill():
    state=snapshot()
    if not state['free_cpu_shares']:return
    stages=[]
    for path in HERE.glob('optional_*/GROUP_INDEX.json'):
        original=path.parent.name
        if original.startswith('optional_nominal_') or not stage_done(original):continue
        stages.append(prepare(original))
    mapping=jobs_by_stage(state)
    for stage in stages:recover(stage,set(mapping))
    available={s:sum(not (HERE/s/'claims'/r['group']).exists()
        for r in read(HERE/s/'GROUP_INDEX.json')) for s in stages}
    counts={s:sum(t==s for t in mapping.values()) for s in stages}
    for _ in range(min(state['free_cpu_shares'],sum(available.values()))):
        stage=min((s for s in stages if available[s]),key=lambda s:counts[s])
        submit(stage);counts[stage]+=1;available[stage]-=1
