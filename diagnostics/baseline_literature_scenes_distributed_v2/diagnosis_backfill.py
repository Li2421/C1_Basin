"""Use otherwise idle slots for required saved-trace analysis during the fixed tail.

No rollout submissions, learning choices or scientific parameters are changed.
Yield after the current diagnosis when the fixed stage finishes.
"""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from analysis_jobs_current import prepare
from coordinator_fast import stage_done, jobs_by_stage
from infrastructure_recovery_active import recover
from resources_released import snapshot, require_capacity
import subprocess
import fcntl
import time


def submit(stage):
    verify_receipt('DIAGNOSIS_BACKFILL_DEPLOYMENT.receipt.json')
    verify_receipt(f'{stage}/WORK_QUEUE.receipt.json')
    with (HERE/'SUBMISSION.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        before=require_capacity()
        job=subprocess.check_output(['sbatch','--parsable',
            f'--export=ALL,BLV2_STAGE={stage}',str(HERE/'analysis_backfill.sbatch')],text=True).strip()
        write(HERE/'submissions'/f'{time.time_ns()}_{job}.json',
            dict(utc=utc(),job_id=job,stage=stage,resource_snapshot=before,
                 posthoc_only=True,backfill=True,cooperative_yield=True))
        print('Submitted saved-trace backfill',job,stage,flush=True)
        return job


def main():
    with (HERE/'DIAGNOSIS_BACKFILL.lock').open('a') as driver_lock:
        fcntl.flock(driver_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_receipt('DIAGNOSIS_BACKFILL_DEPLOYMENT.receipt.json')
        fixed=[f'{layer}_fixed_{scene}' for layer in ('distributed','central') for scene in NAMES]
        originals=[f'distributed_baseline_{scene}' for scene in NAMES]+fixed
        # Only finished originals are eligible. A later main analysis uses the
        # same queue receipts and destination files, so nothing is recomputed.
        stages=[]
        for original in originals:
            if stage_done(original):
                stages.append(prepare(original))
                print('Prepared existing-trace stage',original,flush=True)
        owned=set()
        while True:
            state=snapshot();mapping=jobs_by_stage(state)
            live={j['job_id']:j for j in state['jobs']}
            phase=read(HERE/'STATUS.json').get('phase')
            yielding=all(stage_done(s) for s in fixed) or phase!='fixed_corrections'
            if yielding:
                for job in owned & set(live):
                    args=['scancel',job] if live[job]['state']=='PENDING' else ['scancel','--batch','--signal=USR1',job]
                    subprocess.run(args,check=False,capture_output=True,text=True)
                write(HERE/'DIAGNOSIS_BACKFILL_STATUS.json',dict(utc=utc(),phase='yielding_to_learning',jobs=sorted(owned & set(live)),new_rollouts=0))
                if not owned & set(live):break
                time.sleep(10);continue
            for stage in stages:recover(stage,set(mapping))
            available={stage:sum(not (HERE/stage/'claims'/r['group']).exists()
                for r in read(HERE/stage/'GROUP_INDEX.json')) for stage in stages}
            counts={stage:sum(s==stage for s in mapping.values()) for stage in stages}
            for _ in range(min(state['free_cpu_shares'],sum(available.values()))):
                targets=[s for s in stages if available[s]]
                stage=min(targets,key=lambda s:counts[s])
                owned.add(submit(stage));counts[stage]+=1;available[stage]-=1
            write(HERE/'DIAGNOSIS_BACKFILL_STATUS.json',dict(utc=utc(),phase='diagnosing_completed_traces',jobs=sorted(owned),stages=stages,new_rollouts=0))
            if all(stage_done(s) for s in stages):break
            time.sleep(10)
        write(HERE/'DIAGNOSIS_BACKFILL_EXIT.json',dict(utc=utc(),new_rollouts=0,completed_diagnoses_preserved=True))


if __name__=='__main__':main()
