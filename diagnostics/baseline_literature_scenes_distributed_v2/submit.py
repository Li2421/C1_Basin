"""Serialized, quota-aware submissions; every job has a sealed preflight."""
from experiment import *
from registry import verify_receipt
from resources import require_capacity,snapshot
import subprocess
import fcntl
import time

def submit(stage):
    prereg();verify_receipt('IMPLEMENTATION_FROZEN.receipt.json')
    verify_receipt(f'{stage}_PREFLIGHT.receipt.json');verify_receipt(f'{stage}/WORK_QUEUE.receipt.json')
    (HERE/'logs').mkdir(exist_ok=True)
    with (HERE/'SUBMISSION.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        before=require_capacity()
        job=subprocess.check_output(['sbatch','--parsable',f'--export=ALL,BLV2_STAGE={stage}',str(HERE/'worker.sbatch')],text=True).strip()
        write(HERE/'submissions'/f'{time.time_ns()}_{job}.json',dict(utc=utc(),stage=stage,job_id=job,resource_snapshot=before,preflight_receipt_sha256=sha(HERE/f'{stage}_PREFLIGHT.receipt.json'),script_sha256=sha(HERE/'worker.sbatch')))
        print('Submitted',job,stage,flush=True)
        return job

if __name__=='__main__':
    for _ in range(int(sys.argv[2])):submit(sys.argv[1])
