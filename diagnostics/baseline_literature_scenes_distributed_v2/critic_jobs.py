"""The frozen v1 critic, with user-wide CPU and GPU share accounting."""
from experiment import *
from registry import verify_receipt
from resources import require_capacity
import subprocess
import fcntl
import time

def submit(scene):
    verify_receipt(f'LEARNING_BUNDLE_{scene}.receipt.json')
    with (HERE/'SUBMISSION.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);state=require_capacity(1,1)
        job=subprocess.check_output(['sbatch','--parsable',f'--export=ALL,BLV2_SCENE={scene}',str(HERE/'critic.sbatch')],text=True).strip()
        write(HERE/'submissions'/f'{time.time_ns()}_{job}.json',dict(utc=utc(),stage='critic_'+scene,job_id=job,resource_snapshot=state))
        return job
