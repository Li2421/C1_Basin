"""Required diagnosis on the reserved local CPU; no claims, simulation or DB writes.

Use low priority and yield at a diagnosis boundary if Slurm analysis starts.
Prepared queues and numerical/provenance statements match the sealed worker.
"""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from diagnosis_alias_cache import reuse
from resources_released import snapshot
import numpy as np
import fcntl
import time
import signal

STOP=False


def stop(*args):
    global STOP
    STOP=True


def main():
    with (HERE/'LOCAL_SAVED_DIAGNOSIS.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_receipt('LOCAL_SAVED_DIAGNOSIS.receipt.json')
        verify_receipt('LOCAL_DIAGNOSIS_SCOPE.receipt.json')
        verify_receipt('POSTHOC_CACHE_DEPLOYMENT.receipt.json')
        verify_receipt('DOWNSTREAM_READY.receipt.json')
        signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
        os.nice(10)
        write(HERE/'LOCAL_SAVED_DIAGNOSIS_PID.json',dict(utc=utc(),pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),nice=os.nice(0)))
        from failure_analysis import summary
        scope=read(HERE/'LOCAL_DIAGNOSIS_SCOPE.json');written=0;cached=0;last_check=0
        for stage in scope['stages']:
            verify_receipt(f'{stage}/WORK_QUEUE.receipt.json')
            for entry in read(HERE/stage/'GROUP_INDEX.json'):
                assert sha(entry['file'])==entry['sha256']
                for row in read(entry['file']):
                    if time.monotonic()-last_check>=10:
                        jobs=snapshot()['jobs'];last_check=time.monotonic()
                        if any(j['name'].startswith('blv2_analysis') for j in jobs):
                            write(HERE/'LOCAL_SAVED_DIAGNOSIS_EXIT.json',dict(utc=utc(),reason='Yield to Slurm saved-trace analysis',written=written,new_rollouts=0));return
                        write(HERE/'LOCAL_SAVED_DIAGNOSIS_PROGRESS.json',dict(utc=utc(),stage=stage,written=written,already_cached=cached,new_rollouts=0,local_cpu_affinity=[0],nice=os.nice(0)))
                    if STOP:
                        write(HERE/'LOCAL_SAVED_DIAGNOSIS_EXIT.json',dict(utc=utc(),reason='Cooperative stop after completed diagnosis',written=written,new_rollouts=0));return
                    dest=HERE/'DIAGNOSTICS'/f"{row['rollout_uid']}.json"
                    if dest.exists():cached+=1;continue
                    assert sha(row['path'])==row['sha256'];raw=read(row['path'])
                    assert sha(raw['trace_path'])==raw['trace_sha256']
                    twin=reuse(raw)
                    if twin is None:
                        with np.load(raw['trace_path']) as z:result=summary(raw,z)
                        reuse_info={}
                    else:result=twin['result'];reuse_info=dict(alias_diagnosis_reuse=twin['provenance'])
                    write(dest,dict(utc=utc(),rollout_uid=row['rollout_uid'],source=row,trace_sha256=raw['trace_sha256'],**reuse_info,**result))
                    written+=1
        write(HERE/'LOCAL_SAVED_DIAGNOSIS_EXIT.json',dict(utc=utc(),reason='All prepared completed-source diagnoses available',written=written,already_cached=cached,new_rollouts=0))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        write(HERE/'LOCAL_SAVED_DIAGNOSIS_ERROR.json',dict(utc=utc(),error=repr(exc),traceback=traceback.format_exc(),new_rollouts=0))
        raise
