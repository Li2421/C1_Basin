"""Per-UID exclusive execution across cost groups; no blocked task-group tails."""
from common import *
from rollout_worker_resume import one
from requests_corrected import verify_freezes
import signal

STOP=False
PAUSE=False
def stop(*_):
    global STOP,PAUSE
    STOP=True;PAUSE=True

def main(stage):
    assert stage in('pilot','main')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    verify_inputs();verify_freezes();assert read(HERE/'CHECKPOINT_COMPATIBILITY.json')['passed']
    for p,h in read(HERE/'EXECUTION_FROZEN.receipt.json')['files'].items():assert sha(p)==h,p
    folder=HERE/'QUEUES'/stage;assert sha(folder/'INDEX.json')==read(folder/'QUEUE.receipt.json')['index_sha256']
    groups=read(folder/'INDEX.json');tasks=[]
    for g in groups:
        assert sha(g['path'])==g['sha256'];tasks.extend(read(g['path']))
    job=os.environ['SLURM_ARRAY_JOB_ID']+'_'+os.environ['SLURM_ARRAY_TASK_ID'];end=time.monotonic()+float(os.environ.get('XP2_TIME_LIMIT_S','10800'));deadline=end-3300
    while not STOP and time.monotonic()<deadline:
        if(HERE/'STOP.receipt.json').exists():return
        worked=False
        for c in tasks:
            if STOP or time.monotonic()>deadline:return
            rid=c['rollout_uid'];raw=HERE/stage/'raw'/f'{rid}.json';marker=HERE/stage/'journal_receipts'/f'{rid}.json'
            if raw.exists()and marker.exists():continue
            claim=folder/'uid_claims'/rid;claim.parent.mkdir(parents=True,exist_ok=True)
            try:claim.mkdir()
            except FileExistsError:continue
            write(claim/'OWNER.json',dict(utc=utc(),job=job,pid=os.getpid(),stage=stage,rollout_uid=rid))
            write(HERE/'worker_status'/f'{job}.json',dict(utc=utc(),stage=stage,task=c))
            try:one(stage,c,pause=lambda:PAUSE or time.monotonic()>end-120)
            except BaseException as exc:
                if type(exc).__name__=='CheckpointPause':return
                write(HERE/'WORKER_FATAL'/f'{os.getpid()}.json',dict(utc=utc(),job=job,stage=stage,type=type(exc).__name__,message=str(exc),rollout_uid=rid,traceback=__import__('traceback').format_exc()));raise
            write(claim/'DONE.json',dict(utc=utc()));worked=True
        if all((HERE/stage/'raw'/f"{c['rollout_uid']}.json").exists()and(HERE/stage/'journal_receipts'/f"{c['rollout_uid']}.json").exists()for c in tasks):return
        if not worked:time.sleep(2)

if __name__=='__main__':main(sys.argv[1])
