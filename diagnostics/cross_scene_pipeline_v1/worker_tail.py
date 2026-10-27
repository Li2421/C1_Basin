"""Per-request leases for remaining frozen requests; same rollout function and UID."""
from common import *
from rollout_worker_resume import one
import signal

STOP=False
PAUSE=False
def stop(*_):
    global STOP,PAUSE
    STOP=True;PAUSE=True

def main():
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    verify_inputs();resume=read(HERE/'REQUEST_FIX_RESUME.receipt.json')
    for p,h in resume['active_execution_files'].items():assert sha(p)==h,p
    proof=read(HERE/'TAIL_REBALANCE.receipt.json');assert sha(HERE/'TAIL_CASES.json')==proof['cases_sha256']
    job=(os.environ['SLURM_ARRAY_JOB_ID']+'_'+os.environ['SLURM_ARRAY_TASK_ID'])if 'SLURM_ARRAY_JOB_ID'in os.environ else os.environ['SLURM_JOB_ID']
    end=time.monotonic()+float(os.environ.get('XPL_TIME_LIMIT_S','10800'));deadline=end-3300
    cases=read(HERE/'TAIL_CASES.json')
    while not STOP and time.monotonic()<deadline:
        if(HERE/'RESUME_STOP.receipt.json').exists()or(HERE/'TAIL_FATAL.json').exists():return
        state=read(HERE/'TAIL_GUARDS.json');protected=set(state['protected']);did_work=False
        for c in cases:
            rid=c['rollout_uid'];raw=HERE/'main/raw'/f'{rid}.json'
            if raw.exists()or rid in protected:continue
            claim=HERE/'TAIL_CLAIMS'/rid;claim.parent.mkdir(exist_ok=True)
            try:claim.mkdir()
            except FileExistsError:continue
            write(claim/'OWNER.json',dict(job=job,pid=os.getpid(),utc=utc(),rollout_uid=rid))
            # Recheck publication/guard immediately after the exclusive claim.
            if raw.exists()or rid in set(read(HERE/'TAIL_GUARDS.json')['protected']):continue
            write(HERE/'worker_tail_status'/f'{job}.json',dict(utc=utc(),task=c))
            try:one('main',c,pause=lambda:PAUSE or time.monotonic()>end-120)
            except BaseException as exc:
                if type(exc).__name__=='CheckpointPause':return
                write(HERE/'TAIL_FATAL.json',dict(utc=utc(),job=job,type=type(exc).__name__,message=str(exc),traceback=__import__('traceback').format_exc()));raise
            write(claim/'DONE.json',dict(utc=utc()));did_work=True
            if STOP:return
        if all((HERE/'main/raw'/f"{c['rollout_uid']}.json").exists()for c in cases):return
        if not did_work:time.sleep(2)

if __name__=='__main__':main()
