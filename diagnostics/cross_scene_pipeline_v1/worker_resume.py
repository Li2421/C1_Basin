"""One CPU allocation, source-only fits or journalled scientific execution."""
from common import *
import signal

STOP=False
PAUSE=False
def stop(*_):
    global STOP,PAUSE
    STOP=True;PAUSE=True

def main(stage):
    assert stage in ('pilot','main'), 'No retraining permitted by resume amendment'
    resume=read(HERE/'REQUEST_FIX_RESUME.receipt.json')
    assert resume['prior_stop_sha256']==sha(HERE/'STOP.receipt.json')
    assert resume['reconstruction_check_sha256']==sha(HERE/'REQUEST_RECONSTRUCTION_CHECK.json')
    for p,h in resume['active_execution_files'].items():assert sha(p)==h,p
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    verify_inputs();assert read(HERE/'REPLAY_CHECK.json')['passed']
    folder=HERE/'QUEUES'/stage;assert sha(folder/'INDEX.json')==read(folder/'QUEUE.receipt.json')['index_sha256']
    seconds=float(os.environ.get('XPL_TIME_LIMIT_S','10800'));end=time.monotonic()+seconds;deadline=end-3300
    job=(os.environ['SLURM_ARRAY_JOB_ID']+'_'+os.environ['SLURM_ARRAY_TASK_ID'])if 'SLURM_ARRAY_JOB_ID'in os.environ else os.environ.get('SLURM_JOB_ID','local')
    try:
        for e in read(folder/'INDEX.json'):
            if STOP or time.monotonic()>deadline or(HERE/'RESUME_STOP.receipt.json').exists():return
            done=folder/'done'/e['group'];claim=folder/'claims'/e['group']
            if done.exists():continue
            claim.parent.mkdir(parents=True,exist_ok=True)
            try:claim.mkdir()
            except FileExistsError:continue
            write(claim/'OWNER.json',dict(job=job,pid=os.getpid(),utc=utc(),stage=stage))
            assert sha(e['path'])==e['sha256']
            for task in read(e['path']):
                if STOP:return
                write(HERE/'worker_status'/f'{job}.json',dict(utc=utc(),stage=stage,task=task))
                if stage.startswith('train_'):
                    from train_model import fit
                    row=fit(task);key=task['fold']+'__'+task['mode']+'__'+str(task['seed'])
                    write(HERE/stage/'raw'/f'{key}.json',row)
                else:
                    from rollout_worker_resume import one
                    one(stage,task,pause=lambda:PAUSE or time.monotonic()>end-120)
            write(done,dict(utc=utc(),count=e['count']))
    except BaseException as exc:
        if type(exc).__name__=='CheckpointPause':
            print('Operational checkpoint saved; no scientific label.',str(exc),flush=True);return
        write(HERE/'WORKER_FATAL'/f'{os.getpid()}.json',dict(utc=utc(),stage=stage,type=type(exc).__name__,message=str(exc),traceback=__import__('traceback').format_exc()))
        raise

if __name__=='__main__':main(sys.argv[1])
