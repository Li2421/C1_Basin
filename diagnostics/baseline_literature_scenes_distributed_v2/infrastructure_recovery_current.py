"""Scheduler-state fallback and bounded-memory recovery without changing cases."""
from experiment import *
import subprocess
import re
import time

def job_state(job):
    proc=subprocess.run(['scontrol','show','job','-o',str(job)],capture_output=True,text=True)
    state=re.search(r'JobState=(\S+)',proc.stdout)
    return dict(state=state.group(1) if state else 'not_in_slurm',details=proc.stdout,stderr=proc.stderr)

def classify(stage,job):
    info=job_state(job)
    prefix='analysis' if stage.startswith('analysis_') else 'optional' if stage.startswith('optional_') else 'critic' if stage.startswith('critic_') else 'worker'
    path=HERE/'logs'/f'{prefix}_{job}.log';tail=path.read_text(errors='replace')[-20000:] if path.exists() else ''
    infrastructure=info['state'] in ('CANCELLED','TIMEOUT','PREEMPTED','NODE_FAIL','OUT_OF_MEMORY','BOOT_FAIL') or any(w in tail.lower() for w in ('oom_kill','out of memory','memoryerror','time limit','cancelled at','node failure','std::bad_alloc','cannot allocate memory'))
    # CPU-only JAX logs a benign CUDA-plugin traceback without stopping.
    own_trace="Traceback (most recent call last)" in tail and str(HERE) in tail
    return dict(utc=utc(),job_id=str(job),slurm=info,log=str(path),tail=tail,infrastructure=infrastructure,unhandled_implementation_error=bool(own_trace and not infrastructure))

def recover(stage,live_jobs):
    claims=HERE/stage/'claims';claims.mkdir(parents=True,exist_ok=True);recovered=[]
    for path in claims.iterdir():
        if not path.is_dir() or (path/'COMPLETE.json').exists():continue
        owner=path/'OWNER.json'
        if owner.exists():
            info=read(owner);job=str(info['job'])
            if job in live_jobs:continue
            review=classify(stage,job)
            if review['slurm']['state'] in ('RUNNING','PENDING','COMPLETING','CONFIGURING','SUSPENDED'):continue
            log=HERE/stage/'failure_reviews_v3'/f'{job}.json'
            if not log.exists():write(log,review)
            if review['unhandled_implementation_error']:
                write(HERE/stage/'NEEDS_IMPLEMENTATION_REVIEW.json',dict(utc=utc(),claim=str(path),review=str(log)));continue
            if review['slurm']['state']=='OUT_OF_MEMORY' and ('worker_reservation' in review['slurm']['details'] or stage.startswith('optional_')):
                memory=re.search(r'MinMemoryNode=(\d+)G',review['slurm']['details'])
                old=int(memory.group(1)) if memory else 4
                write(HERE/stage/'MEMORY_OVERRIDE_GB.json',dict(utc=utc(),memory_gb=min(96,old*2),reason='Observed certified-backend OOM; increase allocation only',job=job))
        else:
            if time.time()-path.stat().st_mtime<120:continue
            info=dict(reason='Interrupted before owner receipt')
        archive=HERE/stage/'claims_abandoned';archive.mkdir(exist_ok=True);dest=archive/f'{path.name}_{time.time_ns()}';path.rename(dest)
        recovered.append(dict(group=path.name,owner=info,archive=str(dest)))
    if recovered:write(HERE/stage/f'INFRA_RECOVERY_{time.time_ns()}.json',dict(utc=utc(),claims=recovered,completed_raw_preserved=True,unfinished_not_unknown=True,logs_read_before_retry=True))
    return recovered
