"""Atomic counting of every actual launch, including infrastructure retries."""
from core import *
import fcntl,time,os
LIMIT=80000

def limit():return LIMIT

def used():
    p=HERE/'BUDGET.json';return read(p)['actual_attempts']if p.exists()else 0

def reserve(c):
    with(HERE/'budget.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);n=used()
        if n>=LIMIT:raise RuntimeError('HARD_STOP actual new rollout budget exhausted80000')
        dump(HERE/'attempts'/f"{c['rollout_uid']}_{time.time_ns()}.json",dict(rollout_uid=c['rollout_uid'],job=os.environ.get('SLURM_JOB_ID'),time=time.time(),ordinal=n+1))
        dump(HERE/'BUDGET.json',dict(actual_attempts=n+1,hard_limit=LIMIT));return n+1
