"""Atomic actual-attempt reservation; never launch attempt 90001."""
from core import *
import fcntl,time,os

LIMIT=90000

def used():
    p=HERE/'BUDGET.json'
    return read(p)['actual_attempts']if p.exists()else len(list((HERE/'attempts').glob('*.json')))

def reserve(c):
    with (HERE/'budget.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        n=used()
        if n>=LIMIT:raise RuntimeError('HARD_STOP actual new rollout budget exhausted (90000)')
        p=HERE/'attempts'/f"{c['rollout_uid']}_{time.time_ns()}.json"
        dump(p,dict(rollout_uid=c['rollout_uid'],job=os.environ.get('SLURM_JOB_ID'),time=time.time(),ordinal=n+1))
        dump(HERE/'BUDGET.json',dict(actual_attempts=n+1,hard_limit=LIMIT))
        return n+1
