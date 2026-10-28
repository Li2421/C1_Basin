"""Atomic actual-attempt reservation; never launch attempt 15001."""
from core import *
import fcntl,time,os

LIMITS={'original':15000,'followup':35000}

def phase():return os.environ.get('OWA_BUDGET_GROUP','original')
def limit():return LIMITS[phase()]

def used(group=None):
    p=HERE/'BUDGET.json'
    return read(p).get('by_phase',{}).get(group or phase(),0)if p.exists()else 0

def reserve(c):
    with (HERE/'budget.lock').open('a+')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        group=c.get('budget_group','original');n=used(group);cap=LIMITS[group]
        if n>=cap:raise RuntimeError(f'HARD_STOP {group} actual rollout budget exhausted ({cap})')
        p=HERE/'attempts'/f"{c['rollout_uid']}_{time.time_ns()}.json"
        dump(p,dict(rollout_uid=c['rollout_uid'],job=os.environ.get('SLURM_JOB_ID'),time=time.time(),budget_group=group,ordinal=n+1))
        counts={k:used(k)for k in LIMITS};counts[group]=n+1
        dump(HERE/'BUDGET.json',dict(actual_attempts=sum(counts.values()),by_phase=counts,hard_limits=LIMITS))
        return n+1
