"""Enforce the user's dated CPU allocation, only touching our Slurm jobs."""
import datetime as dt
import getpass
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
DEADLINE = dt.datetime.fromisoformat('2026-10-28T09:00:00+08:00')

def allowance(now=None):
    return 24 if (now or dt.datetime.now(dt.timezone.utc)) < DEADLINE else 18

def jobs():
    out = subprocess.check_output(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%j|%T|%C'], text=True)
    return [dict(job_id=a, name=b, state=c, cpus=int(d))
            for a,b,c,d in (line.split('|') for line in out.splitlines())]

def enforce():
    rows = jobs()
    active = [j for j in rows if j['state'] in ('RUNNING','COMPLETING','CONFIGURING')]
    other = sum(j['cpus'] for j in active if not j['name'].startswith('str_'))
    own = [j for j in active if j['name'].startswith('str_')]
    limit = max(0, allowance()-other)
    cancelled = []
    # One-CPU workers are preferred; pending jobs are also curtailed after deadline.
    total = sum(j['cpus'] for j in own)
    for j in sorted(own, key=lambda x: int(x['job_id'].split('_')[0]), reverse=True):
        if total <= limit: break
        subprocess.run(['scancel', j['job_id']], check=True)
        cancelled.append(j); total -= j['cpus']
    if allowance() == 18:
        remaining = max(0, limit-total)
        for j in rows:
            if j['state'] != 'PENDING' or not j['name'].startswith('str_'): continue
            if j['cpus'] <= remaining: remaining -= j['cpus']
            else:
                subprocess.run(['scancel', j['job_id']], check=True)
                cancelled.append(j)
    record = dict(time=dt.datetime.now(dt.timezone.utc).isoformat(), user_total_limit=allowance(),
                  other_user_jobs_cpus=other, this_experiment_allowance=limit,
                  observed_jobs=rows, cancelled=cancelled,
                  note='Cancellation releases actual Slurm allocation; unfinished requests must resume after cache preflight.')
    p=HERE/'RESOURCE_STATUS.json'; temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2)+'\n');temp.replace(p)
    if cancelled:
        with (HERE/'resource_events.jsonl').open('a') as f: f.write(json.dumps(record)+'\n')
    return record

if __name__ == '__main__':
    (HERE/'resource_watch.pid').write_text(str(os.getpid())+'\n')
    while not (HERE/'RESOURCE_WATCH_STOP').exists():
        try: enforce()
        except Exception as e:
            with (HERE/'resource_events.jsonl').open('a') as f:
                f.write(json.dumps(dict(time=time.time(),error=repr(e)))+'\n')
        # Enforce just before 09:00, then keep accounting for unrelated user jobs.
        seconds=(DEADLINE-dt.datetime.now(dt.timezone.utc)).total_seconds()
        time.sleep(min(10., max(.1, seconds)) if seconds>0 else 10.)
