"""Capacity-aware Slurm launch with the user's absolute 09:00 CPU reduction."""
from experiment import *
from resource_schedule import limit
import subprocess
import time

def main(stage,phase):
    assert not (HERE/'READONLY_BOUNDARY_STOP.json').exists(),'STOPPED: protected v2 cache write; do not restart without resolving the source-protection incident'
    # One explicit all-user queue observation immediately before submission.
    listing=subprocess.check_output(['squeue','-h','-r','-o','%i|%u|%j|%T|%C'],text=True)
    jobs=[]
    for line in listing.splitlines():
        jid,user,name,state,cpus=line.split('|');jobs.append(dict(job=jid,user=user,name=name,state=state,cpus=int(cpus)))
    own=sum(j['cpus'] for j in jobs if j['user']==os.environ.get('USER','zhihan'))
    other=any(j['user']!=os.environ.get('USER','zhihan') for j in jobs)
    budget=min(limit(),18 if other else 24)
    free=max(0,min(budget-1-own,24-sum(j['cpus'] for j in jobs)-1))
    if not free:raise RuntimeError('No authorized CPU capacity; wait without changing another task')
    job=subprocess.check_output(['sbatch','--parsable',f'--array=0-{free-1}',f'--export=ALL,FIX_STAGE={stage},FIX_PHASE={phase}',str(HERE/'worker_fix.sbatch')],text=True).strip()
    write(HERE/'submissions'/f'{time.time_ns()}_{job}.json',dict(utc=utc(),stage=stage,phase=phase,job_id=job,worker_count=free,all_user_jobs_before=jobs,budget=budget,release_deadline_singapore='2026-10-23T09:00:00+08:00',script_sha256=sha(HERE/'worker_fix.sbatch')))
    print(job,free,flush=True)
if __name__=='__main__':main(*sys.argv[1:])
