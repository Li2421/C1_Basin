"""User-authorized cutoff: only this study's tagged extra-capacity jobs."""
from common import *
import subprocess
def main():
    policy=read(HERE/'RESOURCE_AUTHORIZATION.json')
    assert datetime.now(timezone.utc)>=datetime.fromisoformat(policy['extra_capacity_cutoff_utc'])
    jobs=[]
    submissions=read(HERE/'SLURM_SUBMISSIONS.json') if (HERE/'SLURM_SUBMISSIONS.json').exists() else []
    owned_extra_ids={r['job_id'] for r in submissions if r['extra']}
    owned={r['job_id'] for r in submissions}
    def queue():
        rows=[]
        for line in subprocess.check_output(['squeue','-h','-r','-u',os.environ['USER'],'-o','%i|%j|%T'],text=True).splitlines():
            job,name,state=line.split('|',2)
            if name.startswith(('r5_','r5x_')) and job.split('_')[0] in owned:
                rows.append((job,name,state))
        return rows
    # Remove pending jobs first so freed CPU slots cannot immediately start
    # another old job. Pending base jobs are resubmitted with reduced caps.
    for job in sorted({j.split('_')[0] for j,n,s in queue()}):
        result=subprocess.run(['scancel','--state=PENDING',job],text=True,capture_output=True)
        jobs.append(dict(job=job,state_filter='PENDING',returncode=result.returncode,stderr=result.stderr.strip()))
    for job in sorted({j.split('_')[0] for j,n,s in queue() if j.split('_')[0] in owned_extra_ids}):
        result=subprocess.run(['scancel',job],text=True,capture_output=True)
        jobs.append(dict(job=job,state_filter='ALL EXTRA',returncode=result.returncode,stderr=result.stderr.strip()))
    import time
    # scancel is asynchronous. Wait for released jobs to leave the queue, then
    # enforce total CPU cap including the CPU consumed by each GPU shard job.
    for _ in range(30):
        if not any(j.split('_')[0] in owned_extra_ids for j,n,s in queue()):break
        time.sleep(1)
    remaining=[(j,n,s) for j,n,s in queue() if s not in ('PENDING','COMPLETING')]
    overflow=max(0,len(remaining)-policy['after_cutoff']['cpu_total_cap'])
    # Prefer interrupting raw CPU workers: their published raw is immutable and
    # the resume executor skips it. A resource interruption is not a numerical
    # failure and is recorded here separately from closed-loop outcomes.
    candidates=sorted(remaining,key=lambda r:('raw' not in r[1],r[0]))
    for job,name,state in candidates[:overflow]:
        result=subprocess.run(['scancel',job],text=True,capture_output=True)
        jobs.append(dict(job=job,state_filter='TOTAL CPU CAP',returncode=result.returncode,stderr=result.stderr.strip()))
    for _ in range(30):
        running=[(j,n,s) for j,n,s in queue() if s not in ('PENDING','COMPLETING')]
        if len(running)<=policy['after_cutoff']['cpu_total_cap']:break
        time.sleep(1)
    retained=sorted({j.split('_')[0] for j,n,s in queue() if s!='PENDING'})
    write(HERE/'RESOURCE_CUTOFF_RECEIPT.json',dict(utc=utc(),cancelled_jobs=jobs,active_caps=policy['after_cutoff'],retained_jobs=retained,historical_or_other_user_jobs_touched=False))
    from resource_resume import schedule
    schedule(retained)
if __name__=='__main__':main()
