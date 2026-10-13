"""User-authorized cutoff: only this study's tagged extra-capacity jobs."""
from common import *
import subprocess
def main():
    policy=read(HERE/'RESOURCE_AUTHORIZATION.json')
    assert datetime.now(timezone.utc)>=datetime.fromisoformat(policy['extra_capacity_cutoff_utc'])
    jobs=[]
    submissions=read(HERE/'SLURM_SUBMISSIONS.json') if (HERE/'SLURM_SUBMISSIONS.json').exists() else []
    owned_extra_ids={r['job_id'] for r in submissions if r['extra']}
    active=set()
    for line in subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j'],text=True).splitlines():
        job,name=line.split('|',1)
        if name.startswith('r5x_') and job.split('_')[0] in owned_extra_ids:
            active.add(job.split('_')[0])
    for job in sorted(active):
        result=subprocess.run(['scancel',job],text=True,capture_output=True)
        jobs.append(dict(job=job,returncode=result.returncode,stderr=result.stderr.strip()))
    write(HERE/'RESOURCE_CUTOFF_RECEIPT.json',dict(utc=utc(),cancelled_extra_jobs=jobs,active_caps=policy['after_cutoff'],historical_or_other_user_jobs_touched=False))
if __name__=='__main__':main()
