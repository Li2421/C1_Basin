"""Read failed-job logs before resuming only infrastructure-abandoned claims."""
from experiment import *
from work_queue import recover as original_recover
import subprocess

def recover(stage,live_jobs):
    claims=HERE/stage/'claims';claims.mkdir(parents=True,exist_ok=True)
    for claim in claims.iterdir():
        owner=claim/'OWNER.json'
        if not owner.exists() or (claim/'COMPLETE.json').exists():continue
        info=read(owner);job=str(info['job'])
        if job in live_jobs:continue
        prefix='analysis' if stage.startswith('analysis_') else 'optional' if stage.startswith('optional_') else 'worker'
        path=HERE/'logs'/f'{prefix}_{job}.log'
        tail=path.read_text(errors='replace')[-16000:] if path.exists() else ''
        state=subprocess.run(['sacct','-n','-j',job,'--format=State,ExitCode','-P'],capture_output=True,text=True)
        infrastructure=any(word in state.stdout for word in ('CANCELLED','TIMEOUT','PREEMPTED','NODE_FAIL','OUT_OF_MEMORY','BOOT_FAIL')) or any(word in tail.lower() for word in ('out of memory','oom_kill','memoryerror','time limit'))
        review=HERE/stage/'failure_reviews'/f'{job}.json'
        if not review.exists():write(review,dict(utc=utc(),job_id=job,slurm=state.stdout,log=str(path),tail=tail,infrastructure=infrastructure))
        if 'Traceback (most recent call last)' in tail and not infrastructure:
            write(HERE/stage/'NEEDS_IMPLEMENTATION_REVIEW.json',dict(utc=utc(),claim=str(claim),review=str(review),no_automatic_scientific_retry=True))
            return []
    return original_recover(stage,live_jobs)
