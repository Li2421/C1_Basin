"""Independent deadline enforcement; never cancels another experiment's job."""
from experiment import *
import time
import subprocess
from resources_released import snapshot, limits

def enforce():
    state=snapshot();cancelled=[]
    cpu=state['scheduled_cpus'];gpu=state['gpu_shards']
    own=[r for r in state['jobs'] if r['name'].startswith('blv2_')]
    # Completed evidence remains on disk; only unfinished work is rescheduled.
    own.sort(key=lambda r:(r['gpu_shards'],int(r['job_id'].split('_')[0])),reverse=True)
    for job in own:
        if cpu+state['local_cpu_reserve']<=state['cpu_limit'] and gpu<=state['gpu_shard_limit']:
            break
        subprocess.run(['scancel',job['job_id']],check=True)
        cancelled.append(job)
        cpu-=job['cpus'];gpu-=job['gpu_shards']
    write(HERE/'RESOURCE_RELEASE_STATUS.json',dict(utc=utc(),snapshot=state,cancelled=cancelled,
        remaining_scheduled_cpus=cpu,remaining_gpu_shards=gpu,
        completed_raw_and_journals_preserved=True,unfinished_work='Infrastructure interruption; resume same missing cases, not unknown',
        within_limit=cpu+state['local_cpu_reserve']<=state['cpu_limit'] and gpu<=state['gpu_shard_limit']))

def main():
    write(HERE/'RESOURCE_GUARD_PID.json',dict(pid=os.getpid(),utc=utc(),policy=read(HERE/'RESOURCE_SCHEDULE_CORRECTED.json')))
    deadline=datetime.fromisoformat(read(HERE/'RESOURCE_SCHEDULE_CORRECTED.json')['release_at_utc'])
    while True:
        seconds=(deadline-datetime.now(timezone.utc)).total_seconds()
        if seconds<=0:break
        time.sleep(min(30.,seconds))
    while True:
        enforce()
        time.sleep(30.)

if __name__=='__main__':main()
