"""Respect idle CPU authorization; yield only at complete episode boundaries."""
from experiment import *
from atomic_write import write
from resources_corrections_20261021 import snapshot
from registry import verify_receipt
import subprocess,time,signal

def main():
    verify_receipt('CORRECTION_RESOURCE_CODE_20261021.receipt.json')
    write(HERE/'RESOURCE_GUARD_PID.json',dict(utc=utc(),pid=os.getpid(),entrypoint='resource_guard_corrections_20261021.py'))
    while not (HERE/'V2_COMPLETE.json').exists():
        state=snapshot();cpu=state['scheduled_cpus'];gpu=state['gpu_shards'];yielded=[]
        for job in sorted(state['jobs'],key=lambda r:int(r['job_id'].split('_')[0]),reverse=True):
            if cpu+1<=state['cpu_limit'] and gpu<=2:break
            if not job['name'].startswith('blv2_'):continue
            if job['state']=='PENDING':args=['scancel',job['job_id']]
            else:args=['scancel','--batch','--signal=USR1',job['job_id']]
            result=subprocess.run(args,capture_output=True,text=True)
            yielded.append(dict(job=job,result=result.returncode,completed_episode_boundary=job['state']!='PENDING'))
            cpu-=job['cpus'];gpu-=job['gpu_shards']
        write(HERE/'RESOURCE_RELEASE_STATUS.json',dict(utc=utc(),snapshot=state,yield_requests=yielded,hard_cancelled_running_rollouts=0,within_limit=state['scheduled_cpus']+1<=state['cpu_limit'] and state['gpu_shards']<=2,unfinished_episode_drain=bool(yielded)))
        time.sleep(10)
if __name__=='__main__':main()
