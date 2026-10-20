"""Bounded non-rollout jobs, with global queue accounting before submission."""
from common import *
from submit_cpu40 import budget
import subprocess,shlex,re

def submit(name,script,arguments=(),workers=1,gpu=False,memory_gib=2,freeze_paths=()):
    assert memory_gib in [2,8,16]
    b=budget();n=min(workers,b['available']);node=subprocess.check_output(['scontrol','--oneliner','show','node','desktop'],text=True)
    free=int(re.search(r'RealMemory=(\d+)',node)[1])-int(re.search(r'AllocMem=(\d+)',node)[1]);n=min(n,max(0,free//(memory_gib*1024)))
    if gpu:
        used=sum(sum(int(v) for v in re.findall(r'shard:(\d+)',line.split('|')[6])) for line in b['queue'].splitlines());n=min(n,2-used)
    if n<1:return None
    # A worker receives index and actual pool size. No rollout or DB mutation here.
    command=' '.join(['env','PYTHONDONTWRITEBYTECODE=1','JAX_PLATFORMS='+('cuda' if gpu else 'cpu'),'OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',shlex.quote(sys.executable),'-B',shlex.quote(str(HERE/script))]+[shlex.quote(str(a)) for a in arguments]+['$SLURM_ARRAY_TASK_ID',str(n)])
    args=['sbatch','--parsable','-p','gpu' if gpu else 'cpu','--cpus-per-task=1',f'--mem={memory_gib}G','--time=12:00:00','--array',f'0-{n-1}%{n}','-J','r40_'+name,'-o',str(HERE/f'{name}_%A_%a.log'),'--wrap',command]
    if gpu:args[1:1]=['--gres=shard:1']
    seal(f'{name}_SUBMISSION_PREFLIGHT.receipt.json',[HERE/script,HERE/'submit_aux40.py']+list(freeze_paths),budget=b,node=node,requested_cpu=n,requested_gpu=n if gpu else 0,host_memory_GiB_per_worker=memory_gib,new_rollouts=0)
    job=subprocess.check_output(args,text=True).strip();write(HERE/f'{name}_SUBMISSION.json',dict(utc=utc(),job=job,args=args,budget=b,node=node));print(job,flush=True);return job

def wait_submit(name,script,arguments=(),**options):
    from submit_stage import completed
    import time
    p=HERE/f'{name}_SUBMISSION.json';job=read(p)['job'] if p.exists() else None
    while job is None:
        job=submit(name,script,arguments,**options)
        if job is None:time.sleep(5)
    while not completed(job):time.sleep(5)
    return job
