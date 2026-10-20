"""Fresh queue check immediately before every bounded shard submission."""
from common import *
import subprocess,shlex,time

def queue():return subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%P|%j|%T|%C|%b'],text=True)
def submit(stage,workers=18,gpu=False,script='generated_worker.py'):
    q=queue();rows=[v.split('|') for v in q.splitlines()]
    used=sum(int(v[4]) for v in rows if v[3] in ['RUNNING','PENDING','CONFIGURING'])
    used_gpu=sum(1 for v in rows if 'shard:' in v[-1] and v[3] in ['RUNNING','PENDING','CONFIGURING'])
    n=min(workers,2-used_gpu,18-used) if gpu else min(workers,18-used)
    if n<1:return None
    assert (HERE/f'{stage}_PREFLIGHT.receipt.json').exists() if not gpu else (HERE/'DATASET_FROZEN.receipt.json').exists()
    argument='$SLURM_ARRAY_TASK_ID' if gpu else shlex.quote(stage)+' $SLURM_ARRAY_TASK_ID'
    cmd=' '.join(['env','PYTHONDONTWRITEBYTECODE=1','JAX_PLATFORMS='+('cuda' if gpu else 'cpu'),'OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',shlex.quote(sys.executable),'-B',shlex.quote(str(HERE/script)),argument])
    args=['sbatch','--parsable','-p','gpu' if gpu else 'cpu','--cpus-per-task=1','--mem=4G','--time=06:00:00','--array',f'0-{n-1}%{n}','-J','r40_'+stage,'-o',str(HERE/(stage+'_%A_%a.log')),'--wrap',cmd]
    if gpu:args[1:1]=['--gres=shard:1']
    seal(f'{stage}_SUBMISSION_PREFLIGHT.receipt.json',[HERE/script],queue=q,requested_shards=n,gpu=gpu)
    job=subprocess.check_output(args,text=True).strip();write(HERE/f'{stage}_SUBMISSION.json',dict(utc=utc(),job=job,args=args,queue=q));print(job,flush=True);return job
def completed(job):return not subprocess.check_output(['squeue','-h','-j',job],text=True).strip()
def merge(stage):
    from registry import postflight
    with (HERE/f'{stage}_MERGE.log').open('w') as log:subprocess.run([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],check=True,stdout=log)
    postflight(stage)

if __name__=='__main__':submit(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 18)
