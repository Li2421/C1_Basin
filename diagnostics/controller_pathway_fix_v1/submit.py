from util import *
import subprocess
def submit(label,script,indices,gpu=True,dependency=None):
    current=subprocess.check_output(['squeue','-h','-r','-t','RUNNING','-o','%C|%j'],text=True)
    other=sum(int(l.split('|')[0]) for l in current.splitlines() if not l.split('|')[1].startswith('r6_'))
    own=sum(int(l.split('|')[0]) for l in current.splitlines() if l.split('|')[1].startswith('r6_'))
    slots=min(3 if gpu else 15,max(0,18-other-own))
    assert slots>0,'Shared CPU pool full: wait for a slot, do not overallocate or cancel other task'
    array=','.join(map(str,indices))+f'%{slots}'
    cmd=f'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS={"cuda" if gpu else "cpu"} {PY} {HERE}/{script} $SLURM_ARRAY_TASK_ID'
    args=['sbatch','--parsable','-J','r6_'+label,'-p','gpu' if gpu else 'cpu','--cpus-per-task=1','--mem=5G','--time=02:00:00','--array',array,'-o',str(HERE/(label+'_%A_%a.log'))]
    if gpu:args+=['--gres=shard:1']
    if dependency:args+=['--dependency','afterok:'+dependency]
    args+=['--wrap',cmd]
    job=subprocess.check_output(args,text=True).strip().split(';')[0]
    record=dict(job=job,name='r6_'+label,indices=indices,gpu=gpu,concurrency=slots,cpus_per_task=1,other_allocated_cpus=other,own_running_cpus_at_submission=own,shared_cpu_cap=18,gpu_cap=3,command=cmd,utc=utc(),dependency=dependency)
    f=HERE/'SLURM_SUBMISSIONS.json';old=read(f) if f.exists() else [];write(f,old+[record]);print(json.dumps(record));return job
if __name__=='__main__':
    if sys.argv[1]=='replay':submit('replay','replay.py',list(range(38)))
