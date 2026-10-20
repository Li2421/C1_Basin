"""Same frozen training; bound concurrency by available host-memory reservations."""
from common import *
from submit_cpu40 import budget
import subprocess,shlex,re,getpass

def memory_slots(free_mib,memory_gib):return max(0,int(free_mib)//(int(memory_gib)*1024))

def submit(stage,tasks,host_memory_gib=None):
    from freeze_guard39 import verify
    verify();assert tasks
    manifests=[]
    for t in tasks:
        assert t['round_index'] in range(4)
        data=HERE/'datasets39'/f"{t['family']}_{t['kind']}_{t['seed']}_round_{t['round_index']}.npz"
        assert (HERE/(str(data.relative_to(HERE))+'.receipt.json')).exists();manifests.append(data.with_suffix('.json'))
    largest=max(read(p)['transitions']['train_X'] for p in manifests)
    memory=int(host_memory_gib or (16 if largest>3000000 else 8));assert memory in [8,16]
    b=budget();rows=[r.split('|') for r in b['queue'].splitlines() if r.split('|')[0]==getpass.getuser()]
    gpu=sum(sum(int(n) for n in re.findall(r'shard:(\d+)',r[6])) for r in rows)
    node=subprocess.check_output(['scontrol','--oneliner','show','node','desktop'],text=True)
    free=int(re.search(r'RealMemory=(\d+)',node)[1])-int(re.search(r'AllocMem=(\d+)',node)[1])
    n=min(2,2-gpu,b['available'],len(tasks),memory_slots(free,memory))
    if n<1:return None
    path=HERE/f'{stage}_TASKS.json';assert not path.exists();write(path,tasks)
    script=HERE/f'{stage}_entry.py';script.write_text("from common import *\nfrom training39 import train\nfor task in read(HERE/"+repr(path.name)+")[int(sys.argv[1])::"+str(n)+"]:\n try:\n  train(**task)\n except Exception as error:\n  tag=\"{family}_{kind}_{seed}_round_{round_index}\".format(**task)\n  write(HERE/\"training_errors\"/(tag+\".json\"),dict(task=task,type=type(error).__name__,message=str(error)))\n")
    command=' '.join(['env','PYTHONDONTWRITEBYTECODE=1','JAX_PLATFORMS=cuda','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',shlex.quote(sys.executable),'-B',shlex.quote(str(script)),'$SLURM_ARRAY_TASK_ID'])
    args=['sbatch','--parsable','--gres=shard:1','-p','gpu','--cpus-per-task=1',f'--mem={memory}G','--time=12:00:00','--array',f'0-{n-1}%{n}','-J','r40_'+stage,'-o',str(HERE/f'{stage}_%A_%a.log'),'--wrap',command]
    seal(f'{stage}_SUBMISSION_PREFLIGHT.receipt.json',[script,path,Path(__file__),HERE/'training39.py',HERE/'TRAINING39_CONFIG_FROZEN.json']+manifests,queue=b['queue'],node=node,requested_cpu_shards=n,requested_gpu_shards=n,host_memory_GiB_per_worker=memory,largest_training_states=largest,budget=b,no_training_choices_changed=True)
    job=subprocess.check_output(args,text=True).strip();write(HERE/f'{stage}_SUBMISSION.json',dict(utc=utc(),job=job,args=args,budget=b,node=node));return job
