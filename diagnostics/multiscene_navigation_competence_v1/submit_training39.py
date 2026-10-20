"""At most two GPU shards and eighteen requested CPUs across every user job."""
from common import *
from submit_stage import queue
import subprocess,shlex,re

def submit(stage,tasks):
 assert (HERE/'TRAINING_PIPELINE39.receipt.json').exists()
 assert tasks and all(t['round_index'] in range(4) for t in tasks)
 from submit_cpu40 import budget
 import getpass
 b=budget();q=b['queue'];rows=[r.split('|') for r in q.splitlines()];active=[r for r in rows if r[0]==getpass.getuser()]
 gpu=sum(sum(int(n) for n in re.findall(r'shard:(\d+)',r[6])) for r in active)
 n=min(2,2-gpu,b['available'],len(tasks))
 if n<1:return None
 datasets=[]
 for t in tasks:
  p=HERE/'datasets39'/f"{t['family']}_{t['kind']}_{t['seed']}_round_{t['round_index']}.npz"
  assert (HERE/(str(p.relative_to(HERE))+'.receipt.json')).exists();datasets.append(p.with_suffix('.json'))
 # Training scripts use an immutable task list per submission.
 path=HERE/f'{stage}_TASKS.json';write(path,tasks)
 script=HERE/f'{stage}_entry.py';script.write_text("from common import *\nfrom training39 import train\nfor task in read(HERE/"+repr(path.name)+")[int(sys.argv[1])::"+str(n)+"]:\n try:\n  train(**task)\n except Exception as error:\n  tag=\"{family}_{kind}_{seed}_round_{round_index}\".format(**task)\n  write(HERE/\"training_errors\"/(tag+\".json\"),dict(task=task,type=type(error).__name__,message=str(error)))\n")
 command=' '.join(['env','PYTHONDONTWRITEBYTECODE=1','JAX_PLATFORMS=cuda','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',shlex.quote(sys.executable),'-B',shlex.quote(str(script)),'$SLURM_ARRAY_TASK_ID'])
 args=['sbatch','--parsable','--gres=shard:1','-p','gpu','--cpus-per-task=1','--mem=8G','--time=12:00:00','--array',f'0-{n-1}%{n}','-J','r40_'+stage,'-o',str(HERE/f'{stage}_%A_%a.log'),'--wrap',command]
 seal(f'{stage}_SUBMISSION_PREFLIGHT.receipt.json',[script,path,HERE/'training39.py',HERE/'TRAINING_PIPELINE39.receipt.json']+datasets,queue=q,requested_cpu_shards=n,requested_gpu_shards=n,budget=b)
 job=subprocess.check_output(args,text=True).strip();write(HERE/f'{stage}_SUBMISSION.json',dict(utc=utc(),job=job,args=args,queue=q));return job
