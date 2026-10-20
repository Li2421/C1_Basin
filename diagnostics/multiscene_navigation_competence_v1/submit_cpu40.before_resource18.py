"""User-authorized six spare CPUs; all users and pending requests counted."""
from common import *
import subprocess,shlex

def budget():
 q=subprocess.check_output(['squeue','-h','-r','-o','%u|%i|%P|%j|%T|%C|%b'],text=True);rows=[v.split('|') for v in q.splitlines()]
 nodes=subprocess.check_output(['sinfo','-p','cpu','-N','-h','-o','%N|%C'],text=True);idle=0;total=0
 for row in nodes.splitlines():
  values=row.split('|')[1].split('/');idle+=int(values[1]);total+=int(values[3])
 own=sum(int(r[5]) for r in rows if r[3].startswith('r40_'));requested=sum(int(r[5]) for r in rows)
 return dict(queue=q,nodes=nodes,own_cpu_requests=own,all_cpu_requests=requested,idle_cpu=idle,total_cpu=total,available=max(0,min(6-own,idle,total-requested)))

def submit(stage,workers=6,gpu=False,script='expert_worker.py',pool_name=None):
 assert not gpu
 b=budget();n=min(workers,b['available'])
 if n<1:return None
 name=pool_name or stage
 assert (HERE/f'{stage}_PREFLIGHT.receipt.json').exists()
 if pool_name is None:
  pf=planner.preflight(HERE/f'{stage}_PLANNED_ROLLOUTS.json');write(HERE/f'{stage}_FINAL_CACHE_PREFLIGHT.json',pf)
  from shared_rollout_db.src.rollout_db import connect
  with connect(True) as db:
   present=[c['rollout_uid'] for c in read(HERE/f'{stage}_MISSING_CASES.json') if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()]
  assert not present,'New exact cache records appeared: reuse them before submission'
  seal(f'{stage}_FINAL_PREFLIGHT.receipt.json',[HERE/f'{stage}_FINAL_CACHE_PREFLIGHT.json',HERE/f'{stage}_MISSING_CASES.json'],still_genuinely_missing=True)
 command=' '.join(['env','PYTHONDONTWRITEBYTECODE=1','JAX_PLATFORMS=cpu','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',shlex.quote(sys.executable),'-B',shlex.quote(str(HERE/script)),shlex.quote(stage),'$SLURM_ARRAY_TASK_ID'])
 args=['sbatch','--parsable','-p','cpu','--cpus-per-task=1','--mem=2G','--time=12:00:00','--array',f'0-{n-1}%{n}','-J','r40_'+name,'-o',str(HERE/f'{name}_%A_%a.log'),'--wrap',command]
 seal(f'{name}_SUBMISSION_PREFLIGHT.receipt.json',[HERE/script,HERE/'PREREGISTRATION_RESOURCE6_AMENDMENT.receipt.json',HERE/f'{stage}_PREFLIGHT.receipt.json'],budget=b,requested_shards=n,gpu=False)
 job=subprocess.check_output(args,text=True).strip();write(HERE/f'{name}_SUBMISSION.json',dict(utc=utc(),job=job,args=args,budget=b));print(job,flush=True);return job
