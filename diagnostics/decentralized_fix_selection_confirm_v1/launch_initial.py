from support import *
import subprocess
G=guard()
def main():
 prereg();q=subprocess.check_output(['squeue','-h','-r','-u','zhihan','-o','%i|%j|%T|%C|%b'],text=True);cpu=sum(int(l.split('|')[3]) for l in q.splitlines() if 'shard' not in l.split('|')[4]);gpu=0
 for l in q.splitlines():
  if 'shard' in l.split('|')[4]:
   import re
   m=re.search(r'shard(?::[^,:]+)?:(\d+)',l.split('|')[4]);gpu+=int(m.group(1)) if m else 1
 assert cpu+3<=18 and gpu+2<=2,(cpu,gpu,q);seal('INITIAL_SUBMISSION_PREFLIGHT.receipt.json',[HERE/n for n in ['INTEGRITY_BEGIN.json','PARALLEL_TASK_DIRECTORIES.json','PREREGISTRATION.md','PREREGISTRATION.receipt.json','FROZEN_RULES.json','train_gaussian.py','train_entry.py','replay.py','support.py']],squeue=q,planned_CPU_replay_shards=3,planned_GPU_training_shards=2,before_new_training_and_simulation=True,guard=guard())
 base=['sbatch','--parsable','--cpus-per-task=1','--mem=8G','--time=48:00:00'];py=sys.executable;jobs=[]
 args=base+['-p','gpu','--gres=shard:1','--array','0-1%2','-J','r30i_Gtrain','-o',str(HERE/'training_%A_%a.log'),'--wrap',f'PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cuda CUDA_CACHE_PATH={HERE}/cuda_cache {py} -B {HERE}/train_entry.py $SLURM_ARRAY_TASK_ID'];jobs.append(dict(kind='Gaussian training',job=subprocess.check_output(args,text=True).strip(),argv=args))
 for kind in ['round28','round33','round30e']:
  args=base+['-p','cpu','-J','r30i_'+kind,'-o',str(HERE/(kind+'_%j.log')),'--wrap',f'PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu {py} -B {HERE}/replay.py {kind}'];jobs.append(dict(kind=kind,job=subprocess.check_output(args,text=True).strip(),argv=args))
 write(HERE/'INITIAL_SUBMISSIONS.json',dict(utc=utc(),squeue=q,jobs=jobs,guard=guard()));print(jobs,flush=True)
if __name__=='__main__':main()
