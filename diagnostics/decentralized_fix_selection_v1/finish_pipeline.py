from support import *
import subprocess,time
G=guard()
def call(script,*args,python=None):
 r=subprocess.run([python or sys.executable,'-B',str(HERE/script),*args],text=True,capture_output=True);print(r.stdout+r.stderr,flush=True);assert r.returncode==0,(script,r.returncode)
def main():
 while not (HERE/'SIMULATION_COMPLETE.json').exists():
  if (HERE/'ORCHESTRATION_FAILED.json').exists():raise RuntimeError('simulation orchestration stopped')
  time.sleep(20)
 call('postflight.py','supplement');pf=planner.preflight(HERE/'ALL_PLANNED_ROLLOUTS.json');write(HERE/'ALL_CACHE_POSTFLIGHT.json',pf);call('pack_labels.py');q=subprocess.check_output(['squeue','-h','-r','-u','zhihan','-o','%i|%j|%T|%C|%b'],text=True)
 while q.strip():
  time.sleep(10);q=subprocess.check_output(['squeue','-h','-r','-u','zhihan','-o','%i|%j|%T|%C|%b'],text=True)
 assert not q.strip();args=['sbatch','--parsable','-p','cpu','--cpus-per-task=1','--mem=5G','--time','24:00:00','--array','0-4%5','-J','r30h_critic','-o',str(HERE/'critic_%A_%a.log'),'--wrap',f'PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu {sys.executable} -B {HERE}/fit_selector.py $SLURM_ARRAY_TASK_ID'];write(HERE/'CRITIC_SUBMISSION_PREFLIGHT.json',dict(utc=utc(),squeue=q,cpu_shards=5,argv=args,guard=guard()));job=subprocess.check_output(args,text=True).strip();write(HERE/'CRITIC_SUBMISSION.json',dict(utc=utc(),job=job));print('critic job',job,flush=True)
 while not all((HERE/'folds'/f/'CHOICES_FROZEN.receipt.json').exists() for f in FAMILIES):
  q=subprocess.check_output(['squeue','-h','-r','-u','zhihan','-o','%j'],text=True)
  if 'r30h_critic' not in q:raise RuntimeError('Critic jobs ended with incomplete folds. Inspect; preserve evidence.')
  time.sleep(20)
 call('analyze.py');call('plots.py',python='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1-plots/bin/python');call('integrity.py','END');call('report.py');call('verify_delivery.py');print('ALL COMPLETE',utc(),flush=True)
if __name__=='__main__':main()
