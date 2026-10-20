from common import *
import subprocess,shlex

def submit(stage,script,mem='5G'):
 q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C'],text=True);assert not (HERE/f'{stage}_SUBMISSION.json').exists();args=['sbatch','--parsable','--hold','-p','cpu','--cpus-per-task=1','--mem='+mem,'--time','04:00:00','-J','r37_'+stage,'-o',str(HERE/(stage+'_%j.log')),'--wrap',f'JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 {shlex.quote(sys.executable)} -B {shlex.quote(str(HERE/script))} run'];seal(f'{stage}_EXECUTION_FROZEN.receipt.json',[HERE/script,HERE/'PREREGISTRATION.md'],before_submission=True,no_rollouts=True,squeue=q);job=subprocess.check_output(args,text=True).strip();write(HERE/f'{stage}_SUBMISSION.json',dict(utc=utc(),job=job,args=args,squeue=q,held_until_capacity=True));return job
