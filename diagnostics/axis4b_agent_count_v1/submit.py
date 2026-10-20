from common import *
import subprocess,shlex

def phase(stage):
 entry='run_derive.py' if stage in ['source','target'] else 'run.py'
 prereg();assert not (HERE/f'{stage}_SUBMISSION.json').exists();q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C'],text=True);allocated=sum(int(s.split('|')[3]) for s in q.splitlines() if s.split('|')[2] in ['RUNNING','COMPLETING','CONFIGURING']);slots=min(18,len(read(HERE/f'{stage}_MISSING_CASES.json')));assert slots>0;throttle=min(slots,max(1,18-allocated));args=['sbatch','--parsable','-p','cpu','--cpus-per-task=1','--mem=6G','--time','3-00:00:00','--array',f'0-{slots-1}%{throttle}','-J','r36_'+stage,'-o',str(HERE/(stage+'_%A_%a.log'))]
 if allocated>=18:args+=['--hold']
 args+=['--wrap',f'JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 {shlex.quote(sys.executable)} -B {shlex.quote(str(HERE/entry))} {stage} $SLURM_ARRAY_TASK_ID'];seal(f'{stage}_SUBMISSION_PREFLIGHT.receipt.json',[HERE/f'{stage}_PREFLIGHT.receipt.json',HERE/entry,HERE/'exact_reuse.py',HERE/'runtime.py',HERE/'PREREGISTRATION.md'],before_submission=True,queue=q,slots=slots,throttle=throttle,held_for_capacity=allocated>=18);job=subprocess.check_output(args,text=True).strip();write(HERE/f'{stage}_SUBMISSION.json',dict(utc=utc(),job=job,args=args,queue=q,slots=slots));print('Submitted',stage,job,flush=True);return job
if __name__=='__main__':phase(sys.argv[1])
