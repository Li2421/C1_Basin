from common import *
import subprocess,shlex

def phase(phase,dependency=None):
 prereg();assert not (HERE/f'{phase}_SUBMISSION.json').exists();q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C'],text=True);others=sum(int(s.split('|')[3]) for s in q.splitlines() if s.split('|')[2] in ['RUNNING','COMPLETING','CONFIGURING']);slots=min(18,len(read(HERE/f'{phase}_MISSING_CASES.json')));throttle=min(slots,max(1,18-others))
 assert slots>0,(others,q)
 args=['sbatch','--parsable','-p','cpu','--cpus-per-task=1','--mem=5G','--time','24:00:00','--array',f'0-{slots-1}%{throttle}','-J','r35_'+phase,'-o',str(HERE/(phase+'_%A_%a.log'))]
 if dependency:args+=['--dependency','afterany:'+dependency]
 cmd=f'JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 {shlex.quote(sys.executable)} -B {shlex.quote(str(HERE/"run.py"))} {phase} $SLURM_ARRAY_TASK_ID';args+=['--wrap',cmd];seal(f'{phase}_SUBMISSION_PREFLIGHT.receipt.json',[HERE/f'{phase}_PREFLIGHT.receipt.json',HERE/'run.py',HERE/'runtime.py',HERE/'PREREGISTRATION.md'],before_submission=True,slots=slots,queue=q,dependency=dependency);job=subprocess.check_output(args,text=True).strip();write(HERE/f'{phase}_SUBMISSION.json',dict(utc=utc(),job=job,slots=slots,args=args,queue=q));print(job,flush=True);return job
if __name__=='__main__':phase(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else None)
