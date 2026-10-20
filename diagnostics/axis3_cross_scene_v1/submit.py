from common import *
import subprocess,shlex

def phase(phase,entry='run.py',nextstep=None):
 prereg();assert not (HERE/f'{phase}_SUBMISSION.json').exists();q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C'],text=True);current=os.environ.get('SLURM_JOB_ID');others=sum(int(s.split('|')[3]) for s in q.splitlines() if s.split('|')[2] in ['RUNNING','COMPLETING','CONFIGURING'] and s.split('|')[0]!=current);slots=min(18-others,len(read(HERE/f'{phase}_MISSING_CASES.json')));assert slots>0,(others,q);args=['sbatch','--parsable','-p','cpu','--cpus-per-task=1','--mem=5G','--time','12:00:00','--array',f'0-{slots-1}%{slots}','-J','r34_'+phase,'-o',str(HERE/(phase+'_%A_%a.log'))];
 if current:args+=['--dependency','afterok:'+current]
 cmd=f'JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 {shlex.quote(sys.executable)} -B {shlex.quote(str(HERE/entry))} {phase} $SLURM_ARRAY_TASK_ID';args+=['--wrap',cmd];seal(f'{phase}_SUBMISSION_PREFLIGHT.receipt.json',[HERE/f'{phase}_PREFLIGHT.receipt.json',HERE/entry,HERE/'runtime.py',HERE/'PREREGISTRATION.md'],before_submission=True,slots=slots,queue=q);job=subprocess.check_output(args,text=True).strip();write(HERE/f'{phase}_SUBMISSION.json',dict(utc=utc(),job=job,slots=slots,args=args,queue=q));
 if nextstep:stage(nextstep,job)
 print('Submitted',phase,job,slots,flush=True);return job

def stage(step,parent):
 args=['sbatch','--parsable','-p','cpu','--cpus-per-task=1','--mem=12G','--time','04:00:00','--dependency','afterok:'+parent,'-J','r34_stage_'+step,'-o',str(HERE/(step+'_%j.log')),'--wrap',f'JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 {shlex.quote(sys.executable)} -B {shlex.quote(str(HERE/"advance.py"))} {step}'];job=subprocess.check_output(args,text=True).strip();write(HERE/f'{step}_STAGE_JOB.json',dict(utc=utc(),job=job,args=args));return job
if __name__=='__main__':phase(sys.argv[1],nextstep=sys.argv[1]+'_done')
