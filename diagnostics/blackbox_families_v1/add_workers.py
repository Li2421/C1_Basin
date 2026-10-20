from common import *
import subprocess,shlex
phase=sys.argv[1];limit=int(sys.argv[2]);label=sys.argv[3];assert (HERE/f'{phase}_SUBMISSION.json').exists();q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C|%b'],text=True);used=sum(int(x.split('|')[3]) for x in q.splitlines() if x.split('|')[2] in ['RUNNING','COMPLETING','CONFIGURING','PENDING']);slots=min(18-used,limit);assert slots>0
cmd=f'JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 {shlex.quote(sys.executable)} -B {shlex.quote(str(HERE/"worker.py"))} {shlex.quote(phase)} extra{label}_$SLURM_ARRAY_TASK_ID'
args=['sbatch','--parsable','-p','cpu','--cpus-per-task=1','--mem=4G','--time','24:00:00','--array',f'0-{slots-1}%{slots}','-J','r38_'+phase+'_extra','-o',str(HERE/(phase+'_extra'+label+'_%A_%a.log')),'--wrap',cmd]
seal(phase+'_EXTRA_'+label+'.receipt.json',[HERE/'worker.py',HERE/f'{phase}_PREFLIGHT.receipt.json'],queue=q,CPU_slots=slots,shared_cursor_unchanged=True,no_rerun_claimed_or_unknown=True)
job=subprocess.check_output(args,text=True).strip();write(HERE/f'{phase}_EXTRA_{label}.json',dict(job=job,args=args,slots=slots,queue=q));print(job,slots)
