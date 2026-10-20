from support import *
import subprocess,shlex
G=guard()
def main():
 phase=sys.argv[1];wanted=int(sys.argv[2]);prereg();assert read(HERE/'REPLAY_GATE.json')['passed'];assert (HERE/f'{phase}_PREFLIGHT.receipt.json').exists();q=subprocess.check_output(['squeue','-h','-r','-u','zhihan','-o','%i|%j|%T|%C|%b'],text=True);other=sum(int(x.split('|')[3]) for x in q.splitlines() if 'shard' not in x.split('|')[4]);slots=min(wanted,18-other,len(read(HERE/f'{phase}_MISSING_CASES.json')));assert slots>0,(other,q)
 cmd=f'PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu {shlex.quote(sys.executable)} -B {shlex.quote(str(HERE/"run.py"))} {phase} $SLURM_ARRAY_TASK_ID';args=['sbatch','--parsable','-p','cpu','--cpus-per-task=1','--mem=5G','--time','48:00:00','--array',f'0-{slots-1}%{slots}','-J','r30h_'+phase,'-o',str(HERE/(phase+'_%A_%a.log')),'--wrap',cmd];seal(f'{phase}_SUBMISSION_PREFLIGHT.receipt.json',[HERE/f'{phase}_PREFLIGHT.receipt.json',HERE/'run.py'],squeue=q,slots=slots,other_reserved=other,guard=guard());job=subprocess.check_output(args,text=True).strip();write(HERE/f'{phase}_SUBMISSION.json',dict(utc=utc(),job=job,slots=slots,args=args,queue=q));print(job,slots,flush=True)
if __name__=='__main__':main()
