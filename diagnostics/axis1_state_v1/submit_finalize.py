from common import *
import subprocess,shlex
prereg();job=read(HERE/'main_SUBMISSION.json')['job'];q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%T|%C|%b|%j'],text=True);occupied=sum(int(x.split('|')[2]) for x in q.splitlines());assert occupied+1<=18,q
for p in HERE.glob('*.py'):compile(p.read_text(),str(p),'exec')
assert 'PASS: full 5000-bootstrap' in (HERE/'STATISTICS_TESTS.log').read_text()
seal('FINALIZATION_FROZEN.receipt.json',list(HERE.glob('*.py'))+[HERE/'STATISTICS_TESTS.log',HERE/'UNIT_TESTS.log'],pre_result_analysis=True,raw_main_labels_opened=False)
args=['sbatch','--parsable','-p','cpu','-J','r32_finalize','--cpus-per-task=1','--mem=8G','--time','02:00:00','--dependency','afterok:'+job,'-o',str(HERE/'finalize_%j.log'),'--wrap',f'PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 {shlex.quote(sys.executable)} -B {shlex.quote(str(HERE/"finalize.py"))}']
newjob=subprocess.check_output(args,text=True).strip();write(HERE/'FINALIZE_SUBMISSION.json',dict(utc=utc(),job=newjob,args=args,prior_queue=q,depends_on=job));print(newjob)
