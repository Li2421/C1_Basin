from common import *
import subprocess,shlex
name=sys.argv[1];entry={'fast_test':'test_fast_astar.py','expanded':'expanded.py','expanded_retry':'expanded.py','loto':'loto_analysis.py','analysis_valid':'analysis_valid_families.py','legacy_test':'test_legacy.py','corrected_test':'test_corrected_astar_reference.py','inventory':'import_inventory.py','plot':'plot_results.py'}[name];gpu=name.startswith('expanded')
q=subprocess.check_output(['squeue','-h','-r','-u',str(os.getuid()),'-o','%i|%j|%T|%C|%b'],text=True);used=sum(int(x.split('|')[3]) for x in q.splitlines() if x.split('|')[2] in ['RUNNING','COMPLETING','CONFIGURING','PENDING']);assert used<18,(used,q)
if gpu:assert sum('shard' in x.split('|')[4] for x in q.splitlines())<2
cmd=f'JAX_PLATFORMS={"cuda" if gpu else "cpu"} XLA_PYTHON_CLIENT_MEM_FRACTION=.40 PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 {shlex.quote(sys.executable)} -B {shlex.quote(str(HERE/entry))}'
args=['sbatch','--parsable','-p','gpu' if gpu else 'cpu','--cpus-per-task=1','--mem=8G','--time','04:00:00','-J','r38_'+name,'-o',str(HERE/(name+'_%j.log')),'--wrap',cmd]
if gpu:args[1:1]=['--gres=shard:1']
seal(name+'_SUBMISSION.receipt.json',[HERE/entry],queue=q,total_CPU_reserved_after=used+1,GPU_shards_requested=int(gpu));job=subprocess.check_output(args,text=True).strip();write(HERE/(name+'_SUBMISSION.json'),dict(utc=utc(),job=job,args=args,queue=q));print(job)
