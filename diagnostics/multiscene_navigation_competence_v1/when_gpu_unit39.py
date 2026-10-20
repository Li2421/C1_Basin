"""Run a zero-physical-step nominal interface unit in one spare CPU slot."""
from common import *
from submit_cpu40 import budget
import subprocess,time,re,signal,shlex

while True:
    b=budget();rows=[r.split('|') for r in b['queue'].splitlines()]
    own_gpu=[r for r in rows if r[3].startswith('r40_') and r[4]=='RUNNING' and 'shard:' in r[6]]
    if not own_gpu or b['available']<1:time.sleep(5);continue
    pattern='^/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -B diagnostics/multiscene_navigation_competence_v1/pilot_training40.py$'
    pids=subprocess.check_output(['pgrep','-f',pattern],text=True).split();assert len(pids)==1;pid=int(pids[0]);os.kill(pid,signal.SIGSTOP)
    try:
        b=budget();node=subprocess.check_output(['scontrol','--oneliner','show','node','desktop'],text=True)
        free=int(re.search(r'RealMemory=(\d+)',node)[1])-int(re.search(r'AllocMem=(\d+)',node)[1])
        if b['available']<1 or free<1024:continue
        command=' '.join(['env','PYTHONDONTWRITEBYTECODE=1','JAX_PLATFORMS=cpu','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1',shlex.quote(sys.executable),'-B',shlex.quote(str(HERE/'test_nominal39_callback.py'))])
        args=['sbatch','--parsable','-p','cpu','--cpus-per-task=1','--mem=1G','--time=00:15:00','-J','r40_nominal39_unit','-o',str(HERE/'NOMINAL39_CALLBACK_UNIT_%j.log'),'--wrap',command]
        seal('NOMINAL39_CALLBACK_UNIT_SUBMISSION_PREFLIGHT.receipt.json',[HERE/n for n in ['test_nominal39_callback.py','nominal39_callback.py','generated_runtime.py','GENERATED_RUNTIME39_COPY_PROVENANCE.json']],queue=b['queue'],node=node,requested_cpu=1,requested_gpu=0,host_memory_GiB=1,physical_steps=0,primary_wait_coordinator_only_paused=True)
        job=subprocess.check_output(args,text=True).strip();write(HERE/'NOMINAL39_CALLBACK_UNIT_SUBMISSION.json',dict(utc=utc(),job=job,args=args,budget=b,node=node))
        while subprocess.check_output(['squeue','-h','-j',job],text=True).strip():time.sleep(5)
        p=HERE/'NOMINAL39_CALLBACK_UNIT_TEST.json';write(HERE/'NOMINAL39_CALLBACK_UNIT_EXECUTION.json',dict(utc=utc(),job=job,passed=p.exists() and read(p)['passed'],physical_steps=0,rollouts_registered=0))
        break
    finally:
        path=Path(f'/proc/{pid}/cmdline')
        if path.exists() and b'pilot_training40.py' in path.read_bytes():os.kill(pid,signal.SIGCONT)
