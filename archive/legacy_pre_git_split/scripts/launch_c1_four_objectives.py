"""Four bounded training workers; each evaluates every final checkpoint."""
import os,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/c1_four_objectives_multiseed'
cpus=sorted(os.sched_getaffinity(0));arms=['P','PS','Pg','full']
def call(script,args,slot,name):
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (OUT/'logs'/f'{name}.log').open('a') as f:
        command=['taskset','-c',','.join(map(str,cpus[slot::4])),str(ROOT/'.venv-c1/bin/python'),str(ROOT/'scripts'/script),*args]
        r=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError(f'{name} failed with exit{r.returncode}')
def worker(slot):
    arm=arms[slot]
    for seed in [0,1,2]:
        name=f'{arm}_seed{seed}'
        call('train_c1_four_objectives.py',['--arm',arm,'--seed',str(seed)],slot,'train_'+name)
        print('TRAIN COMPLETE',name,flush=True)
        call('evaluate_c1_four_objectives.py',['--arm',arm,'--seed',str(seed)],slot,'eval_'+name)
        print('EVAL COMPLETE',name,flush=True)
    if slot==0:call('evaluate_c1_four_objectives.py',['--arm','baseline'],slot,'eval_baseline')
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(worker,range(4)))
print('ALL TRAINING AND EVALUATION COMPLETE',flush=True)
