"""Bounded process launcher; workers use distinct CPU sets and frozen state shards."""
import os
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/c1_frozen_unseen_64';(OUT/'logs').mkdir(exist_ok=True)
cpus=sorted(os.sched_getaffinity(0));workers=8;processes=[];files=[]
for i in range(workers):
    affinity=cpus[i::workers];log=(OUT/'logs'/f'worker{i}.log').open('a');files.append(log)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    command=['taskset','-c',','.join(map(str,affinity)),str(ROOT/'.venv-c1/bin/python'),str(ROOT/'scripts/evaluate_c1_frozen_unseen.py'),'--worker',str(i),'--workers',str(workers)]
    processes.append(subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
last=-1
while any(p.poll() is None for p in processes):
    done=len(list((OUT/'states').glob('*.json')))
    if done!=last:print('completed states',done,'/64',flush=True);last=done
    time.sleep(1)
codes=[p.returncode for p in processes]
for f in files:f.close()
print('worker exit codes',codes,flush=True)
if any(c!=0 for c in codes):raise SystemExit(1)
