"""Eight independent CPU workers for the frozen mismatch panel."""
import os
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/c1_pretraining_audits/mismatch'
cpus=sorted(os.sched_getaffinity(0));processes=[];logs=[]
for i in range(8):
    log=(OUT/'logs'/f'worker{i}.log').open('a');logs.append(log)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    processes.append(subprocess.Popen(['taskset','-c',','.join(map(str,cpus[i::8])),str(ROOT/'.venv-c1/bin/python'),
        str(ROOT/'scripts/audit_c1_prediction_mismatch.py'),'--worker',str(i)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT))
codes=[p.wait() for p in processes]
for log in logs:log.close()
print('worker exit codes',codes,flush=True)
if any(codes):raise SystemExit(1)
