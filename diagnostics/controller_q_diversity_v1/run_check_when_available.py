"""Admission outside Slurm: does not reserve a CPU while waiting."""
import subprocess
import time
from pathlib import Path

def call(args):return subprocess.run(args,text=True,capture_output=True)

array='11021'
call(['scontrol','update',f'JobId={array}','ArrayTaskThrottle=11'])
try:
    while True:
        out=call(['squeue','-h','-t','R','-o','%j|%C']).stdout
        used=sum(int(line.split('|')[1]) for line in out.splitlines() if line.startswith('controller_q_'))
        if used<=11:break
        time.sleep(5)
    cmd='cd /home/zhihan/research/Basin_C1 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -u -m diagnostics.controller_q_diversity_v1.check_analysis'
    r=call(['sbatch','--parsable','--job-name=controller_q_checks','--partition=cpu','--cpus-per-task=1','--mem=4G','--time=00:10:00','--output=diagnostics/controller_q_diversity_v1/logs/checks_%j.log','--wrap',cmd])
    assert r.returncode==0,r.stderr
    job=r.stdout.strip().split(';')[0];print('Submitted',job,flush=True)
    while call(['squeue','-h','-j',job]).stdout.strip():time.sleep(5)
    assert Path('diagnostics/controller_q_diversity_v1/analysis_contract_checks.json').exists(),job
    print('Checks completed',flush=True)
finally:
    call(['scontrol','update',f'JobId={array}','ArrayTaskThrottle=12'])
