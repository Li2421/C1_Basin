"""Advance gates and training only. Deliberately cannot open locked arm labels."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import time
import sys
import os
import fcntl
from resource_guard import require,queue

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'

def read(name):return json.loads((HERE/name).read_text())
def write(name,value):(HERE/name).write_text(json.dumps(value,indent=2)+'\n')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def failure(name):
    path=HERE/f'{name}_SUBMISSION.json'
    if not path.exists():return None
    job=read(path.name)['job_id'].split(';')[0]
    # This workstation has accounting storage disabled. Active/recent jobs are
    # available from slurmctld even when sacct cannot query the accounting DB.
    result=subprocess.run(['scontrol','show','job',job,'--oneliner'],text=True,capture_output=True)
    if result.returncode:
        active=subprocess.check_output(['squeue','-h','-r','-j',job,'-o','%T'],text=True).splitlines()
        if active:
            states=active
        else:
            return ['Job absent from controller and required completion evidence missing; inspect retained log']
    else:
        states=[token.split('=',1)[1] for token in result.stdout.split() if token.startswith('JobState=')]
    bad=[s.strip() for s in states if any(s.strip().startswith(v) for v in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL'))]
    return bad or None

def submit(name,commands,gpu=False,array=None):
    receipt=HERE/f'{name}_SUBMISSION.json'
    if receipt.exists():return
    count=2 if array else 1
    before=require(cpu=count,gpu=count if gpu else 0)
    path=HERE/f'{name}.sbatch'
    script=f'''#!/bin/bash
#SBATCH --job-name=crosstype1b_{name}
#SBATCH --partition={'gpu' if gpu else 'cpu'}
#SBATCH --cpus-per-task=1
#SBATCH --mem={12 if gpu else 4}G
#SBATCH --time=24:00:00
#SBATCH --output={HERE}/{name}_%A_%a.log
'''
    if gpu:script+='#SBATCH --gres=shard:1\n'
    if array:script+='#SBATCH --array=0-1%2\n'
    script+=f'''set -euo pipefail
cd {ROOT}
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS={'cuda' if gpu else 'cpu'}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
'''
    script+='\n'.join(commands)+'\n';path.write_text(script)
    files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}
    data=dict(utc=utc(),queue_before=before,files=files,script_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write(f'{name}_SUBMISSION_PREFLIGHT.json',data)
    data['job_id']=subprocess.check_output(['sbatch','--parsable',str(path)],text=True).strip()
    write(receipt.name,data)


