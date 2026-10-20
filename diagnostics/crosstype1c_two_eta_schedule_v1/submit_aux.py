"""Slurm submissions, sealed queue checks and bounded temporary allocations."""
from resource_guard import *
import hashlib,shlex,math
ROOT=HERE.parent.parent
PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
def read(name):return json.loads((HERE/name).read_text())
def failure(name):
    path=HERE/f'{name}_SUBMISSION.json'
    if not path.exists():return None
    job=read(path.name)['job_id'].split(';')[0]
    r=subprocess.run(['scontrol','show','job',job,'--oneliner'],text=True,capture_output=True)
    if r.returncode:return ['Submission no longer available in controller; completion artifact required']
    states=[x.split('=',1)[1] for x in r.stdout.split() if x.startswith('JobState=')]
    bad=[s for s in states if s in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','DEADLINE')]
    return bad or None
def submit(name,commands,gpu=False,workers=1,temporary=False):
    receipt=HERE/f'{name}_SUBMISSION.json'
    if receipt.exists():return read(receipt.name)
    now=datetime.now(timezone.utc).timestamp()
    if temporary:
        if now>=DEADLINE-900:raise RuntimeError('Resources unavailable: temporary allocation drain window')
        minutes=max(1,math.floor((DEADLINE-now)/60)-2)
        duration=f'{minutes//60:02}:{minutes%60:02}:00'
    else:duration='24:00:00'
    before=require(cpu=workers,gpu=workers if gpu else 0)
    lines=['#!/bin/bash',f'#SBATCH --job-name=crosstype1c_{name}',f'#SBATCH --partition={"gpu" if gpu else "cpu"}','#SBATCH --cpus-per-task=1',f'#SBATCH --mem={12 if gpu else 4}G',f'#SBATCH --time={duration}',f'#SBATCH --output={HERE}/{name}_%A_%a.log']
    if workers>1:lines.append(f'#SBATCH --array=0-{workers-1}%{workers}')
    if gpu:lines.append('#SBATCH --gres=shard:1')
    if temporary:
        # sbatch parses the deadline in the same host timezone as this client.
        lines.append('#SBATCH --deadline='+datetime.fromtimestamp(DEADLINE).strftime('%Y-%m-%dT%H:%M:%S'))
    lines+=['set -euo pipefail','cd '+shlex.quote(str(ROOT)),f'export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS={"cuda" if gpu else "cpu"}', 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',f'export CROSSTYPE_TEMPORARY={int(temporary)}',*commands]
    path=HERE/f'{name}.sbatch';path.write_text('\n'.join(lines)+'\n')
    data=dict(utc=utc(),name=name,workers=workers,gpu=gpu,temporary=temporary,release_deadline='2026-10-19T09:00:00+08:00' if temporary else None,queue_before=before,script_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
    (HERE/f'{name}_SUBMISSION_PREFLIGHT.json').write_text(json.dumps(data,indent=2)+'\n')
    data['job_id']=subprocess.check_output(['sbatch','--parsable',str(path)],text=True).strip()
    receipt.write_text(json.dumps(data,indent=2)+'\n');print('Submitted',name,data['job_id'],flush=True)
    return data
