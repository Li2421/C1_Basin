from resource_guard import *
import hashlib,shlex
PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
def submit(name,commands,gpu=False,workers=1):
    receipt=HERE/f'{name}_SUBMISSION.json'
    if receipt.exists():return json.loads(receipt.read_text())
    before=require(cpu=workers,gpu=workers if gpu else 0)
    lines=['#!/bin/bash',f'#SBATCH --job-name=crosstype3_{name}',f'#SBATCH --partition={"gpu" if gpu else "cpu"}','#SBATCH --cpus-per-task=1',f'#SBATCH --mem={12 if gpu else 6}G','#SBATCH --time=24:00:00',f'#SBATCH --output={HERE}/{name}_%A_%a.log']
    if workers>1:lines.append(f'#SBATCH --array=0-{workers-1}%{workers}')
    if gpu:lines.append('#SBATCH --gres=shard:1')
    lines+=['set -euo pipefail','cd '+shlex.quote(str(HERE.parent.parent)),f'export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS={"cuda" if gpu else "cpu"}','export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',f'export CUDA_CACHE_PATH={HERE}/cuda_cache',*commands]
    path=HERE/f'{name}.sbatch';path.write_text('\n'.join(lines)+'\n');r=dict(utc=utc(),name=name,workers=workers,gpu=gpu,queue_before=before,script_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
    (HERE/f'{name}_SUBMISSION_PREFLIGHT.json').write_text(json.dumps(r,indent=2)+'\n');r['job_id']=subprocess.check_output(['sbatch','--parsable',str(path)],text=True).strip();receipt.write_text(json.dumps(r,indent=2)+'\n');print('Submitted',name,r['job_id'],flush=True);return r
def failure(name):
    p=HERE/f'{name}_SUBMISSION.json'
    if not p.exists():return None
    job=json.loads(p.read_text())['job_id'].split(';')[0];r=subprocess.run(['scontrol','show','job',job,'--oneliner'],capture_output=True,text=True)
    if r.returncode:return ['Job unavailable; require completion artifact']
    states=[x.split('=',1)[1] for x in r.stdout.split() if x.startswith('JobState=')];bad=[s for s in states if s in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','DEADLINE')];return bad or None
