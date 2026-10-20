"""Submit only sealed missing requests, respecting the latest user resource cap."""
from pathlib import Path
import hashlib,json,subprocess,sys,datetime
from resource_guard import require

HERE=Path(__file__).resolve().parent
PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'


def submit(name,workers,phases,worker_offset=0):
    receipt=HERE/f'{name}_SUBMISSION.json'
    if receipt.exists():raise ValueError('Submission already exists; do not duplicate work')
    assert json.loads((HERE/'GENERATED_ACTION_TESTS.json').read_text())['passed']
    for phase in phases:
        frozen=json.loads((HERE/f'{phase}_EXECUTION_FROZEN.receipt.json').read_text())
        for path,digest in frozen['files'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        if phase.startswith('old_'):
            budget=json.loads((HERE/'OLD_TYPE_FINAL_PREFLIGHT_BUDGET.json').read_text())
            assert phase in budget['phases']
        if frozen.get('split')=='new_evaluation':
            assert (HERE/'NEW_EVAL_CHOICES.receipt.json').exists()
    before=require(cpu=workers)
    script=f'''#!/bin/bash
#SBATCH --job-name=crosstype2_{name}
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --array=0-{workers-1}%{workers}
#SBATCH --output={HERE}/{name}_%A_%a.log
set -euo pipefail
cd {HERE.parents[1]}
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
'''
    assert isinstance(worker_offset,int) and worker_offset>=0
    script+='\n'.join(f'{PYTHON} -B {HERE}/worker.py {phase} "$((SLURM_ARRAY_TASK_ID + {worker_offset}))"' for phase in phases)+'\n'
    path=HERE/f'{name}.sbatch';path.write_text(script)
    value=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),queue_before=before,phases=phases,
        workers=workers,worker_offset=worker_offset,script_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        execution_receipts={phase:hashlib.sha256((HERE/f'{phase}_EXECUTION_FROZEN.receipt.json').read_bytes()).hexdigest() for phase in phases})
    (HERE/f'{name}_SUBMISSION_PREFLIGHT.json').write_text(json.dumps(value,indent=2)+'\n')
    value['job_id']=subprocess.check_output(['sbatch','--parsable',str(path)],text=True).strip()
    receipt.write_text(json.dumps(value,indent=2)+'\n');print(name,value['job_id'],flush=True)


if __name__=='__main__':submit(sys.argv[1],int(sys.argv[2]),sys.argv[3:])
