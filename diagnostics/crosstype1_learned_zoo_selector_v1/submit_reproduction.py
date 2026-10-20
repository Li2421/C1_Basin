"""One-shot submission with immediate queue check and sealed code receipt."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
from resource_guard import require

HERE=Path(__file__).resolve().parent
PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'

def main():
    target=HERE/'REPRODUCTION_SUBMISSION.json'
    assert not target.exists()
    receipt=json.loads((HERE/'REPRODUCTION_DESIGN.receipt.json').read_text())
    for path,digest in receipt['files'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
    before=require(cpu=2)
    script=HERE/'reproduction.sbatch'
    script.write_text(f'''#!/bin/bash
#SBATCH --job-name=crosstype1_reproduction
#SBATCH --partition=cpu
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output={HERE}/reproduction_%A_%a.log
set -euo pipefail
cd {HERE.parents[1]}
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec {PYTHON} -B {HERE}/reproduction.py run "$SLURM_ARRAY_TASK_ID"
''')
    frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')
            if p.name in ('reproduction.py','episode.py','controllers.py','historical_controllers.py',
                          'legacy_controllers.py','learned.py','gaussian.py','core.py','common.py','trigger.py','right_hand.py','progress.py')}
    row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),queue_before=before,files=frozen,
             script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),phase='32 mandatory historical reproduction checks')
    (HERE/'REPRODUCTION_SUBMISSION_PREFLIGHT.json').write_text(json.dumps(row,indent=2)+'\n')
    job=subprocess.check_output(['sbatch','--parsable',str(script)],text=True).strip()
    row['job_id']=job
    target.write_text(json.dumps(row,indent=2)+'\n')
    print(job,flush=True)

if __name__=='__main__':main()
