"""No CPU reservation while waiting; finish cache ingestion and source fits."""
from __future__ import annotations
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent
PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
ROOT=HERE.parents[1]

def run(args):
    r=subprocess.run(args,cwd=ROOT,text=True,capture_output=True)
    if r.returncode:raise RuntimeError(str(args)+'\n'+r.stdout[-2000:]+'\n'+r.stderr[-2000:])
    return r.stdout


def wait(job):
    while run(['squeue','-h','-j',str(job)]).strip():time.sleep(15)


def main():
    wait(11021)
    assert datetime.now().astimezone()<datetime.fromisoformat('2026-10-09T08:15:00+08:00'),'Insufficient fitting/confirmation budget; do not fake completion'
    print('Source array ended; exact postflight begins',flush=True)
    print(run([PY,'-m','diagnostics.controller_q_diversity_v1.experiment','merge']),flush=True)
    print(run([PY,'-m','diagnostics.controller_q_diversity_v1.experiment','preflight','--stage','source_q1']),flush=True)
    pre=json.loads((HERE/'batches/source_q1/exact_identity_preflight.json').read_text())
    if pre['genuinely_missing']:
        raise RuntimeError(f"Source missing {pre['genuinely_missing']} attempts; inspect worker failures and resume only missing")
    cmd=f'cd {ROOT} && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu {PY} -u -m diagnostics.controller_q_diversity_v1.dataset'
    job=run(['sbatch','--parsable','--job-name=controller_q_assemble','--partition=cpu','--cpus-per-task=1','--mem=4G',
             '--time=00:20:00',f'--output={HERE}/logs/assemble_%j.log','--wrap',cmd]).strip().split(';')[0]
    print('Dataset assembly job',job,flush=True);wait(job)
    assert (HERE/'data/source128/manifest.json').exists(),job
    job=run(['sbatch','--parsable','--array=0-17%12',str(HERE/'fit.sbatch')]).strip().split(';')[0]
    (HERE/'source_fit_job.json').write_text(json.dumps(dict(job=job,submitted=datetime.now().astimezone().isoformat()),indent=2)+'\n')
    print('Source-only model fit array',job,flush=True);wait(job)
    done=list((HERE/'fits/source128').glob('*/seed*/complete.json'))
    assert len(done)==18,f'{len(done)}/18 fits complete; inspect before resuming'
    (HERE/'source_fits_ready.json').write_text(json.dumps(dict(complete=18,time=datetime.now().astimezone().isoformat(),final_labels_opened=False),indent=2)+'\n')
    print('SOURCE_FITS_READY: review source validation before freezing confirmation',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as ex:
        (HERE/'coordinator_error.json').write_text(json.dumps(dict(time=datetime.now().astimezone().isoformat(),error=repr(ex)),indent=2)+'\n')
        raise
