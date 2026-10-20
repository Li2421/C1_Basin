"""Wait for frozen training; preflight before submitting any student rollout."""
from core import *
import subprocess,time

def main():
    while not (HERE/'TRAINING_FROZEN.receipt.json').exists():
        jobs=subprocess.check_output(['squeue','-h','-j','21367','-o','%T'],text=True).strip()
        if not jobs:raise RuntimeError('Offline job ended without a training freeze receipt; inspect logs')
        time.sleep(20)
    from phase_design import create
    create('pilot')
    # Frozen episode bookkeeping reads this filename; preserve identical pilot content.
    (HERE/'main_protocol.json').write_bytes((HERE/'pilot_protocol.json').read_bytes())
    assert read(HERE/'pilot_CACHE_PREFLIGHT.json')['summary']['ambiguous']==0
    seal('PILOT_SUBMISSION_READY.receipt.json',[HERE/'pilot_PREFLIGHT.receipt.json',HERE/'run.py',HERE/'episode_extended.py',HERE/'actions.py',HERE/'pipeline.py'],before_submission=True)
    out=subprocess.check_output(['sbatch','--array=0-17%18',str(HERE/'rollout.sbatch'),'pilot','18'],text=True)
    write(HERE/'PILOT_SUBMISSION.json',dict(utc=utc(),sbatch=out,preflight_sha256=sha(HERE/'pilot_PREFLIGHT.receipt.json')));print(out,flush=True)
if __name__=='__main__':main()
