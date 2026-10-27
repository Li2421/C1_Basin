from pathlib import Path
import json,subprocess,time,sys
from resume_resources import capacity
HERE=Path(__file__).resolve().parent

def main(phase):
    assert not (HERE/'RESUME_FATAL_STOP.json').exists()
    assert json.loads((HERE/'WRITE_PATH_AUDIT.json').read_text())['passed']
    proof=json.loads((HERE/'RESUME_QUEUES'/phase/'QUEUE.receipt.json').read_text())
    n,rows=capacity()
    if not n:raise RuntimeError('No currently authorized CPU capacity')
    job=subprocess.check_output(['sbatch','--parsable',f'--array=0-{n-1}',f'--export=ALL,RESUME_PHASE={phase}',str(HERE/'resume_worker.sbatch')],text=True).strip()
    p=HERE/'submissions'/f'{time.time_ns()}_resume_{job}.json';p.parent.mkdir(exist_ok=True)
    p.write_text(json.dumps(dict(phase=phase,job_id=job,workers=n,all_user_jobs_before=rows,queue_receipt=proof),indent=2)+'\n')
    print(job,n,flush=True)
if __name__=='__main__':main(sys.argv[1])
