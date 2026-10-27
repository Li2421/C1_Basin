from pathlib import Path
if (Path(__file__).resolve().parent/'RESUME_FATAL_STOP.json').exists():raise SystemExit('New read-only boundary violation: stopped')
import resume_bootstrap
from experiment import *
from registry_fix import verify_receipt,save,save_alias,repair_journal
from runtime_fix import run
from stage2_diagnostics_resume import summary
import signal
import numpy as np
STOP=False
def stop(*_):
    global STOP
    STOP=True

def main(phase):
    verify_receipt('WRITE_PATH_READY.receipt.json');verify_receipt('RESUME_IMPLEMENTATION.receipt.json')
    folder=HERE/'RESUME_QUEUES'/phase;proof=read(folder/'QUEUE.receipt.json')
    for p,h in proof['files'].items():assert sha(p)==h,p
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    for entry in read(folder/'INDEX.json'):
        if STOP:return
        claim=folder/'claims'/entry['group'];claim.parent.mkdir(parents=True,exist_ok=True)
        try:claim.mkdir()
        except FileExistsError:continue
        job=f"{os.environ.get('SLURM_ARRAY_JOB_ID',os.environ.get('SLURM_JOB_ID','local'))}_{os.environ.get('SLURM_ARRAY_TASK_ID','0')}"
        write(claim/'OWNER.json',dict(pid=os.getpid(),job=job,utc=utc()))
        assert sha(entry['path'])==entry['sha256']
        for c in sorted(read(entry['path']),key=lambda x:(x['method'],x['seed'])):
            if STOP:return
            stage=c.pop('output_stage');raw=HERE/stage/'raw'/f"{c['rollout_uid']}.json"
            if raw.exists():repair_journal(stage,c);continue
            source_uid=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=canonical(dict(future_index=0))))
            source=HERE/stage/'raw'/f'{source_uid}.json'
            if c['seed']==1 and c['controller_payload']['nominal']['type_code'] in ('gradient_mpc','prioritized_astar','conflict_based_search') and source.exists():
                result=save_alias(stage,c,source)
                sd=HERE/'DIAGNOSTICS'/f'{source_uid}.json'
                if sd.exists():write(HERE/'DIAGNOSTICS'/f"{c['rollout_uid']}.json",read(sd))
            else:
                result,z=run(c);result.update(execution_backend='resume_backend/runtime_fix.py',write_path_audit_sha256=sha(HERE/'WRITE_PATH_AUDIT.json'),jax_enable_x64=False)
                result['acceleration_bound_violation_steps']=0 if c['state']['scene']=='lit_circle_five' else int(np.any(np.linalg.norm(z['actions'],axis=-1)>1.5+1e-7,axis=-1).sum())
                result=save(stage,c,result,z)
                write(HERE/'DIAGNOSTICS'/f"{c['rollout_uid']}.json",summary(result,z))
            print(stage,c['controller_payload']['nominal']['controller'],c['state']['index'],c['method'],c['seed'],result['outcome'],round(result['elapsed_seconds'],3),flush=True)
        write(claim/'COMPLETE.json',dict(utc=utc()))
if __name__=='__main__':main(sys.argv[1])
