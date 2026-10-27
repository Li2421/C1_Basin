from pathlib import Path
if (Path(__file__).resolve().parent/'FATAL_STOP.json').exists():raise SystemExit('Read-only boundary fatal stop')
import bootstrap
from experiment import *
from registry import verify_receipt,save,save_alias,repair_journal
from swap_runtime import run as new_run
from runtime_correction_pa_memo_query_disk_candidate_spatial_mask_20261022 import run as old_run
from swap_diagnostics import summary
import signal
import numpy as np
STOP=False
def stop(*_):
    global STOP
    STOP=True

def main(stage):
    verify_receipt('WRITE_PATH_READY.receipt.json');verify_receipt('IMPLEMENTATION_FROZEN.receipt.json')
    folder=HERE/'QUEUES'/stage;proof=read(folder/'QUEUE.receipt.json')
    for p,h in proof['files'].items():assert sha(p)==h,p
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    for entry in read(folder/'INDEX.json'):
        if STOP:return
        claim=folder/'claims'/entry['group'];claim.parent.mkdir(parents=True,exist_ok=True)
        try:claim.mkdir()
        except FileExistsError:continue
        job=f"{os.environ.get('SLURM_ARRAY_JOB_ID',os.environ.get('SLURM_JOB_ID','local'))}_{os.environ.get('SLURM_ARRAY_TASK_ID','0')}";write(claim/'OWNER.json',dict(pid=os.getpid(),job=job,utc=utc()))
        assert sha(entry['path'])==entry['sha256']
        for c in sorted(read(entry['path']),key=lambda x:(x['requested_action'],x['seed'])):
            if STOP:return
            raw=HERE/stage/'raw'/f"{c['rollout_uid']}.json"
            if raw.exists():repair_journal(stage,c);continue
            source_uid=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=canonical(dict(future_index=0))));source=HERE/stage/'raw'/f'{source_uid}.json';isnew=c['state']['scene'].startswith('lit_swap8')
            if c['seed']==1 and c['controller_payload']['nominal']['type_code'] in ('gradient_mpc','prioritized_astar','conflict_based_search') and source.exists():
                result=save_alias(stage,c,source);sd=HERE/'DIAGNOSTICS'/f'{source_uid}.json'
                if sd.exists():write(HERE/'DIAGNOSTICS'/f"{c['rollout_uid']}.json",read(sd))
            else:
                result,z=(new_run if isnew else old_run)(c);result.update(execution_backend='swap_runtime' if isnew else 'private_v2_equivalent',write_path_audit_sha256=sha(HERE/'WRITE_PATH_AUDIT.json'))
                result=save(stage,c,result,z)
                if isnew and stage!='capability':write(HERE/'DIAGNOSTICS'/f"{c['rollout_uid']}.json",summary(result,z))
            print(stage,c['state']['scene'],c['controller_payload']['nominal']['controller'],c['state']['index'],c['state'].get('agent'),c['requested_action'],c['seed'],result['outcome'],round(result['elapsed_seconds'],3),flush=True)
        write(claim/'COMPLETE.json',dict(utc=utc()))
if __name__=='__main__':main(sys.argv[1])
