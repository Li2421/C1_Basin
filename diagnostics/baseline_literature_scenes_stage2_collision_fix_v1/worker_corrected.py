"""Bounded worker. Only preflight-missing requests; immutable raw and journal."""
from experiment import *
from registry_fix import save,save_alias,repair_journal
from runtime_fix import run
from stage2_diagnostics import summary
import numpy as np
import signal
import time
STOP=False
def stop(*_):
    global STOP
    STOP=True

def main(stage,phase):
    assert read(HERE/'STAGE2_FIX.receipt.json')['scenes'][stage.rsplit('_',1)[0]]['passed']
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
    # All workers claim actor/state groups; seeds of a deterministic actor stay
    # together, and the explicit alias carries its source identity and hash.
    cases=read(HERE/(f'{stage}_'+('PILOT_CASES.json' if phase=='pilot' else 'MISSING_CASES.json')))
    groups={}
    for c in cases:groups.setdefault((c['controller_payload']['nominal']['controller'],c['state']['N'],c['state']['index']),[]).append(c)
    for key,rows in groups.items():
        if STOP:return
        claim=HERE/stage/'claims'/phase/uid('group',key)
        claim.parent.mkdir(parents=True,exist_ok=True)
        try:claim.mkdir()
        except FileExistsError:continue
        write(claim/'OWNER.json',dict(pid=os.getpid(),job=os.environ.get('SLURM_JOB_ID'),utc=utc()))
        for c in sorted(rows,key=lambda x:(x['method'],x['seed'])):
            if STOP:return
            raw=HERE/stage/'raw'/f"{c['rollout_uid']}.json"
            if raw.exists():repair_journal(stage,c);continue
            source_uid=uid('roll',dict(state=c['state_uid'],eta=c['eta_uid'],controller=c['controller_uid'],seed=canonical(dict(future_index=0))))
            source=HERE/stage/'raw'/f'{source_uid}.json'
            if c['seed']==1 and c['controller_payload']['nominal']['type_code'] in ('gradient_mpc','prioritized_astar','conflict_based_search') and source.exists():
                r=save_alias(stage,c,source)
                sd=HERE/'DIAGNOSTICS'/f'{source_uid}.json'
                if sd.exists():write(HERE/'DIAGNOSTICS'/f"{c['rollout_uid']}.json",read(sd))
            else:
                result,z=run(c);result['jax_enable_x64']=False
                r=save(stage,c,result,z)
                write(HERE/'DIAGNOSTICS'/f"{c['rollout_uid']}.json",summary(r,z))
            print(c['state']['scene'],c['controller_payload']['nominal']['controller'],c['method'],c['seed'],r['outcome'],round(r['elapsed_seconds'],3),flush=True)
        write(claim/'COMPLETE.json',dict(utc=utc()))
if __name__=='__main__':main(*sys.argv[1:])
