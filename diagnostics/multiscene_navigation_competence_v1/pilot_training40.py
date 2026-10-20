"""Execute the already frozen pilot stages; a blocked family never blocks its peer."""
from common import *
from collect39 import POLICIES,prepare,checkpoint,freeze_dataset
from submit_cpu40 import submit
from submit_stage import completed,merge
from submit_training39 import submit as train_submit
import time

def rollout_stage(stage,script):
    done=HERE/f'{stage}_POSTFLIGHT.json'
    if done.exists():return
    submission=HERE/f'{stage}_SUBMISSION.json'
    job=read(submission)['job'] if submission.exists() else None
    while job is None:
        job=submit(stage,18,script=script)
        if job is None:time.sleep(5)
    while not completed(job):time.sleep(5)
    merge(stage)

def main():
    while not (HERE/'EXPERT_GATE_RESULTS.json').exists():time.sleep(5)
    results=read(HERE/'EXPERT_GATE_RESULTS.json')['families']
    for family in 'BF':
        if not results[family]['passed']:continue
        gate=HERE/f'NAVIGATION_CAPABILITY_{family}.json';gate_receipt=HERE/f'NAVIGATION_CAPABILITY_{family}.receipt.json'
        if gate.exists() and gate_receipt.exists():
            assert all(sha(path)==expected for path,expected in read(gate_receipt)['files'].items())
            continue
        try:
            for r in range(4):
                write(HERE/'ACTIVE_STAGE.json',dict(utc=utc(),phase='pilot_training',family=family,round=r))
                if r==0:
                    stage=f'data39_{family}_round_0'
                    if not (HERE/f'{stage}_CASES.json').exists():prepare(family,0)
                    rollout_stage(stage,'collect39.py')
                else:
                    for kind,seed in POLICIES:
                        stage=f'data39_{family}_round_{r}_{kind}_{seed}'
                        if not (HERE/f'{stage}_CASES.json').exists():prepare(family,r,kind,seed)
                        rollout_stage(stage,'collect39.py')
                tasks=[]
                for kind,seed in POLICIES:
                    manifest=HERE/'datasets39'/f'{family}_{kind}_{seed}_round_{r}.npz.receipt.json'
                    if not manifest.exists():freeze_dataset(family,kind,seed,r)
                    if not (checkpoint(family,kind,seed,r).parent/'COMPLETE.receipt.json').exists():tasks.append(dict(family=family,kind=kind,seed=seed,round_index=r))
                if tasks:
                    name=f'train39_{family}_round_{r}';p=HERE/f'{name}_SUBMISSION.json'
                    job=read(p)['job'] if p.exists() else None
                    while job is None:
                        job=train_submit(name,tasks)
                        if job is None:time.sleep(5)
                    while not completed(job):time.sleep(5)
                failed=[dict(kind=k,seed=s) for k,s in POLICIES if not (checkpoint(family,k,s,r).parent/'COMPLETE.receipt.json').exists()]
                if failed:
                    write(HERE/f'TRAINING_BLOCKED_{family}.json',dict(utc=utc(),family=family,round=r,failed=failed,reason='Frozen training task did not complete; no replacement or tuning'))
                    break
            else:
                from navigation_gate40 import prepare as nav_prepare,analyze
                stage='navigation40_'+family
                if not (HERE/f'{stage}_CASES.json').exists():nav_prepare(family)
                rollout_stage(stage,'navigation_gate40.py');analyze(family)
        except Exception as error:
            # Preserve the full error; source/cache failures require immediate whole-round stop.
            write(HERE/f'PILOT_EXECUTION_ERROR_{family}.json',dict(utc=utc(),type=type(error).__name__,message=str(error)))
            raise
    write(HERE/'PILOT_TRAINING_EXECUTION_COMPLETE.json',dict(utc=utc(),families=list(results),status='Review diagnostics and pilot decision before any expansion'))

if __name__=='__main__':main()
