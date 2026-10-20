"""Full stage after both pilots pass; reuse their complete results unchanged."""
from common import *
from full_geometry40 import require_pilot,verify_full,FAMILIES,catalog_path
import full_geometry40 as geometry_stage
import full_expert40 as expert_stage
import full_collection40 as data_stage
import full_navigation40 as nav_stage
from collect39 import POLICIES,checkpoint
from pilot_training40 import rollout_stage
from submit_stage import completed
from memory_bounded_training40 import submit as submit_training
from navigation_gate40 import analyze as analyze_navigation
import time

def family_status(family,status,**values):write(HERE/f'FULL_{family}_STATUS.json',dict(utc=utc(),family=family,status=status,**values))

def run_family(family):
    require_pilot();verify_full()
    if (HERE/f'NAVIGATION_CAPABILITY_{family}.json').exists():return
    status_path=HERE/f'FULL_{family}_STATUS.json'
    if status_path.exists() and read(status_path)['status']=='GEOMETRY_GAP':return
    if not catalog_path(family).exists():
        try:
            if not catalog_path(family,False).exists():stage=geometry_stage.prepare(family)
            else:
                stages=[p.name[:-len('_CASES.json')] for p in HERE.glob(f'full_joint_{family}_*_CASES.json')]
                stage=max(stages,key=lambda s:int(s.rsplit('_',1)[1]))
            while stage:
                rollout_stage(stage,'joint_worker.py')
                stage=geometry_stage.accept(family,stage,int(stage.rsplit('_',1)[1])+1)
        except RuntimeError as error:
            if 'seed block exhausted' not in str(error).lower():raise
            family_status(family,'GEOMETRY_GAP',reason=str(error),all_unknown_records_preserved=True);return
    if not (HERE/f'FULL_{family}_NAVIGATION_STATES_FROZEN.receipt.json').exists():nav_stage.freeze_states(family)
    stage='full_expert_'+family
    if not (HERE/f'{stage}_CASES.json').exists():expert_stage.prepare(family)
    rollout_stage(stage,'expert_worker.py')
    p=HERE/f'FULL_{family}_EXPERT_GATE_RESULTS.json'
    expert=read(p)['families'][family] if p.exists() else expert_stage.analyze(family)
    if not expert['passed']:
        family_status(family,'EXPERT_GAP',reason='Fixed expert did not meet at least99% observed success and zero wall collisions',expert=expert);return
    for r in range(4):
        family_status(family,'TRAINING',round=r)
        for kind,seed in ([(None,None)] if r==0 else POLICIES):
            stage=f'data39_{family}_round_{r}'+(f'_{kind}_{seed}' if r else '')
            if not (HERE/f'{stage}_CASES.json').exists():data_stage.prepare(family,r,kind,seed)
            rollout_stage(stage,'collect39.py')
        tasks=[]
        for kind,seed in POLICIES:
            receipt=HERE/'datasets39'/f'{family}_{kind}_{seed}_round_{r}.npz.receipt.json'
            if not receipt.exists():data_stage.freeze_dataset(family,kind,seed,r)
            if not (checkpoint(family,kind,seed,r).parent/'COMPLETE.receipt.json').exists():tasks.append(dict(family=family,kind=kind,seed=seed,round_index=r))
        if tasks:
            name=f'full_train39_{family}_round_{r}';p=HERE/f'{name}_SUBMISSION.json';job=read(p)['job'] if p.exists() else None
            while job is None:
                job=submit_training(name,tasks)
                if job is None:time.sleep(5)
            while not completed(job):time.sleep(5)
        missing=[dict(kind=k,seed=s) for k,s in POLICIES if not (checkpoint(family,k,s,r).parent/'COMPLETE.receipt.json').exists()]
        if missing:
            family_status(family,'TRAINING_GAP',round=r,missing=missing,reason='Frozen training did not complete; preserve error evidence, no instance replacement');return
    stage='navigation40_'+family
    if not (HERE/f'{stage}_CASES.json').exists():nav_stage.prepare(family)
    rollout_stage(stage,'navigation_gate40.py');result=analyze_navigation(family)
    family_status(family,'NAVIGATION_COMPLETE',passed_instances=sum(x['passed'] for x in result),instances=result)

def main():
    require_pilot();verify_full()
    for family in FAMILIES:run_family(family)
    gates={}
    for family in 'ABCDEFG':
        p=HERE/f'NAVIGATION_CAPABILITY_{family}.json'
        gates[family]=read(p) if p.exists() else dict(family=family,status='NOT_RUN',family_passed=False,instances=[],reason=read(HERE/f'FULL_{family}_STATUS.json') if (HERE/f'FULL_{family}_STATUS.json').exists() else 'No completed navigation gate')
    eligible=[f for f in 'ABCDEF' if gates[f]['family_passed']]
    p=HERE/'ALL_ROUND40_NAVIGATION_GATES.json';write(p,dict(families=gates,eligible_first_six_families=eligible))
    seal('ALL_ROUND40_NAVIGATION_GATES.receipt.json',[p],before_correction_evaluation_labels=True,no_instance_replacement=True,no_retraining_after_navigation_gate=True)
    write(HERE/'FULL_NAVIGATION_PHASE_COMPLETE.json',dict(utc=utc(),eligible_first_six_families=eligible,correction_stage_eligible=len(eligible)>=4,correction_labels_opened=0,locked_corrective_test_not_opened=True,next_stage='Freeze remaining correction study before any target correction evaluation labels' if len(eligible)>=4 else 'Deliver navigation results and gaps; correction study blocked'))

if __name__=='__main__':main()
