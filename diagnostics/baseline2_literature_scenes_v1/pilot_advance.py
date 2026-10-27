"""Called by the single scheduler after all original reference/L1 work."""
from experiment import *
from coordinator import command,done,exists,active,launch,queue,SCENES

def prepare(file,*args):
    if file=='pilot_design.py' or file=='learning_extended.py' and args[0] in ('plan','freeze'):command('db_merge.py')
    command(file,*args)

def advance():
    if not all(done('fixed_'+s) and done('whitebox_'+s) and done('nominal_'+s) for s in SCENES):return
    from learning_design import selected_states
    for scene in SCENES:
        gap=HERE/f'PILOT_GAP_{scene}.json'
        if gap.exists():continue
        try:selected_states(scene,'train');selected_states(scene,'evaluation')
        except ValueError as e:write(gap,dict(reason=str(e),scope='BC pilot requires original preregistered N8/N14 training/evaluation states; no replacement sampling'));continue
        dataset=HERE/f'PILOT_DEMONSTRATIONS_{scene}.receipt.json'
        if not dataset.exists():
            try:
                from pilot_training import data
                data(scene)
            except ValueError as e:write(gap,dict(reason=str(e),scope='cached expert demonstration gap'));continue
        trained=all((HERE/'pilot_policies'/scene/f'{f}_{seed}'/'COMPLETE.receipt.json').exists() for f in ('deterministic','mixture') for seed in (2026103301,2026103302))
        if not trained:
            stage='pilot_train_'+scene
            if not active(stage):
                previous=HERE/f'{stage}_SUBMISSION.json'
                if previous.exists() and not (HERE/f'{stage}_INFRA_RESUME_AUTHORIZED.json').exists():
                    write(HERE/f'{stage}_NEEDS_LOG_INSPECTION.json',dict(utc=utc(),reason='Training job ended without all checkpoints; inspect infrastructure versus numerical/implementation cause before retry',submission=read(previous)))
                else:
                    _,used=queue()
                    if used<16:command('submit.py',stage,'train_pilot.sbatch',1,1)
            continue
        cap='pilot_capability_'+scene
        if not exists(cap):prepare('pilot_design.py',scene,'capability')
        if not done(cap):launch(cap,4,'flow_pilot.sbatch');continue
        if not (HERE/f'{cap}_QUALIFICATION.json').exists():command('phase_reports.py','capability',cap)
        safety='pilot_safety_'+scene
        if not exists(safety):prepare('pilot_design.py',scene,'safety')
        if not done(safety):launch(safety,4,'flow_pilot.sbatch');continue
        if not (HERE/f'PILOT_BASELINE_USABILITY_{scene}.json').exists():command('analyze_pilot_baseline.py',scene)
        fixed='pilot_fixed_'+scene
        if not exists(fixed):prepare('pilot_design.py',scene,'fixed')
        if not done(fixed):launch(fixed,4,'flow_pilot.sbatch');continue
        nominal='pilot_nominal_'+scene
        if not exists(nominal):prepare('pilot_design.py',scene,'nominal')
        if not done(nominal):launch(nominal,2,'flow_pilot_nominal.sbatch')
        learninggap=HERE/f'PILOT_LEARNING_GAP_{scene}.json';labels='pilot_learning_labels_'+scene;tag=scene+'_with_bc'
        # Exact old requests may still be running. Wait for their journals
        # rather than creating a second stage that executes the same UID.
        base_labels='learning_labels_'+scene
        if exists(base_labels) and not done(base_labels):continue
        if not learninggap.exists() and not exists(labels):prepare('learning_extended.py','plan',scene)
        if learninggap.exists():continue
        if not done(labels):
            if not (HERE/labels/'STREAM_QUEUE.receipt.json').exists():command('stream_queue.py','build',labels)
            launch(labels,4,'flow_extended.sbatch');continue
        if not (HERE/f'LEARNING_BUNDLE_{tag}.receipt.json').exists():command('learning_extended.py','assemble',scene)
        if not (HERE/f'FROZEN_SCENE_CRITIC_{tag}.receipt.json').exists():
            _,used=queue()
            if used<16 and not active('critic_'+tag):command('submit.py','critic_'+tag,'train_critic_extended.sbatch',1,1)
            continue
        evaluation='learning_eval_'+tag
        base_evaluation='learning_eval_'+scene
        if exists(base_evaluation) and not done(base_evaluation):continue
        if not exists(evaluation):prepare('learning_extended.py','freeze',scene)
        if not done(evaluation):
            if not (HERE/evaluation/'STREAM_QUEUE.receipt.json').exists():command('stream_queue.py','build',evaluation)
            launch(evaluation,6,'flow_extended.sbatch')

if __name__=='__main__':advance()
