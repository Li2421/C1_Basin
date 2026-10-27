"""Register only evaluation requests for a frozen training-selected menu action."""
from experiment import *
from registry import register,verify_receipt
from learning_design_revised_20261021 import make_case
from analysis_standard_supplement_20261022 import select,FIXED


def stage_name(scene):return 'learning_eval_revised_20261021_trainfixed_'+scene


def plan(scene):
    prefix=FIXED+'_'+scene;selected=select(scene)
    if selected is None:return
    verify_receipt(prefix+'.receipt.json')
    stage=stage_name(scene)
    if (HERE/f'{stage}_PREFLIGHT.receipt.json').exists():verify_receipt(f'{stage}_PREFLIGHT.receipt.json');return
    split=read(HERE/f'LEARNING_SPLIT_REVISED_20261021_{scene}.json')
    states=split['evaluation'];training={(s['N'],s['index']) for s in split['training']}
    assert not training.intersection((s['N'],s['index']) for s in states)
    ctls=read(HERE/'CONTROLLER_MANIFEST.json');ai=selected['selected_menu_index'];action=selected['selected_action']
    rows=[make_case(s,ctl,action,ai,seed,method='train_selected_fixed') for ctl in ctls for s in states for seed in (0,1)]
    assert len(rows)==(1536 if scene=='lit_circle_five' else 3072)
    evidence=dict(utc=utc(),scene=scene,selected_action=ai,selected_action_receipt=prefix+'.receipt.json',selection_uses_evaluation_outcomes=False,matching_scope='Same48 controllers, evaluation states and two seeds as learned evaluation',history_checked=['REVISED_DISTRIBUTED_FIXED_COMPLETE_20261021.receipt.json',f'learning_eval_revised_20261021_{scene}_PREFLIGHT.receipt.json',f'learning_labels_revised_20261021_{scene}_PREFLIGHT.receipt.json'],compatibility='Exact canonical execution identities via frozen revised_case_factory; reporting method alias does not change execution identity',remaining_gap='Only requests official preflight identifies genuinely missing; all compatible stored outcomes reused',planned_logical_requests=len(rows),no_retraining=True)
    write(HERE/f'{stage}_EVIDENCE_AUDIT.json',evidence)
    seal(f'{stage}_EVIDENCE_AUDIT.receipt.json',[Path(__file__),HERE/f'{stage}_EVIDENCE_AUDIT.json',HERE/(prefix+'.receipt.json')],before_preflight=True)
    register(stage,rows)


if __name__=='__main__':
    os.sched_setaffinity(0,{read(HERE/'LOCAL_CPU_GUARD_STATUS.json')['local_cpu']})
    for scene in (sys.argv[1:] or NAMES):plan(scene)
