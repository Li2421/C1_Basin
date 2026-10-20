"""One coordinator for post-capability journals, source fits and held-out analysis."""
from common import *
from pilot_training40 import rollout_stage
from corrective_menu40 import prepare,run_probe
from submit_aux40 import wait_submit
import time

def verify():
    receipt=read(HERE/'CORRECTIVE_STUDY40_IMPLEMENTATION_FROZEN.receipt.json')
    assert all(sha(p)==h for p,h in receipt['files'].items())
    assert read(HERE/'SELECTOR_INFERENCE40_UNIT_TEST.json')['passed']
    assert read(HERE/'SELECTOR_FEATURES40_UNIT_TEST.json')['passed']
    assert read(HERE/'PROBE_STATISTICS40_UNIT_TEST.json')['passed']

def main():
    while not (HERE/'AFTER_FULL_NAVIGATION40_COMPLETE.json').exists():
        if (HERE/'CORRECTIVE_STAGE40_BLOCKED.json').exists():return
        time.sleep(5)
    verify();families=read(HERE/'AF_MULTIAGENT_CAPABILITY_GATES.json')['eligible_families']
    if not families:
        write(HERE/'CORRECTIVE_STUDY40_BLOCKED.json',dict(utc=utc(),reason='No family retained at least two instances after the original multiagent capability gate',G_corrective_labels_opened=0));return
    for family in families:
        for phase in ['pilot_full','probe']:
            stage=f'{phase}40_{family}'
            if not (HERE/f'{stage}_CASES.json').exists():prepare(family,phase)
            if phase=='probe':run_probe(stage)
            else:rollout_stage(stage,'multiagent_worker40.py')
    if not all((HERE/f'{f}_SELECTOR_INPUTS40_FROZEN.receipt.json').exists() and (HERE/'selector_choices40'/f'{f}_fewshot.receipt.json').exists() for f in families):
        wait_submit('selector_inputs_and_pilot_choices40','selector_preparation_worker40.py',['inputs'],workers=len(families),memory_gib=2,freeze_paths=[HERE/'CORRECTIVE_STUDY40_IMPLEMENTATION_FROZEN.receipt.json'])
    assert all((HERE/f'{f}_SELECTOR_INPUTS40_FROZEN.receipt.json').exists() and (HERE/'selector_choices40'/f'{f}_fewshot.receipt.json').exists() for f in families)
    # Results are packaged as separate closed target files. Never inspect the evaluation
    # worker logs here; each model opens only its enumerated other-family source files.
    from selector_data40 import generated_labels
    for family in families:
        stage=f'evaluation40_{family}'
        if not (HERE/f'{stage}_CASES.json').exists():prepare(family,'evaluation')
        rollout_stage(stage,'multiagent_worker40.py')
        if not (HERE/f'{family}_SELECTOR_LABELS40_PACKAGED.receipt.json').exists():generated_labels(family)
    from selector_plan40 import prepare as prepare_models
    if not (HERE/'SELECTOR40_MODEL_PLAN_FROZEN.receipt.json').exists():prepare_models()
    plan=read(HERE/'SELECTOR40_MODEL_PLAN_FROZEN.json')
    if plan['models']:
        wait_submit('source_set_selector_fits40','selector_plan40.py',['worker'],workers=2,gpu=True,memory_gib=8,freeze_paths=[HERE/'SELECTOR40_MODEL_PLAN_FROZEN.receipt.json',HERE/'CORRECTIVE_STUDY40_IMPLEMENTATION_FROZEN.receipt.json'])
        for model in plan['models']:assert (HERE/'fits40'/model['model_id']/'MODEL_FROZEN.receipt.json').exists()
    if not (HERE/'ALL_AF_CHOICES40_FROZEN.receipt.json').exists():
        wait_submit('all_AF_selector_predictions40','selector_preparation_worker40.py',['predictions'],workers=1,memory_gib=2,freeze_paths=[HERE/'SELECTOR40_MODEL_PLAN_FROZEN.receipt.json'])
    assert (HERE/'ALL_AF_CHOICES40_FROZEN.receipt.json').exists()
    if not (HERE/'AF_ANALYSIS_FROZEN.receipt.json').exists():
        wait_submit('AF_corrective_analysis40','selector_preparation_worker40.py',['analysis_AF'],workers=1,memory_gib=2,freeze_paths=[HERE/'ALL_AF_CHOICES40_FROZEN.receipt.json'])
    assert (HERE/'AF_ANALYSIS_FROZEN.receipt.json').exists()
    # The original locked family was blocked before navigation training because its
    # prescribed old training cohort overlaps its fixed evaluation geometry.
    gap=read(HERE/'FULL_G_STATUS.json');assert gap['status']=='GEOMETRY_GAP'
    write(HERE/'LOCKED_G40_NOT_RUN.json',dict(utc=utc(),status='GAP',reason=gap['reason'],corrective_evaluation_count=0,corrective_labels_opened=0,final_selector_training='Not run: no qualified private locked-family policy, no locked test can be performed',source_G_rows=0,required_resolution=gap['required_resolution']))
    write(HERE/'CORRECTIVE_STUDY40_COMPLETE.json',dict(utc=utc(),analyzed_families=families,G_evaluations=0,G_gap_preserved=True,unknown_retries=0))

if __name__=='__main__':main()
