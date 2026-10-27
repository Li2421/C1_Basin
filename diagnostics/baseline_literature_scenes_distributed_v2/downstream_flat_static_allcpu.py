"""Follow the prescribed order after baseline comparisons; resumable receipts."""
from experiment import *
from coordinator_flat_static_allcpu import action,prepare,run_stages,stage_ready,jobs_by_stage
from resources_idle_cpu import snapshot
import time

def old_learning(scene):
    from source_inputs import load,index
    from registry import register
    stage='central_learning_eval_'+scene
    if (HERE/f'{stage}_PREFLIGHT.receipt.json').exists():return stage
    source=V1/f'learning_eval_{scene}_CASES.json';cases=load(source)
    for c in cases:
        c['state_provenance']=dict(sampler=index()[str(V1/'states.py')],state_index=c['state']['index'])
        c['safety_layer']='central'
    register(stage,cases)
    write(HERE/f'CENTRAL_LEARNING_REUSE_{scene}.json',dict(source=str(source),sha256=sha(source),requests=len(cases),new_training=False,new_rollouts=False))
    return stage

def analyze(stages,phase):
    from analysis_jobs_current import prepare as prepare_analysis
    run_stages([prepare_analysis(stage) for stage in stages],phase)

def train(scenes):
    from critic_jobs_released_allcpu import submit
    from status_released_allcpu import main as status
    while True:
        pending=[s for s in scenes if not (HERE/f'FROZEN_SCENE_CRITIC_{s}.receipt.json').exists()]
        if not pending:return
        state=snapshot();mapping=jobs_by_stage(state)
        for scene in pending:
            if 'critic_'+scene in mapping.values():continue
            current=snapshot()
            if current['free_cpu_shares'] and current['free_gpu_shards']:submit(scene)
        status('scene_critics_training');time.sleep(20)

def main():
    # Required phase 6: recompute existing central traces, without new actors.
    baselines=['central_baseline_'+s for s in NAMES]
    analyze(baselines,'central_saved_trace_diagnosis')
    fixed=[prepare(layer,'fixed',scene) for layer in ('distributed','central') for scene in NAMES]
    run_stages(fixed,'fixed_corrections')
    # Expose all baseline and fixed exact labels before the learning preflight.
    action('official_merge_before_labels','db_merge.py')
    learning=[]
    for scene in NAMES:
        if (HERE/f'LEARNING_GAP_{scene}.json').exists():continue
        stage='learning_labels_'+scene
        if not (HERE/f'{stage}_PREFLIGHT.receipt.json').exists():action('labels_plan_'+scene,'learning_design.py','plan',scene)
        if (HERE/f'LEARNING_GAP_{scene}.json').exists():continue
        if not stage_ready(stage):action('labels_queue_'+scene,'work_queue.py',stage)
        learning.append(scene)
    from scene_learning_scheduler_flat_static_allcpu import run as run_learning
    evaluations=run_learning(learning)
    for scene in ('lit_narrow_passage','lit_doorway'):old_learning(scene)
    diagnostics=[*['distributed_baseline_'+s for s in NAMES],*fixed,*evaluations,
                 *['central_learning_eval_'+s for s in ('lit_narrow_passage','lit_doorway')]]
    analyze(diagnostics,'all_main_saved_trace_diagnosis')
    action('official_merge_main','db_merge.py')
    if not (HERE/"MIGRATION_NUMBERS.receipt.json").exists():action('numeric_results','statistics.py')
    # Main numbers must be sealed before optional dynamics, variants or charts.
    while not (HERE/'OPTIONAL_DELIVERY_READY.receipt.json').exists():
        from status_released_allcpu import main as status
        status('main_numbers_sealed_awaiting_optional_delivery_implementation');time.sleep(30)
    action('optional_delivery','optional_delivery_flat_static_allcpu.py')

if __name__=='__main__':main()
