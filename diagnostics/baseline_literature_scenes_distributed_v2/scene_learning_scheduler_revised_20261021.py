"""Independent scenes advance through the unchanged sealed learning functions."""
from experiment import *
from atomic_write import write
from registry import verify_receipt
from coordinator_correction_recovery_20261021 import action,stage_done,jobs_by_stage,stage_ready
from resources_corrections_20261021 import snapshot
from infrastructure_recovery_correction_20261021 import classify
from analysis_jobs_current import prepare as prepare_analysis
import fine_dispatch_correction_recovery_20261021 as fine_dispatch
import subprocess
import time


def review_critics(scenes,state,reviewed):
    live={j['job_id'] for j in state['jobs']}
    for path in (HERE/'submissions').glob('*.json'):
        row=read(path);job=row['job_id'];stage=row['stage']
        if not stage.startswith('critic_revised_20261021_') or job in live or job in reviewed:continue
        scene=stage.removeprefix('critic_revised_20261021_')
        if scene not in scenes:continue
        reviewed.add(job)
        if (HERE/f'FROZEN_SCENE_CRITIC_REVISED_20261021_{scene}.receipt.json').exists():continue
        review=classify(stage,job)
        oom=any(w in review['tail'].lower() for w in ('resource_exhausted','cuda_error_out_of_memory','out of memory'))
        if oom:
            review['infrastructure']=True;review['unhandled_implementation_error']=False
            write(HERE/'CRITIC_CONCURRENCY.json',dict(utc=utc(),maximum_simultaneous_jobs=1,
                reason='Observed GPU memory failure; unchanged parameters and seeds'))
        write(HERE/'critic_failure_reviews'/f'{job}.json',review)
        if not review['infrastructure']:
            write(HERE/f'LEARNING_GAP_REVISED_20261021_{scene}.json',dict(utc=utc(),required_by_protocol=True,
                reason='Critic implementation failed; retain labels and continue other scenes',
                review=str(HERE/'critic_failure_reviews'/f'{job}.json')))


def run(scenes):
    from critic_jobs_revised_20261021 import submit as submit_critic
    from diagnosis_backfill import submit as submit_backfill
    from coordinator_correction_recovery_20261021 import status
    verify_receipt('PIPELINED_SCHEDULING.receipt.json')
    labels=['learning_labels_revised_20261021_'+s for s in scenes]
    for stage in labels:fine_dispatch.prepare(stage)
    reviewed=set();diagnostics={};evaluations={}
    while True:
        state=snapshot();mapping=jobs_by_stage(state)
        review_critics(scenes,state,reviewed)
        active=[s for s in scenes if not (HERE/f'LEARNING_GAP_REVISED_20261021_{s}.json').exists()]
        for scene in active:
            stage='learning_labels_revised_20261021_'+scene
            fine_dispatch.recover(stage,set(mapping));fine_dispatch.sync_original(stage)
            if (HERE/stage/'NEEDS_IMPLEMENTATION_REVIEW.json').exists():
                write(HERE/f'LEARNING_GAP_REVISED_20261021_{scene}.json',dict(utc=utc(),required_by_protocol=True,
                    reason='Label worker implementation failure; other scenes continue',stage=stage));continue
            if stage_done(stage) and stage not in mapping.values():
                if not (HERE/f'LEARNING_BUNDLE_REVISED_20261021_{scene}.receipt.json').exists():
                    # Same official merger and label assembly, once this scene
                    # is complete. No other scene's labels enter its model.
                    action('scene_merge_before_model_'+scene,'db_merge.py')
                    action('scene_bundle_'+scene,'learning_design_revised_20261021.py','assemble',scene)
                if (HERE/f'FROZEN_SCENE_CRITIC_REVISED_20261021_{scene}.receipt.json').exists():
                    evaluation='learning_eval_revised_20261021_'+scene
                    if not (HERE/f'{evaluation}_PREFLIGHT.receipt.json').exists():
                        action('scene_freeze_choices_'+scene,'freeze_scene_choices_revised_20261021.py',scene)
                    if not stage_ready(evaluation):action('scene_evaluation_queue_'+scene,'work_queue.py',evaluation)
                    fine_dispatch.prepare(evaluation);evaluations[scene]=evaluation
        active=[s for s in active if not (HERE/f'LEARNING_GAP_REVISED_20261021_{s}.json').exists()]
        rollout_stages=['learning_labels_revised_20261021_'+s for s in active]+[evaluations[s] for s in active if s in evaluations]
        state=snapshot();mapping=jobs_by_stage(state)
        for stage in rollout_stages:
            fine_dispatch.recover(stage,set(mapping));fine_dispatch.sync_original(stage)
        pending=[s for s in rollout_stages if not stage_done(s)]
        ready_critics=[s for s in active if (HERE/f'LEARNING_BUNDLE_REVISED_20261021_{s}.receipt.json').exists()
            and not (HERE/f'FROZEN_SCENE_CRITIC_REVISED_20261021_{s}.receipt.json').exists()]
        needs_cpu=bool(any(fine_dispatch.available(s) for s in pending) or
            any('critic_revised_20261021_'+s not in mapping.values() for s in ready_critics))
        # Diagnostic work is preemptible at a completed diagnosis boundary.
        # Main rollout or GPU training always gets the freed slot first.
        if needs_cpu:
            for job,stage in mapping.items():
                if stage.startswith('analysis_'):
                    subprocess.run(['scancel','--batch','--signal=USR1',job],capture_output=True,text=True)
        cap=read(HERE/'CRITIC_CONCURRENCY.json')['maximum_simultaneous_jobs'] if (HERE/'CRITIC_CONCURRENCY.json').exists() else 2
        for scene in ready_critics:
            current=snapshot();live=jobs_by_stage(current)
            if 'critic_revised_20261021_'+scene in live.values():continue
            if sum(s.startswith('critic_revised_20261021_') for s in live.values())>=cap:break
            if current['free_cpu_shares'] and current['free_gpu_shards']:submit_critic(scene)
        current=snapshot();live=jobs_by_stage(current)
        available={s:fine_dispatch.available(s) for s in pending}
        counts={s:sum(stage==s for stage in live.values()) for s in pending}
        for _ in range(min(current['free_cpu_shares'],sum(available.values()))):
            target=min((s for s in pending if available[s]),key=lambda s:counts[s])
            fine_dispatch.submit(target);counts[target]+=1;available[target]-=1
        # Consume saved evidence during slow tails. Main numerical functions,
        # source traces, statistical weights and output destinations are fixed.
        current=snapshot();live=jobs_by_stage(current)
        if current['free_cpu_shares']:
            originals=[f'distributed_revised_fixed_{s}' for s in NAMES]
            originals += [f'central_revised_fixed_{s}' for s in NAMES if stage_done(f'central_revised_fixed_{s}')]
            originals += [evaluations[s] for s in active if s in evaluations and stage_done(evaluations[s])]
            for original in originals:
                if original not in diagnostics:diagnostics[original]=prepare_analysis(original)
            from infrastructure_recovery_correction_20261021 import recover
            choices=list(diagnostics.values())
            for stage in choices:recover(stage,set(live))
            available_diag={s:sum(not (HERE/s/'claims'/r['group']).exists()
                for r in read(HERE/s/'GROUP_INDEX.json')) for s in choices}
            counts_diag={s:sum(stage==s for stage in live.values()) for s in choices}
            for _ in range(min(current['free_cpu_shares'],sum(available_diag.values()))):
                target=min((s for s in choices if available_diag[s]),key=lambda s:counts_diag[s])
                submit_backfill(target);counts_diag[target]+=1;available_diag[target]-=1
        state=status('scene_learning_pipeline')
        write(HERE/'SCHEDULER_DUTY.json',dict(utc=utc(),poll_seconds=10,
            pending_rollout_stages=pending,ready_critics=ready_critics,
            scene_evaluation_stages=evaluations,free_cpu_shares=state['resources']['free_cpu_shares'],
            independent_diagnostics_backfill=True,case_claim_maximum=2,
            scientific_functions_unchanged=True))
        finished=all(s in evaluations and stage_done(evaluations[s]) for s in active)
        if finished:
            current=snapshot();live=jobs_by_stage(current)
            own_main=[j for j,s in live.items() if s in rollout_stages or s.startswith('critic_revised_20261021_')]
            if not own_main:
                for job,stage in live.items():
                    if stage.startswith('analysis_'):
                        subprocess.run(['scancel','--batch','--signal=USR1',job],capture_output=True,text=True)
                return [evaluations[s] for s in active]
        time.sleep(10)
