"""Resource-aware stage progression, preserving every completed/unknown claim."""
from pathlib import Path
import datetime,json,subprocess,sys,time,fcntl,traceback
from submit_aux import submit,failure,PYTHON
from resource_guard import queue,cpu_capacity,gpu_capacity

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def read(name):return json.loads((HERE/name).read_text())
def exists(name):return (HERE/name).exists()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def run(script,*args):
    with (HERE/'ADVANCE_COMMANDS.log').open('a') as stream:
        stream.write(f'\n{utc()} {script} {args}\n');stream.flush()
        subprocess.run([PYTHON,'-B',str(HERE/script),*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)


def phase_done(phase):
    if exists(f'{phase}_COMPLETE.receipt.json'):return True
    path=HERE/f'{phase}_MISSING_CASES.json'
    if not path.exists():return False
    cases=json.loads(path.read_text());folder=HERE/phase/'raw'
    if not folder.exists() or sum(1 for _ in folder.glob('*.json'))<len(cases):return False
    assert all((folder/f"{c['rollout_uid']}.json").exists() for c in cases)
    run('finish_phase.py',phase);return True


def submission_failed(name):
    if not exists(f'{name}_SUBMISSION.json'):return False
    # Slurm discards completed jobs quickly. Completed original workers can
    # legitimately exit while later-added workers finish their claimed fits.
    if name=='critic_training' and all(exists(f'CRITIC_TRAINING_WORKER_{i}_COMPLETE.json') for i in (0,1)):return False
    if name=='critic_training_extra' and all(exists(f'CRITIC_TRAINING_WORKER_{i}_COMPLETE.json') for i in range(2,8)):return False
    bad=failure(name)
    if bad:raise RuntimeError(f'{name}: {bad}; retained claims are not automatically retried')
    return False


def all_fits_complete(plan_name):
    return all((HERE/'critic_fits'/f"{t['mode']}_{t['initialization']}"/t['source_key']/'COMPLETE.receipt.json').exists() for t in read(plan_name)['tasks'])


def step():
    phases=('training_legacy','training_mixed','training_ppo','probe')
    completed=[phase_done(p) for p in phases]
    if not all(completed):return dict(stage='collecting_training_and_probes',completed=dict(zip(phases,completed)),queue=queue())
    if not exists('CONTROLLER_FEATURES.receipt.json'):
        # Both query array workers must have written every nonhistorical policy.
        p=read('probe_protocol.json')
        oldtypes={'Flow','gaussian','deterministic','mixture','diffusion','chunking','recurrent','cvae','ring_ippo'}
        missing=[n for n in p['controller_order'] if p['types'][n] not in oldtypes and not exists(f'probe/initial_actions/{n}.receipt.json')]
        if missing:raise RuntimeError('Initial nominal-query evidence missing: '+str(missing))
        run('assemble_features.py');run('assemble_training.py')
    if not exists('SELECTABLE_SPACE.receipt.json'):
        submission_failed('menu_headroom')
        submit('menu_headroom',[f'{PYTHON} -B {HERE}/headroom.py'])
        if not exists('CONTROLLER_INNER_TESTS.receipt.json'):submit('controller_inner_unit',[f'{PYTHON} -B {HERE}/test_controller_inner.py'],gpu=True)
        return dict(stage='model_free_headroom_and_source_isolation_tests')
    if not exists('CONTROLLER_INNER_TESTS.receipt.json'):
        submission_failed('controller_inner_unit');submit('controller_inner_unit',[f'{PYTHON} -B {HERE}/test_controller_inner.py'],gpu=True)
        return dict(stage='source_controller_inner_tests')
    if not exists('CRITIC_TRAINING_PLAN.receipt.json'):run('train_critic.py','plan')
    if not all_fits_complete('CRITIC_TRAINING_PLAN.json'):
        submission_failed('critic_training')
        submission_failed('critic_training_extra')
        submit('critic_training',[f'{PYTHON} -B {HERE}/train_critic.py run "$SLURM_ARRAY_TASK_ID"'],gpu=True,array=True)
        return dict(stage='nested_critic_training',tasks=len(read('CRITIC_TRAINING_PLAN.json')['tasks']))
    if not exists('DEVELOPMENT_CHOICES.receipt.json'):
        submission_failed('critic_selection')
        submit('critic_selection',[f'{PYTHON} -B {HERE}/select_mixed.py',f'{PYTHON} -B {HERE}/test_frozen_inference.py'],gpu=True)
        return dict(stage='source_nested_selection_and_serialization_gate')
    if not exists('FROZEN_INFERENCE_TESTS.receipt.json'):
        submission_failed('critic_selection');return dict(stage='serialization_gate')
    if not exists('ARCHITECTURE_FIXED.receipt.json'):
        submission_failed('development_statistics')
        submit('development_statistics',[f'{PYTHON} -B {HERE}/development_statistics.py'])
        return dict(stage='development_targets_and_architecture_fix')
    if not exists('LEAVE_ONE_FAMILY.receipt.json'):
        submission_failed('family_validation')
        submit('family_validation',[f'{PYTHON} -B {HERE}/family_validation.py'],gpu=True)
        return dict(stage='descriptive_leave_one_family')
    if not exists('SCALING_TRAINING_PLAN.receipt.json'):run('scaling.py','plan')
    if not all_fits_complete('SCALING_TRAINING_PLAN.json'):
        submission_failed('scaling_training')
        submit('scaling_training',[f'{PYTHON} -B {HERE}/scaling.py run "$SLURM_ARRAY_TASK_ID"'],gpu=True,array=gpu_capacity()[0])
        return dict(stage='source_diversity_scaling',tasks=len(read('SCALING_TRAINING_PLAN.json')['tasks']))
    if not exists('SCALING_RESULTS.receipt.json'):
        submission_failed('scaling_analysis')
        submit('scaling_analysis',[f'{PYTHON} -B {HERE}/scaling.py evaluate'],gpu=True)
        return dict(stage='two_axis_scaling_analysis')
    if not exists('FROZEN_PIPELINE.receipt.json'):run('freeze_pipeline.py')
    if not exists('NEW_EVAL_CHOICES.receipt.json'):
        submission_failed('new_eval_choices')
        submit('new_eval_choices',[f'{PYTHON} -B {HERE}/new_choices.py'],gpu=True)
        return dict(stage='frozen_new_state_choices')
    if not exists('new_evaluation_EXECUTION_FROZEN.receipt.json'):run('evaluation_design.py','design')
    if not phase_done('new_evaluation'):
        from submit_rollouts import submit as submit_cpu
        if not exists('new_evaluation_run_SUBMISSION.json'):submit_cpu('new_evaluation_run',cpu_capacity()[0],['new_evaluation'])
        return dict(stage='one_time_confirmatory_labels',planned=len(read('new_evaluation_CASES.json')))
    if not exists('locked_few_labels_EXECUTION_FROZEN.receipt.json'):run('evaluation_design.py','few_locked')
    if not phase_done('locked_few_labels'):
        from submit_rollouts import submit as submit_cpu
        if not exists('locked_few_label_run_SUBMISSION.json'):submit_cpu('locked_few_label_run',cpu_capacity()[0],['locked_few_labels'])
        return dict(stage='post_confirmation_locked_few_label_baseline')
    if not exists('NEW_EVAL_BUNDLE.receipt.json'):run('assemble_evaluation.py')
    if not exists('CONFIRMATORY_RESULTS.receipt.json'):
        submission_failed('confirmatory_statistics')
        submit('confirmatory_statistics',[f'{PYTHON} -B {HERE}/confirmatory_statistics.py'])
        return dict(stage='prespecified_endpoints_and_bootstrap')
    if not exists('DELIVERY_CHECKS.receipt.json'):
        submission_failed('delivery_artifacts')
        commands=[f'{PYTHON} -B {HERE}/{script}' for script in ('legacy_default_reference.py','descriptive_analysis.py','export_features.py','zoo_manifest.py','deliver_tables.py','plot_evidence.py','preview_figures.py','delivery_checks.py')]
        submit('delivery_artifacts',commands)
        return dict(stage='delivery_tables_figures_and_validation')
    return dict(stage='statistical_results_complete_delivery_pending')


def main():
    with (HERE/'ADVANCE.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                state=dict(utc=utc(),**step());(HERE/'ADVANCE_STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
                with (HERE/'ADVANCE_HISTORY.jsonl').open('a') as stream:stream.write(json.dumps(state)+'\n')
                print(json.dumps(state),flush=True)
                if state['stage']=='statistical_results_complete_delivery_pending':return
            except RuntimeError as exc:
                if str(exc).startswith('Resources unavailable:'):
                    print(utc(),str(exc),flush=True)
                else:
                    (HERE/'ADVANCE_FAILURE.json').write_text(json.dumps(dict(utc=utc(),error=str(exc),traceback=traceback.format_exc()),indent=2)+'\n');raise
            except Exception as exc:
                (HERE/'ADVANCE_FAILURE.json').write_text(json.dumps(dict(utc=utc(),error=str(exc),traceback=traceback.format_exc()),indent=2)+'\n');raise
            time.sleep(30)


if __name__=='__main__':main()
