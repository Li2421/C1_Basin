"""Persistent source-only development → freeze → once-only confirmation."""
from common import *
from submit_aux import submit,failure,PYTHON
from resource_guard import DEADLINE
import subprocess,fcntl,time,traceback
def exists(name):return (HERE/name).exists()
def run(script,*args):
    with (HERE/'ADVANCE_MODELS_COMMANDS.log').open('a') as f:subprocess.run([PYTHON,'-B',str(HERE/script),*args],cwd=HERE,stdout=f,stderr=subprocess.STDOUT,check=True)
def checked(name):
    bad=failure(name)
    if bad:raise RuntimeError(f'{name}: {bad}')
def gpu_stage(name,script,receipt):
    if exists(receipt):return True
    checked(name);submit(name,[f'{PYTHON} -B {HERE}/{script}'],gpu=True);return False
def cpu_stage(name,scripts,receipt):
    if exists(receipt):return True
    checked(name);submit(name,[f'{PYTHON} -B {HERE}/{s}' for s in scripts]);return False
def fits_complete():
    config=read(HERE/'CRITIC_TRAINING_PLAN.json')
    return sum((HERE/'critic_fits'/t['mode']/t['source_key']/'COMPLETE.receipt.json').exists() for t in config['tasks'])
def fresh_done():
    if exists('new_evaluation_COMPLETE.receipt.json'):return True
    missing=read(HERE/'new_evaluation_MISSING_CASES.json')
    with rollout_db.connect(True) as db:
        for c in missing:
            if (HERE/'new_evaluation/raw'/f"{c['rollout_uid']}.json").exists():continue
            if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone() is None:return False
    run('finish_phase.py','new_evaluation');return True
def step():
    if exists('ADVANCE_COLLECTION_FAILURE.json'):raise RuntimeError('Collection coordinator failure; inspect its preserved trace')
    if not exists('SELECTABLE_SPACE.receipt.json'):return dict(stage='awaiting_complete_pure_training_evidence')
    if not exists('GENERATOR_EXTENSION_DECISION.receipt.json'):
        report=read(HERE/'SELECTABLE_SPACE.json')['summaries']['combined']['type_equal'];write(HERE/'GENERATOR_EXTENSION_DECISION.json',dict(utc=utc(),enabled=False,optional_extension_not_claimed=True,proposal_only_scope='Frozen generator eight-proposal experiment',training_non_diagonal_oracle_pct=report['non_diagonal_oracle_pct'],human_step500_reference_pct=report['human_step500_reference_pct'],reason='Optional specialized schedule generator is not enabled in this version; preserve the shared10000 additional-label budget for user-authorized required comparators. Conclusions must explicitly remain conditional on the frozen eight-proposal generator. No claim that a separately trained generator would fail.'))
        seal('GENERATOR_EXTENSION_DECISION.receipt.json',[HERE/'GENERATOR_EXTENSION_DECISION.json',HERE/'SELECTABLE_SPACE.receipt.json'])
        with (HERE/'ITERATION_LOG.md').open('a') as f:f.write('\n完整开发前报告后：本版本未启用可选时间表生成器，保留追加标签预算供用户授权的必要对照。结论范围限于冻结生成器8提议，不将未做的专门生成器实验称为已失败。\n')
    if not (exists('CRITIC_TESTS.receipt.json') and exists('STATISTICS_TESTS.receipt.json')):
        checked('critic_and_statistics_tests');submit('critic_and_statistics_tests',[f'{PYTHON} -B {HERE}/test_critic.py',f'{PYTHON} -B {HERE}/test_statistics.py'],gpu=True);return dict(stage='source_isolation_and_sparse_statistics_tests')
    if not exists('CRITIC_TRAINING_PLAN.receipt.json'):run('train_critic.py','plan')
    count=fits_complete()
    if count<92:
        checked('critic_base');submit('critic_base',[f'{PYTHON} -B {HERE}/train_critic.py run "${{SLURM_ARRAY_TASK_ID:-0}}"'],gpu=True,workers=4)
        if time.time()<DEADLINE-900:submit('critic_temporary',[f'{PYTHON} -B {HERE}/train_critic.py run "$(( ${{SLURM_ARRAY_TASK_ID:-0}} + 4 ))"'],gpu=True,workers=4,temporary=True)
        if time.time()<DEADLINE-600:checked('critic_temporary')
        return dict(stage='source_nested_critic_training',completed_three_seed_fits=count,total=92)
    if not cpu_stage('source_only_selection',['select_models.py','development_statistics.py'],'DEVELOPMENT_RESULTS.receipt.json'):return dict(stage='source_nested_selection_and_development_statistics')
    if not gpu_stage('frozen_inference_tests','inference.py','FROZEN_INFERENCE_TESTS.receipt.json'):return dict(stage='new_and_original_cross_type1b_bitwise_inference_tests')
    if not exists('FROZEN_PIPELINE.receipt.json'):run('freeze_pipeline.py')
    if not gpu_stage('new_state_choices','new_choices.py','NEW_EVAL_CHOICES.receipt.json'):return dict(stage='freeze_all_choices_before_fresh_rollouts')
    if not exists('new_evaluation_EXECUTION_FROZEN.receipt.json'):run('new_design.py')
    if not fresh_done():
        checked('new_evaluation_base');submit('new_evaluation_base',[f'{PYTHON} -B {HERE}/worker.py new_evaluation "${{SLURM_ARRAY_TASK_ID:-0}}"'],workers=18)
        if time.time()<DEADLINE-900:submit('new_evaluation_temporary',[f'{PYTHON} -B {HERE}/worker.py new_evaluation "$(( ${{SLURM_ARRAY_TASK_ID:-0}} + 18 ))"'],workers=6,temporary=True)
        if time.time()<DEADLINE-600:checked('new_evaluation_temporary')
        return dict(stage='one_time_confirmatory_evaluation',completed=len(list((HERE/'new_evaluation/raw').glob('*.json'))),genuinely_missing=len(read(HERE/'new_evaluation_MISSING_CASES.json')))
    if not exists('NEW_EVAL_BUNDLE.receipt.json'):run('assemble_evaluation.py')
    if not cpu_stage('confirmatory_statistics',['confirmatory_statistics.py'],'CONFIRMATORY_RESULTS.receipt.json'):return dict(stage='prespecified_endpoints_and_bootstrap')
    if not cpu_stage('delivery_artifacts',['descriptive_analysis.py','deliver_tables.py','plot_evidence.py','preview_figures.py','delivery_checks.py'],'DELIVERY_CHECKS.receipt.json'):return dict(stage='tables_figures_and_delivery_validation')
    return dict(stage='statistics_complete_manual_visual_review_and_end_audit_pending')
def main():
    with (HERE/'ADVANCE_MODELS.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                state=dict(utc=utc(),**step());write(HERE/'ADVANCE_MODELS_STATUS.json',state)
                with (HERE/'ADVANCE_MODELS_HISTORY.jsonl').open('a') as stream:stream.write(json.dumps(state)+'\n')
                print(json.dumps(state),flush=True)
                if state['stage']=='statistics_complete_manual_visual_review_and_end_audit_pending':return
            except RuntimeError as exc:
                if str(exc).startswith('Resources unavailable:'):print(utc(),str(exc),flush=True)
                else:write(HERE/'ADVANCE_MODELS_FAILURE.json',dict(utc=utc(),error=str(exc),traceback=traceback.format_exc()));raise
            except Exception as exc:write(HERE/'ADVANCE_MODELS_FAILURE.json',dict(utc=utc(),error=str(exc),traceback=traceback.format_exc()));raise
            time.sleep(30)
if __name__=='__main__':main()
