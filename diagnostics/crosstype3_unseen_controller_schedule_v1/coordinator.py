"""Resource-gated continuation; permanent claims and receipts prevent repeats."""
from common import *
from submit_aux import submit,PYTHON,failure
from resource_guard import queue,CPU_LIMIT,GPU_LIMIT
import subprocess,time,fcntl,traceback

def exists(name):return (HERE/name).exists()
def call(script,*args):
    command=[PYTHON,'-B',str(HERE/script),*map(str,args)]
    with (HERE/'COORDINATOR_ACTIONS.log').open('a') as log:
        log.write(utc()+' '+json.dumps(command)+'\n');log.flush();subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,env=dict(os.environ,JAX_PLATFORMS='cpu'))
def command(script,*args):return ' '.join([PYTHON,'-B',str(HERE/script),*map(str,args)])
def capacity(cpu,gpu=0):
    q=queue();return q['cpu_shards']+cpu<=CPU_LIMIT and q['gpu_shards']+gpu<=GPU_LIMIT
def workers_done(phase,count):return len(list((HERE/phase).glob('WORKER_*_COMPLETE.json')))==count
def check_jobs():
    q=queue();active={r['job'].split('_')[0] for r in q['rows']}
    for p in HERE.glob('*_SUBMISSION.json'):
        r=read(p);job=r['job_id'].split(';')[0]
        if job in active:continue
        bad=failure(r['name'])
        if bad and bad!=['Job unavailable; require completion artifact']:raise RuntimeError(f"Job {r['name']} failed: {bad}")

def advance():
    revision=read(HERE/'PROMPT_REVISION_LATEST.json')
    assert sha(ROOT/'docs/prompt_crosstype3_unseen_controller_two_eta_schedule_20261019.md')==revision['prompt_new_sha256'],'Prompt changed; require explicit revision review before further submissions'
    check_jobs()
    # Extra workers consume the same locked, permanently claimed UID queue;
    # they change throughput only, never candidate/seed/rollout budgets.
    if not exists('training_COMPLETE.receipt.json') and not workers_done('training',12) and not exists('training_spare_SUBMISSION.json') and capacity(1):submit('training_spare',[command('worker.py','training',12)])
    probe_finished=exists('probe_COMPLETE.receipt.json')
    if probe_finished and not exists('training_COMPLETE.receipt.json') and not exists('training_boost_SUBMISSION.json') and capacity(4):submit('training_boost',[command('worker.py','training','$((SLURM_ARRAY_TASK_ID+13))')],workers=4)
    initial_names=read(HERE/'probe_protocol.json')['controller_order'] if exists('probe_protocol.json') else []
    queries_finished=bool(initial_names) and all(exists('probe/initial_actions/'+n+'.receipt.json') for n in initial_names)
    if probe_finished and queries_finished and exists('training_boost_SUBMISSION.json') and not exists('training_COMPLETE.receipt.json') and not exists('training_final_spare_SUBMISSION.json') and capacity(1):submit('training_final_spare',[command('worker.py','training',17)])
    if not exists('capability_COMPLETE.receipt.json') and workers_done('capability',6):call('finish_phase.py','capability')
    if exists('capability_COMPLETE.receipt.json') and not exists('UNSEEN_CAPABILITY.receipt.json'):call('design.py','capability_collect')
    if exists('UNSEEN_CAPABILITY.receipt.json') and not exists('probe_EXECUTION_FROZEN.receipt.json'):call('design.py','probes')
    if exists('probe_EXECUTION_FROZEN.receipt.json') and not exists('probe_labels_SUBMISSION.json') and capacity(4):submit('probe_labels',[command('worker.py','probe','${SLURM_ARRAY_TASK_ID}')],workers=4)
    if exists('probe_EXECUTION_FROZEN.receipt.json') and not exists('initial_queries_SUBMISSION.json') and capacity(1):submit('initial_queries',[command('mechanisms.py','initial')])
    if not exists('CRITIC_TESTS.receipt.json') and not exists('critic_tests_SUBMISSION.json') and capacity(1,1):submit('critic_tests',[command('test_critic.py')],gpu=True)
    if exists('probe_labels_SUBMISSION.json') and workers_done('probe',4) and not exists('probe_COMPLETE.receipt.json'):call('finish_phase.py','probe')
    if exists('probe_COMPLETE.receipt.json') and not exists('CONTROLLER_FEATURES.receipt.json'):
        names=read(HERE/'probe_protocol.json')['controller_order']
        if all(exists('probe/initial_actions/'+n+'.receipt.json') for n in names):call('mechanisms.py','assemble')
    training_workers=12+(1 if exists('training_spare_SUBMISSION.json') else 0)+(4 if exists('training_boost_SUBMISSION.json') else 0)+(1 if exists('training_final_spare_SUBMISSION.json') else 0)
    if not exists('training_COMPLETE.receipt.json') and workers_done('training',training_workers):call('finish_phase.py','training')
    if exists('training_COMPLETE.receipt.json') and not exists('TRAINING_BUNDLE.receipt.json'):call('assemble.py')
    if exists('TRAINING_BUNDLE.receipt.json') and not exists('SELECTABLE_SPACE.receipt.json'):call('assemble.py','space')
    if exists('SELECTABLE_SPACE.receipt.json') and exists('CRITIC_TESTS.receipt.json') and not exists('CRITIC_TRAINING_PLAN.receipt.json'):call('train_critic.py','plan')
    if exists('CRITIC_TRAINING_PLAN.receipt.json') and not exists('critic_training_SUBMISSION.json') and capacity(2,2):submit('critic_training',[command('train_critic.py','run','${SLURM_ARRAY_TASK_ID}')],gpu=True,workers=2)
    fit_done=exists('CRITIC_TRAINING_PLAN.json') and all((HERE/'critic_fits'/t['mode']/t['source_key']/'COMPLETE.receipt.json').exists() for t in read(HERE/'CRITIC_TRAINING_PLAN.json')['tasks'])
    if fit_done and not exists('model_selection_SUBMISSION.json') and capacity(1):submit('model_selection',[command('select_models.py'),command('family_selection.py')])
    if exists('LEGACY_DEVELOPMENT_CHOICES.receipt.json') and not exists('development_statistics_SUBMISSION.json') and capacity(1):submit('development_statistics',[command('development.py')])
    if exists('DEVELOPMENT_CHOICES.receipt.json') and not exists('inference_gate_SUBMISSION.json') and capacity(1,1):submit('inference_gate',[command('inference_gate.py'),command('legacy_development.py')],gpu=True)
    ready=all(exists(n) for n in ('DEVELOPMENT_RESULTS.receipt.json','LEAVE_ONE_FAMILY_RESULTS.receipt.json','FROZEN_INFERENCE_TESTS.receipt.json','CONTROLLER_FEATURES.receipt.json'))
    if ready and not exists('FROZEN_PIPELINE.receipt.json'):call('freeze_pipeline.py')
    if exists('FROZEN_PIPELINE.receipt.json') and not exists('new_choices_SUBMISSION.json') and capacity(1,1):submit('new_choices',[command('new_choices.py')],gpu=True)
    if exists('NEW_EVAL_CHOICES.receipt.json') and not exists('NEW_EVAL_BUDGET.receipt.json'):call('new_design.py')
    if exists('NEW_EVAL_BUDGET.receipt.json') and not exists('new_evaluation_labels_SUBMISSION.json') and capacity(16):submit('new_evaluation_labels',[command('worker.py','new_evaluation','${SLURM_ARRAY_TASK_ID}')],workers=16)
    if exists('new_evaluation_labels_SUBMISSION.json') and workers_done('new_evaluation',16) and not exists('new_evaluation_COMPLETE.receipt.json'):call('finish_phase.py','new_evaluation')
    if exists('new_evaluation_COMPLETE.receipt.json') and not exists('NEW_EVAL_BUNDLE.receipt.json'):call('assemble_evaluation.py')
    if exists('NEW_EVAL_BUNDLE.receipt.json') and not exists('confirmatory_statistics_SUBMISSION.json') and capacity(1):submit('confirmatory_statistics',[command('confirmatory_statistics.py')])
    if exists('CONFIRMATORY_RESULTS.receipt.json') and not exists('POSTHOC_LABEL_DESIGN.receipt.json'):call('posthoc_design.py','design')
    if exists('POSTHOC_LABEL_DESIGN.receipt.json') and not exists('diagnostic_labels_SUBMISSION.json') and capacity(16):submit('diagnostic_labels',[command('worker.py','diagnostic','${SLURM_ARRAY_TASK_ID}')],workers=16)
    if exists('diagnostic_labels_SUBMISSION.json') and workers_done('diagnostic',16) and not exists('diagnostic_COMPLETE.receipt.json'):call('finish_phase.py','diagnostic')
    if exists('diagnostic_COMPLETE.receipt.json') and not exists('failure_attribution_SUBMISSION.json') and capacity(1):submit('failure_attribution',[command('failure_attribution.py')])
    if exists('FAILURE_ATTRIBUTION.receipt.json') and exists('KNOWN_NEW_STATE_RESULTS.receipt.json') and not exists('delivery_artifacts_SUBMISSION.json') and capacity(1):submit('delivery_artifacts',[command('plots.py'),command('close_audit.py'),command('report.py'),command('verify_delivery.py')])
    stage=next((name for name in ('FAILURE_ATTRIBUTION.receipt.json','CONFIRMATORY_RESULTS.receipt.json','FROZEN_PIPELINE.receipt.json','DEVELOPMENT_CHOICES.receipt.json','CRITIC_TRAINING_PLAN.receipt.json','TRAINING_BUNDLE.receipt.json','CONTROLLER_FEATURES.receipt.json','UNSEEN_CAPABILITY.receipt.json') if exists(name)),'labels_in_progress')
    write(HERE/'COORDINATOR_STATUS.json',dict(utc=utc(),stage=stage,queue=queue(),training_raw_count=len(list((HERE/'training/raw').glob('*.json'))),capability_raw_count=len(list((HERE/'capability/raw').glob('*.json'))),confirmation_or_posthoc_outcomes_not_printed=True))
    return exists('FINAL_VERIFICATION.receipt.json')

def main():
    with (HERE/'COORDINATOR.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                if advance():break
            except Exception:
                write(HERE/'COORDINATOR_ERROR.json',dict(utc=utc(),traceback=traceback.format_exc(),no_automatic_rollout_rerun=True));raise
            time.sleep(20)
if __name__=='__main__':main()
