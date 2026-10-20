"""Proceed only after reviewed freeze; choose before submitting any new rollout."""
from core import *
from submit_aux import submit as auxiliary,failure
from submit_rollouts import submit
from resource_guard import queue,cpu_capacity
import subprocess,time,fcntl,traceback

PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
def exists(name):return (HERE/name).exists()
def run(name,*args):subprocess.run([PYTHON,'-B',str(HERE/name),*map(str,args)],cwd=ROOT,check=True)
def ready():
    if not exists('new_evaluation_MISSING_CASES.json'):return False
    present={p.stem for p in (HERE/'new_evaluation/raw').glob('*.json')}
    return all(c['rollout_uid'] in present for c in read(HERE/'new_evaluation_MISSING_CASES.json'))


def stage():
    assert exists('FROZEN_PIPELINE.receipt.json'),'Root must inspect development and freeze first'
    if not exists('NEW_EVAL_CHOICES.receipt.json'):
        if not exists('new_evaluation_choices_SUBMISSION.json'):
            usage=queue()
            if usage['reserved_cpu_shards']<18 and usage['reserved_gpu_shards']<2:
                auxiliary('new_evaluation_choices',[f'{PYTHON} -B {HERE}/new_evaluation_choices.py'],gpu=True)
        if exists('new_evaluation_choices_SUBMISSION.json') and failure('new_evaluation_choices'):
            raise RuntimeError('Frozen inference failed; inspect retained log')
        return 'frozen_new_state_inference_running'
    if not exists('new_evaluation_EXECUTION_FROZEN.receipt.json'):
        run('design_execution.py','new_evaluation','new_evaluation','all')
    if not exists('new_evaluation_run_SUBMISSION.json'):
        free=cpu_capacity()[0]-queue()['reserved_cpu_shards']
        if free>0:submit('new_evaluation_run',min(cpu_capacity()[0],free),['new_evaluation'])
    if not exists('new_evaluation_COMPLETE.receipt.json'):
        if ready():run('finish_phase.py','new_evaluation')
        else:
            if exists('new_evaluation_run_SUBMISSION.json') and failure('new_evaluation_run'):
                raise RuntimeError('New-state rollout worker failed; never rerun unknown or unresolved claims')
            return 'collecting_all_fourteen_candidates_two_seeds'
    if not exists('NEW_EVAL_BUNDLE.receipt.json'):run('assemble_new_evaluation.py')
    if not exists('CONFIRMATORY_RESULTS.receipt.json'):
        if not exists('confirmatory_statistics_SUBMISSION.json') and queue()['reserved_cpu_shards']<18:
            auxiliary('confirmatory_statistics',[f'{PYTHON} -B {HERE}/confirmatory_statistics.py'])
        if exists('confirmatory_statistics_SUBMISSION.json') and failure('confirmatory_statistics'):
            raise RuntimeError('Final statistical analysis failed; preserve outcomes and fix analysis only')
        return 'confirmatory_statistics_running'
    return 'confirmatory_evidence_ready_for_final_review'


def main():
    with (HERE/'ADVANCE_CONFIRMATION.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(HERE/'ADVANCE_CONFIRMATION_PID.json',dict(pid=os.getpid(),utc=utc()))
        while True:
            try:status=stage()
            except Exception as e:
                write(HERE/'CONFIRMATION_ERROR.json',dict(utc=utc(),error_type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
                status='blocked_confirmation_error'
            write(HERE/'CONFIRMATION_LIVE_STATUS.json',dict(utc=utc(),stage=status,
                new_evaluation_rollouts_submitted=exists('new_evaluation_run_SUBMISSION.json')))
            if status in ('blocked_confirmation_error','confirmatory_evidence_ready_for_final_review'):return
            time.sleep(30)


if __name__=='__main__':main()
