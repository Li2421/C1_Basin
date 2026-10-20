"""Proceed through frozen locked probes, sealed choices, then one label opening."""
from pathlib import Path
import datetime,json,subprocess,time,fcntl,os,traceback
from resource_guard import queue
from submit_rollouts import submit
from advance_training import submit as submit_aux,failure

HERE=Path(__file__).resolve().parent
PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
def read(name):return json.loads((HERE/name).read_text())
def exists(name):return (HERE/name).exists()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(name,data):
    path=HERE/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def run(file,*args):subprocess.run([PYTHON,'-B',str(HERE/file),*map(str,args)],cwd=HERE.parents[1],check=True)
def complete(phase):return exists(f'{phase}_COMPLETE.receipt.json')
def finish(phase):
    if complete(phase) or not exists(f'{phase}_MISSING_CASES.json'):return
    missing=read(f'{phase}_MISSING_CASES.json')
    if all(exists(f"{phase}/raw/{c['rollout_uid']}.json") for c in missing):run('finish_phase.py',phase)


def stage():
    if not exists('FROZEN_PIPELINE.receipt.json'):
        if not exists('SCALE_CURVE_COMPLETE.receipt.json') or not exists('DEVELOPMENT_SELECTION_VALUE.receipt.json'):
            return 'waiting_scale_and_selection_value_before_freeze'
        selected=read('DEVELOPMENT_FINAL_SELECTION.json')['selected']
        if not exists('final_freeze_SUBMISSION.json') and queue()['reserved_cpu_shards']<18:
            submit_aux('final_freeze',[f'{PYTHON} -B {HERE}/freeze_pipeline.py {HERE}/DEVELOPMENT_BUNDLE.npz {selected}'])
        if exists('final_freeze_SUBMISSION.json') and failure('final_freeze'):
            raise ValueError('Final freeze job failed; no locked probes submitted')
        return 'final_model_freeze_running'
    assert len(read('FROZEN_PIPELINE.json')['feature_names'])==24,'This orchestrator implements the original mechanism feature schema'
    for phase in ('locked_probe','locked_evaluation','locked_tuning'):finish(phase)
    if not exists('locked_probe_EXECUTION_FROZEN.receipt.json'):run('design_execution.py','locked_probe','locked','probe')
    if not exists('locked_probe_run_SUBMISSION.json'):
        free=18-queue()['reserved_cpu_shards']
        if free>1:submit('locked_probe_run',min(17,free-1),['locked_probe'])
    if not exists('locked_initial_actions_SUBMISSION.json') and queue()['reserved_cpu_shards']<18:
        submit_aux('locked_initial_actions',[f'{PYTHON} -B {HERE}/initial_actions.py locked_probe 0 1'])
    proto=read('locked_probe_protocol.json')
    actions=all(exists(f'locked_probe/initial_actions/{n}.receipt.json') for n in proto['controller_order'])
    if not complete('locked_probe') or not actions:
        for name in ('locked_probe_run','locked_initial_actions'):
            if exists(f'{name}_SUBMISSION.json'):
                bad=failure(name)
                if bad:raise ValueError(f'{name}: {bad}; preserve raw records and inspect, do not rerun')
        return 'collecting_locked_outcome_free_probes'
    if not exists('LOCKED_FEATURES.receipt.json'):run('assemble.py','features','locked_probe','locked')
    if not exists('LOCKED_CHOICES.receipt.json'):run('choose_locked.py')
    for split in ('evaluation','tuning'):
        phase=f'locked_{split}'
        if not exists(f'{phase}_EXECUTION_FROZEN.receipt.json'):run('design_execution.py',phase,'locked',split)
    if not exists('locked_labels_run_SUBMISSION.json'):
        free=18-queue()['reserved_cpu_shards']
        if free>0:submit('locked_labels_run',min(18,free),['locked_evaluation','locked_tuning'])
    if not all(complete(p) for p in ('locked_evaluation','locked_tuning')):
        if exists('locked_labels_run_SUBMISSION.json'):
            bad=failure('locked_labels_run')
            if bad:raise ValueError(f'Locked labels job incomplete: {bad}; no automatic rerun')
        return 'collecting_locked_labels_after_sealed_choices'
    if not exists('LOCKED_BUNDLE.receipt.json'):run('assemble.py','labels','locked_evaluation','locked_tuning','locked')
    if not exists('locked_confirmatory_SUBMISSION.json') and queue()['reserved_cpu_shards']<18:
        submit_aux('locked_confirmatory',[f'{PYTHON} -B {HERE}/confirmatory_analysis.py locked_evaluation locked_tuning'])
    if exists('CONFIRMATORY_RESULTS.receipt.json'):return 'locked_confirmation_complete'
    return 'one_time_confirmatory_analysis_running'


def main():
    with (HERE/'ADVANCE_LOCKED.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write('ADVANCE_LOCKED_PID.json',dict(pid=os.getpid(),utc=utc()))
        while True:
            try:status=stage()
            except RuntimeError as e:status='waiting_resources_or_evidence: '+str(e)
            except Exception as e:
                write('LOCKED_SUPERVISOR_ERROR.json',dict(utc=utc(),error=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
                status='blocked_locked_supervisor_error'
            write('LOCKED_LIVE_STATUS.json',dict(utc=utc(),stage=status,
                frozen=exists('FROZEN_PIPELINE.receipt.json'),choices_sealed=exists('LOCKED_CHOICES.receipt.json')))
            if status.startswith('blocked_') or status=='locked_confirmation_complete':return
            time.sleep(20)


if __name__=='__main__':main()
