"""Pilot report before remaining types; never submit new-evaluation states."""
from pathlib import Path
import json,datetime,subprocess,time,fcntl,os,traceback
from resource_guard import queue
from submit_rollouts import submit
from submit_aux import submit as auxiliary,failure

HERE=Path(__file__).resolve().parent
PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
def exists(name):return (HERE/name).exists()
def read(name):return json.loads((HERE/name).read_text())
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(name,data):
    path=HERE/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def run(name,*args):subprocess.run([PYTHON,'-B',str(HERE/name),*map(str,args)],cwd=HERE.parents[1],check=True)
def complete(phase):return exists(f'{phase}_COMPLETE.receipt.json')
def ready(phase):
    if not exists(f'{phase}_MISSING_CASES.json'):return False
    present={p.stem for p in (HERE/phase/'raw').glob('*.json')}
    return all(c['rollout_uid'] in present for c in read(f'{phase}_MISSING_CASES.json'))


def stage():
    if not complete('pilot'):
        if ready('pilot'):run('finish_phase.py','pilot')
        else:return 'collecting_pilot'
    if not exists('PILOT_TRAINING_BUNDLE.receipt.json'):run('assemble_training.py','pilot')
    if not exists('PILOT_REPORT.receipt.json'):
        if not exists('pilot_headroom_SUBMISSION.json') and queue()['reserved_cpu_shards']<18:
            auxiliary('pilot_headroom',[f'{PYTHON} -B {HERE}/headroom.py pilot'])
        if exists('pilot_headroom_SUBMISSION.json') and failure('pilot_headroom'):raise ValueError('Pilot report job failed; inspect preserved output')
        return 'pilot_model_free_space_report_running'
    if not exists('remaining_EXECUTION_FROZEN.receipt.json'):run('design_execution.py','remaining','training','remaining')
    if not exists('remaining_run_SUBMISSION.json'):
        free=18-queue()['reserved_cpu_shards']
        if free>0:submit('remaining_run',min(16,free),['remaining'])
    if not complete('remaining'):
        if ready('remaining'):run('finish_phase.py','remaining')
        else:return 'collecting_remaining_development_types'
    if not exists('TRAINING_BUNDLE.receipt.json'):run('assemble_training.py','all')
    if not exists('SELECTABLE_SPACE.receipt.json'):
        if not exists('full_headroom_SUBMISSION.json') and queue()['reserved_cpu_shards']<18:
            auxiliary('full_headroom',[f'{PYTHON} -B {HERE}/headroom.py all'])
        if exists('full_headroom_SUBMISSION.json') and failure('full_headroom'):raise ValueError('Selectable-space job failed; inspect preserved output')
        return 'full_model_free_space_report_running'
    return 'training_evidence_ready_for_critic_development'


def main():
    with (HERE/'ADVANCE_COLLECTION.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write('ADVANCE_COLLECTION_PID.json',dict(pid=os.getpid(),utc=utc()))
        while True:
            try:status=stage()
            except RuntimeError as e:status='waiting_resources_or_evidence: '+str(e)
            except Exception as e:
                write('COLLECTION_ERROR.json',dict(utc=utc(),error_type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
                status='blocked_collection_error'
            write('LIVE_STATUS.json',dict(utc=utc(),stage=status,new_evaluation_rollouts_submitted=False))
            if status.startswith('blocked_') or status=='training_evidence_ready_for_critic_development':return
            time.sleep(30)


if __name__=='__main__':main()
