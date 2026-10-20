"""Advance sealed evidence collection only; never freeze models or open locked evaluation."""
from pathlib import Path
import datetime,json,subprocess,sys,time,fcntl,os,traceback
from resource_guard import queue
from submit_rollouts import submit
from advance_training import submit as submit_aux

HERE=Path(__file__).resolve().parent
PYTHON='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
def read(path):return json.loads(Path(path).read_text())
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(name,value):
    path=HERE/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def run(file,*args):subprocess.run([PYTHON,'-B',str(HERE/file),*map(str,args)],cwd=HERE.parents[1],check=True)
def exists(name):return (HERE/name).exists()
def complete(phase):return exists(f'{phase}_COMPLETE.receipt.json')

def submit_if_capacity(name,phases,maximum=3,worker_offset=0):
    if exists(f'{name}_SUBMISSION.json'):return
    free=18-queue()['reserved_cpu_shards']
    if free>0:submit(name,min(maximum,free),phases,worker_offset=worker_offset)

def finish_ready():
    phases=set()
    for receipt in HERE.glob('*_SUBMISSION.json'):
        phases.update(read(receipt).get('phases',[]))
    for phase in sorted(phases):
        if complete(phase):continue
        missing=read(HERE/f'{phase}_MISSING_CASES.json')
        present=set(f.stem for f in (HERE/phase/'raw').glob('*.json'))
        # finish_phase recovers verified raw records into an idempotent journal
        # before merging, including a worker crash before append_journal.
        if all(c['rollout_uid'] in present for c in missing):run('finish_phase.py',phase)

def stage():
    finish_ready()
    if complete('ppo_checkpoint') and not exists('PPO_CHECKPOINTS_SELECTED.json'):
        # No correction arms are involved in selecting these training checkpoints.
        run('ppo_checkpoint_selection.py','select');write('PPO_CHECKPOINTS_SELECTED.json',dict(utc=utc(),passed=True))
    if exists('PPO_CHECKPOINTS_SELECTED.json'):
        phase='locked_capability_initial'
        if not exists(f'{phase}_EXECUTION_FROZEN.receipt.json'):run('design_execution.py',phase,'locked','capability')
        submit_if_capacity('locked_cap_initial',[phase],2)
        if complete(phase) and not exists('LOCKED_CAPABILITY_INITIAL_COLLECTED.json'):
            run('capability.py',phase,'locked');write('LOCKED_CAPABILITY_INITIAL_COLLECTED.json',dict(utc=utc()))
    cfgs=read(HERE/'TRAINING_CONFIGS.json')
    dev=[c for c in cfgs if c['family'] in ('chunking','recurrent','cvae')]
    all_attempted=all(exists(f"policies/{c['name']}/COMPLETE.receipt.json") or exists(f"training_attempts/{c['name']}.failure.json") for c in dev)
    if all_attempted:
        phase='new_capability_all'
        if not exists(f'{phase}_EXECUTION_FROZEN.receipt.json'):run('design_execution.py',phase,'development','capability')
        submit_if_capacity('new_cap_all',[phase],3)
        if complete(phase) and not exists('DEVELOPMENT_CAPABILITY_FINAL_COLLECTED.json'):
            run('capability.py',phase,'development');write('DEVELOPMENT_CAPABILITY_FINAL_COLLECTED.json',dict(utc=utc()))
    if exists('DEVELOPMENT_CAPABILITY_FINAL_COLLECTED.json'):
        phases=[]
        for split in ('probe','evaluation','tuning'):
            phase=f'new_{split}';phases.append(phase)
            if not exists(f'{phase}_EXECUTION_FROZEN.receipt.json'):run('design_execution.py',phase,'development',split,'new_only')
        submit_if_capacity('new_evidence',phases,6)
        if complete('new_tuning') and not complete('old_tuning_v2'):
            submit_if_capacity('old_evidence_rebalanced',['old_evaluation','old_tuning_v2'],6,200)
        if complete('new_probe') and not exists('new_initial_actions_SUBMISSION.json') and queue()['reserved_cpu_shards']<18:
            submit_aux('new_initial_actions',[f'{PYTHON} -B {HERE}/initial_actions.py new_probe 0 1'])
        if complete('old_probe') and complete('new_probe'):
            p=read(HERE/'new_probe_protocol.json')
            old=read(HERE/'old_probe_protocol.json')
            ready=all(exists(f"{phase}/initial_actions/{n}.receipt.json") for phase,proto in [('old_probe',old),('new_probe',p)] for n in proto['controller_order'])
            if ready and not exists('development_probe_MERGED.receipt.json'):run('merge_evidence.py','development_probe','old_probe','new_probe')
            if ready and not exists('DEVELOPMENT_FEATURES.receipt.json'):run('assemble.py','features','development_probe','development')
            if ready and not exists('DEVELOPMENT_COMPARATOR_CHOICES.receipt.json'):run('seal_comparators.py','development')
        if exists('DEVELOPMENT_FEATURES.receipt.json') and all(complete(ph) for ph in ('old_evaluation','old_tuning_v2','new_evaluation','new_tuning')):
            for split,oldphase in [('evaluation','old_evaluation'),('tuning','old_tuning_v2')]:
                target=f'development_{split}'
                if not exists(f'{target}_MERGED.receipt.json'):run('merge_evidence.py',target,oldphase,f'new_{split}')
            if not exists('DEVELOPMENT_BUNDLE.receipt.json'):run('assemble.py','labels','development_evaluation','development_tuning','development')
            return 'development_bundle_ready_for_agent_iteration'
    return 'collecting_development_evidence_and_locked_capability_only'

def main():
    lock=(HERE/'ADVANCE_EVIDENCE.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write('ADVANCE_EVIDENCE_PID.json',dict(pid=os.getpid(),utc=utc()))
    while True:
        try:status=stage()
        except RuntimeError as exc:
            status='waiting_resources_or_evidence: '+str(exc)
        except Exception as exc:
            write('EVIDENCE_SUPERVISOR_ERROR.json',dict(utc=utc(),error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            status='blocked_evidence_supervisor_error'
        write('EVIDENCE_LIVE_STATUS.json',dict(utc=utc(),stage=status,locked_eval_or_tuning_labels_opened=False))
        if status.startswith('blocked_') or status=='development_bundle_ready_for_agent_iteration':return
        time.sleep(30)

if __name__=='__main__':main()
