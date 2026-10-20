"""Persistent collection, mandatory pilot report, then remaining types."""
from common import *
from submit_aux import submit,failure,PYTHON
from resource_guard import DEADLINE,queue,limits
import subprocess,fcntl,time
def run(script,*args):
    with (HERE/'ADVANCE_COLLECTION_COMMANDS.log').open('a') as out:subprocess.run([PYTHON,'-B',str(HERE/script),*args],cwd=HERE,stdout=out,stderr=subprocess.STDOUT,check=True)
def done(phase):
    if (HERE/f'{phase}_COMPLETE.receipt.json').exists():return True
    missing=read(HERE/f'{phase}_MISSING_CASES.json')
    with rollout_db.connect(True) as db:
        for c in missing:
            if (HERE/phase/'raw'/f"{c['rollout_uid']}.json").exists():continue
            if db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone() is None:return False
    run('finish_phase.py',phase);return True
def start(phase):
    if (HERE/f'{phase}_base_SUBMISSION.json').exists():return
    count=18
    submit(phase+'_base',[f'{PYTHON} -B {HERE}/worker.py {phase} "${{SLURM_ARRAY_TASK_ID:-0}}"'],workers=count)
    if time.time()<DEADLINE-900:
        submit(phase+'_temporary',[f'{PYTHON} -B {HERE}/worker.py {phase} "$(( ${{SLURM_ARRAY_TASK_ID:-0}} + 18 ))"'],workers=6,temporary=True)
def step():
    for phase in ('pilot','remaining'):
        if phase=='remaining' and not (HERE/'PILOT_REPORT.receipt.json').exists():
            run('assemble.py','pilot');run('headroom.py','pilot')
        if not (HERE/f'{phase}_EXECUTION_FROZEN.receipt.json').exists():run('design.py',phase)
        start(phase)
        if not done(phase):
            for suffix in ('base','temporary'):
                if (HERE/f'{phase}_{suffix}_SUBMISSION.json').exists():
                    bad=failure(phase+'_'+suffix)
                    if bad and not (suffix=='temporary' and time.time()>=DEADLINE-600):raise RuntimeError(f'{phase}_{suffix}: {bad}')
            return dict(stage=phase+'_collection',completed=len(list((HERE/phase/'raw').glob('*.json'))),genuinely_missing=len(read(HERE/f'{phase}_MISSING_CASES.json')))
    if not (HERE/'TRAINING_BUNDLE.receipt.json').exists():run('assemble.py','all')
    if not (HERE/'SELECTABLE_SPACE.receipt.json').exists():run('headroom.py','all')
    return dict(stage='training_evidence_ready_for_critic')
def main():
    with (HERE/'ADVANCE_COLLECTION.lock').open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                state=dict(utc=utc(),**step());write(HERE/'ADVANCE_COLLECTION_STATUS.json',state)
                with (HERE/'ADVANCE_COLLECTION_HISTORY.jsonl').open('a') as stream:stream.write(json.dumps(state)+'\n')
                print(json.dumps(state),flush=True)
                if state['stage']=='training_evidence_ready_for_critic':return
            except RuntimeError as exc:
                if str(exc).startswith('Resources unavailable:'):print(utc(),str(exc),flush=True)
                else:
                    import traceback
                    write(HERE/'ADVANCE_COLLECTION_FAILURE.json',dict(utc=utc(),error=str(exc),traceback=traceback.format_exc()));raise
            except Exception as exc:
                import traceback
                write(HERE/'ADVANCE_COLLECTION_FAILURE.json',dict(utc=utc(),error=str(exc),traceback=traceback.format_exc()));raise
            time.sleep(30)
if __name__=='__main__':main()
