"""Official single merger; preserve unknowns and distinguish absent evidence."""
from common import *
import fcntl
import subprocess


def main(phase):
    p=read(HERE/f'{phase}_protocol.json');rows=read(HERE/f'{phase}_CASES.json')
    missing=read(HERE/f'{phase}_MISSING_CASES.json');probe=p['phase']=='probe'
    absent=[];records=[]
    with rollout_db.connect(True) as db:
        for c in missing:
            path=HERE/phase/'raw'/f"{c['rollout_uid']}.json"
            if path.exists():
                record=read(path)
                assert record['case']==c
                assert sha(record['trace_path'])==record['trace_sha256']
                records.append(record)
            elif probe or db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone() is None:
                absent.append(c['rollout_uid'])
    if absent:
        write(HERE/f'{phase}_INCOMPLETE.json',dict(utc=utc(),absent=absent,
              missing_is_not_unknown=True,reason='Unfinished claims or failed worker process; no automatic rerun'))
        raise RuntimeError(f'{len(absent)} unfinished requests')
    # Recover possible crash between writing a raw result and appending journal.
    # Duplicate journals are idempotently merged; no rollout is reexecuted.
    if records:
        journal=cache_writer.append_journal(records,EXP,f'{phase}_raw_recovery')
        assert journal.resolve().is_relative_to(ROOT/'shared_rollout_db/journals')
    with (HERE/'OFFICIAL_MERGER.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (HERE/f'{phase}_MERGE.log').open('w') as stream:
            subprocess.run([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],
                           cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    post=planner.preflight(HERE/f'{phase}_PLANNED_ROLLOUTS.json')
    write(HERE/f'{phase}_POSTFLIGHT.json',post)
    if probe:
        with rollout_db.connect(True) as db:
            fake=[]
            for c in rows:
                if db.execute('SELECT 1 FROM rollout WHERE state_uid=? AND controller_uid=?',
                              (c['state_uid'],c['controller_uid'])).fetchone() is not None:fake.append(c['rollout_uid'])
        assert not fake,'Unlabeled probes must not create task-success labels'
        write(HERE/f'{phase}_PROBE_POSTFLIGHT.json',dict(utc=utc(),planned=len(rows),verified_raw=len(records),
             complete=sum(r['complete'] for r in records),unknown_probe_count=sum(not r['complete'] for r in records),
             task_label_rows_created=0,official_planner_missing_expected=True,
             reason='Task rollout schema cannot encode outcome-free probe; verified source trajectories are the cache'))
    else:
        absent=[];unknown=[]
        with rollout_db.connect(True) as db:
            for c in rows:
                record=db.execute('SELECT numerical_failure FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
                if record is None:absent.append(c['rollout_uid'])
                elif record['numerical_failure']:unknown.append(c['rollout_uid'])
        write(HERE/f'{phase}_POSTFLIGHT_UNKNOWN.json',dict(utc=utc(),absent_rows=absent,unknown_preserved=unknown,
             unknown_reruns=0))
        assert not absent
    seal(f'{phase}_COMPLETE.receipt.json',[HERE/f'{phase}_protocol.json',HERE/f'{phase}_CASES.json',
         HERE/f'{phase}_POSTFLIGHT.json',HERE/f'{phase}_MERGE.log'],unknown_not_reexecuted=True)

if __name__=='__main__':main(sys.argv[1])
