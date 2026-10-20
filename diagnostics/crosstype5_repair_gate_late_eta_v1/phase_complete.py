"""Finish real journals while accounting separately for local inferred outcomes."""
from common import *
import fcntl,subprocess
from dynamic_reuse import aliases as collect_aliases

def main(stage):
    rows=read(HERE/f'{stage}_CASES.json');execute=read(HERE/f'{stage}_EXECUTION_CASES.json');aliases=collect_aliases(stage);records=[];absent=[]
    with rollout_db.connect(True) as db:
        for c in execute:
            if c['rollout_uid'] in aliases:continue
            path=HERE/stage/'raw'/f"{c['rollout_uid']}.json"
            if path.exists():
                r=read(path);assert r['case']==c and sha(r['trace_path'])==r['trace_sha256'];records.append(r)
            elif not db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone():absent.append(c['rollout_uid'])
    if absent:
        write(HERE/f'{stage}_INCOMPLETE.json',dict(utc=utc(),absent=absent,no_automatic_reruns=True));raise RuntimeError(f'{len(absent)} unfinished executions')
    if records:cache_writer.append_journal(records,EXP,f'{stage}_raw_recovery')
    with (HERE/'OFFICIAL_MERGER.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with (HERE/f'{stage}_MERGE.log').open('w') as stream:subprocess.run([sys.executable,'-B','-m','shared_rollout_db.src.merger','--experiment',EXP],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    write(HERE/f'{stage}_ALL_LOCAL_EQUIVALENCE.json',aliases)
    post=planner.preflight(HERE/f'{stage}_PLANNED_ROLLOUTS.json');write(HERE/f'{stage}_POSTFLIGHT.json',post);missing=[];unknown=[];actual=0
    with rollout_db.connect(True) as db:
        for c in rows:
            rid=c['rollout_uid'];r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(rid,)).fetchone()
            if r is not None:
                actual+=1
                if r['numerical_failure']:unknown.append(rid)
            elif rid in aliases:
                a=aliases[rid];assert a['written_to_shared_DB'] is False;assert sha(a['source_trace_path'])==a['source_trace_sha256'];assert db.execute('SELECT 1 FROM rollout WHERE rollout_uid=?',(a['source_rollout_uid'],)).fetchone()
                if a['outcome']['numerical_failure']:unknown.append(rid)
            else:missing.append(rid)
    write(HERE/f'{stage}_COMPLETENESS.json',dict(utc=utc(),requested_unique=len(rows),actual_DB=actual,actual_executed_this_stage=len(records),local_inferred=len(aliases),missing=missing,unknown=unknown,unknown_reruns=0,inferred_labels_written_to_DB=0,official_postflight_missing_inferred_expected=True))
    assert not missing
    seal(f'{stage}_COMPLETE.receipt.json',[HERE/f'{stage}_COMPLETENESS.json',HERE/f'{stage}_POSTFLIGHT.json',HERE/f'{stage}_MERGE.log',HERE/f'{stage}_ALL_LOCAL_EQUIVALENCE.json',HERE/f'{stage}_EXECUTION_FROZEN.receipt.json'],actual_and_inferred_separate=True)
if __name__=='__main__':main(sys.argv[1])
