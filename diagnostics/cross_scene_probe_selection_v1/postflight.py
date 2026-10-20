from core import *
from shared_rollout_db.src.rollout_db import connect

def main():
    prereg();active=set(read(HERE/'PROBE_SELECTION_FREEZE.json')['active_variants']);cases=[c for c in read(HERE/'main_CASES.json') if c['variant'] in active];assert len(cases)==10080;raw_missing=[];trace_missing=[];db_missing=[];unknown=0;duplicate=len({c['rollout_uid'] for c in cases})!=len(cases)
    with connect(True) as db:
        for c in cases:
            path=HERE/'main/raw'/f"{c['rollout_uid']}.json"
            if not path.exists():raw_missing.append(c['rollout_uid']);continue
            r=read(path);assert r['case']==c and r['protocol_sha256']==sha(HERE/'protocol.json');unknown+=r['numerical_failure']
            if not Path(r['trace_path']).exists() or sha(r['trace_path'])!=r['trace_sha256']:trace_missing.append(c['rollout_uid'])
            row=db.execute('SELECT success,numerical_failure,conflict_quarantined FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            if row is None or row['success']!=r['success'] or row['numerical_failure']!=r['numerical_failure'] or row['conflict_quarantined']:db_missing.append(c['rollout_uid'])
    passed=not(raw_missing or trace_missing or db_missing or duplicate);write(HERE/'main_POSTFLIGHT.json',dict(utc=utc(),status='PASS' if passed else 'FAIL',planned=len(cases),raw_missing=raw_missing,trace_missing=trace_missing,db_missing=db_missing,duplicate_uids=duplicate,unknown=unknown,unknown_never_rerun=True,probe_freeze_receipt_sha256=sha(HERE/'PROBE_SELECTION_FREEZE.receipt.json')));assert passed,'Main postflight failed';print('Main postflight PASS',len(cases),'unknown',unknown)
if __name__=='__main__':main()
