from core import *
from shared_rollout_db.src.rollout_db import connect,transaction,canonical

def main(phase):
    prereg();rows=read(HERE/f'{phase}_CASES.json');counts=dict(total=len(rows),valid=0,unknown=0,missing=0,quarantined=0,raw=0,trace_verified=0);receipts=[]
    with connect(True) as db:
        for c in rows:
            r=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            if r is None:counts['missing']+=1;continue
            if r['conflict_quarantined']:counts['quarantined']+=1;continue
            counts['unknown' if r['numerical_failure'] else 'valid']+=1
            f=HERE/phase/'raw'/f"{c['rollout_uid']}.json";record=read(f);counts['raw']+=1;assert record['case']==c and record['seed_key']==r['seed_key']
            assert all(record[k]==r[k] for k in ('success','deadlock','timeout','collision','numerical_failure','episode_length','outcome'))
            assert sha(record['trace_path'])==record['trace_sha256'];counts['trace_verified']+=1
    result=dict(utc=utc(),status='PASS' if counts['missing']==counts['quarantined']==0 and counts['raw']==counts['trace_verified']==len(rows) else 'FAIL',phase=phase,**counts)
    write(HERE/f'{phase}_POSTFLIGHT.json',result);print(json.dumps(result));assert result['status']=='PASS'
    if phase=='main':
        with connect() as db,transaction(db):
            n=db.execute('SELECT count(*) FROM rollout WHERE experiment_uid=?',(EXP,)).fetchone()[0];db.execute('UPDATE experiment SET end_time=CURRENT_TIMESTAMP,new_rollout_count=?,reused_rollout_count=?,metadata_json=? WHERE experiment_uid=?',(n,0,canonical(dict(main=18688,gate=32,unknown_preserved=True,prospective_fresh64=True)),EXP))
if __name__=='__main__':main(sys.argv[1])
