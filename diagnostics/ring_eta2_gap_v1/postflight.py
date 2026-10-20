"""Audit exact DB and raw presence for numerical failures treated as unknown."""
from core import *
from shared_rollout_db.src.rollout_db import connect

def main(phase):
    assert phase in ('train','test')
    post=read(HERE/f'{phase}_POSTFLIGHT.json')
    assert post['summary']['ambiguous']==0
    cases={}
    for c in read(HERE/f'{phase}_MISSING_CASES.json'):
        key=(c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key'])
        assert key not in cases;cases[key]=c
    missing=[]
    with connect(True) as con:
        for d in post['details']:
            for seed in d['missing_seeds']:
                key=(d['state_uid'],d['eta_uid'],d['controller_uid'],seed)
                assert key in cases,key
                c=cases[key]
                db=con.execute('SELECT rollout_uid,success,numerical_failure,outcome FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?',key).fetchone()
                assert db is not None and db['rollout_uid']==c['rollout_uid'] and db['success']==0 and db['numerical_failure']==1 and db['outcome']=='numerical_failure',key
                raw=HERE/phase/'raw'/f"{c['rollout_uid']}.json"
                marker=HERE/phase/'journaled'/f"{c['rollout_uid']}.json"
                assert raw.exists() and marker.exists() and read(raw)['numerical_failure']==1 and read(marker)['raw_sha256']==sha(raw),key
                missing.append(c['rollout_uid'])
    assert len(missing)==post['summary']['genuinely_missing']==len(set(missing))
    write(HERE/f'{phase}_POSTFLIGHT_AUDIT.json',dict(status='PASS',utc=utc(),phase=phase,
          total_requests=len(read(HERE/f'{phase}_CASES.json')),
          standard_postflight_sha256=sha(HERE/f'{phase}_POSTFLIGHT.json'),
          unknown_present_in_db=len(missing),unknown_rollout_uids=missing,
          true_absent=0,ambiguous=0,
          reason='Standard reuse planner flags present numerical_failure outcomes as missing for reuse; each remains a retained unknown.'))
    print(json.dumps(dict(phase=phase,status='PASS',unknown=len(missing),true_absent=0)))

if __name__=='__main__':main(sys.argv[1])
