"""Distinguish a present, deliberately retained unknown from an absent rollout."""
from core import *
import sqlite3

def main():
    post=read(HERE/'completed_POSTFLIGHT.json')
    assert post['summary']['ambiguous']==0
    con=sqlite3.connect(ROOT/'shared_rollout_db'/'rollout.sqlite')
    con.row_factory=sqlite3.Row
    cases={}
    for k in range(1,5):
        for c in read(HERE/f'wave{k}_MISSING_CASES.json'):
            key=(c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key'])
            assert key not in cases
            cases[key]=(k,c)
    asserted=[]
    for d in post['details']:
        for seed in d['missing_seeds']:
            key=(d['state_uid'],d['eta_uid'],d['controller_uid'],seed)
            assert key in cases,key
            k,c=cases[key]
            db=con.execute('SELECT rollout_uid,success,numerical_failure,outcome FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?',key).fetchone()
            assert db is not None and db['rollout_uid']==c['rollout_uid'] and db['success']==0 and db['numerical_failure']==1 and db['outcome']=='numerical_failure',key
            raw=HERE/f'wave{k}'/'raw'/f"{c['rollout_uid']}.json"
            marker=HERE/f'wave{k}'/'journaled'/f"{c['rollout_uid']}.json"
            assert raw.exists() and marker.exists() and read(raw)['numerical_failure']==1 and read(marker)['raw_sha256']==sha(raw),key
            asserted.append(c['rollout_uid'])
    assert len(asserted)==post['summary']['genuinely_missing']==376
    assert len(set(asserted))==len(asserted)
    assert len(cases)==57344
    in_db=con.execute('SELECT count(*) FROM rollout WHERE experiment_uid=?',(EXP,)).fetchone()[0]
    assert in_db==57376,(in_db,len(cases))
    write(HERE/'POSTFLIGHT_UNKNOWN_AUDIT.json',dict(utc=utc(),status='PASS',standard_postflight_summary=post['summary'],standard_postflight_sha256=sha(HERE/'completed_POSTFLIGHT.json'),main_cases=len(cases),database_rows_including_32_gate=in_db,unknown_present_in_db=len(asserted),true_absent=0,ambiguous=0,reason='The standard reuse planner classifies numerical_failure outcomes as missing for reuse, although each is present in DB and is deliberately retained as unknown under the preregistration.',unknown_rollout_uids=asserted))
    print('postflight unknown audit PASS',len(asserted))

if __name__=='__main__':main()
