from collections import Counter
import json,time
from .study import HERE,EXPERIMENT,read,write,sha
from shared_rollout_db.src.rollout_db import connect,transaction,uid,ROOT as DBROOT
FLAGS=("success","deadlock","timeout","collision","numerical_failure")

def run():
    p=read(HERE/'protocol.json');expected=sha(HERE/'protocol.json');stats=Counter()
    expected_cases={r['rollout_uid']:r for r in read(HERE/'cases.json')}
    states={s['state_uid']:s for s in p['states']}
    with connect() as db:
        for path in sorted((DBROOT/'journals'/EXPERIMENT).glob('*.jsonl')):
            source=uid('source',dict(path=str(path),sha256=sha(path)))
            if db.execute('SELECT 1 FROM source_file WHERE source_uid=?',(source,)).fetchone():continue
            records=[json.loads(v) for v in path.read_text().splitlines()]
            with transaction(db):
                db.execute('INSERT INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen) VALUES(?,?,?,?,?,?,?)',
                    (source,EXPERIMENT,str(path),sha(path),'.jsonl','SEED_EXACT',len(records)))
                added=duplicates=0
                for line,r in enumerate(records,1):
                    c=r['case'];s=states[c['state_uid']]
                    assert r['protocol_sha256']==expected and r['experiment_uid']==EXPERIMENT
                    assert c==expected_cases[c['rollout_uid']]
                    if not path.name.startswith('pilot_'):
                        assert r['execution_semantics_version']=='scene_precision_guard_v1_common_exogenous_coupling'
                        assert r['jax_enable_x64']==p['controllers'][c['scene']][c['controller_name']]['payload']['jax_enable_x64']
                    assert not(r['success'] and (r['collision'] or r['numerical_failure']))
                    core=tuple(r[k] for k in FLAGS)
                    old=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
                    if old:
                        assert core==tuple(old[k] for k in FLAGS) and r['episode_length']==old['episode_length']
                        duplicates+=1
                    else:
                        db.execute('INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,outcome,experiment_uid,original_source_file,compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                            (c['rollout_uid'],c['state_uid'],c['eta_uid'],c['controller_uid'],c['seed_key'],c['seed_key'],*core,r['episode_length'],r['outcome'],EXPERIMENT,str(path),'EXACT_REUSE',uid('record',r)))
                        added+=1
                    db.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)',(c['rollout_uid'],source,line))
                db.execute('UPDATE source_file SET rows_imported=?,rows_duplicate=? WHERE source_uid=?',(added,duplicates,source))
                stats.update(journals=1,inserted=added,duplicates=duplicates)
        db.commit()
    write(HERE/'merge_audits'/f'{time.time_ns()}.json',dict(stats));print(dict(stats),flush=True)


if __name__=="__main__":run()
