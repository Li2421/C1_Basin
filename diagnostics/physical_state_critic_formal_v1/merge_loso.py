"""Single-writer exact-identity merge and Q4/Q16 tensor collection for LOSO."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from shared_rollout_db.src.rollout_db import connect, transaction, uid, eta_identity, ROOT as DB_ROOT

HERE=Path(__file__).resolve().parent
EXPERIMENT='exp_formal_physical_state_critic_loso_20261008_v1'
FLAGS=('success','deadlock','timeout','collision','numerical_failure')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def merge():
    protocol=read(HERE/'loso_protocol.json')
    digest=sha(HERE/'loso_protocol.json')
    stats=Counter()
    with connect() as db:
        for path in sorted((DB_ROOT/'journals'/EXPERIMENT).glob('*.jsonl')):
            source=uid('source',dict(path=str(path),sha256=sha(path)))
            if db.execute('SELECT 1 FROM source_file WHERE source_uid=?',(source,)).fetchone():
                stats['already_merged']+=1;continue
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            with transaction(db):
                db.execute('INSERT INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen) VALUES(?,?,?,?,?,?,?)',
                           (source,EXPERIMENT,str(path),sha(path),'.jsonl','SEED_EXACT',len(rows)))
                added=duplicate=0
                for lineno,record in enumerate(rows,1):
                    assert record['experiment_uid']==EXPERIMENT and record['protocol_sha256']==digest
                    case,raw=record['case'],record['raw']
                    assert raw['state_uid']==case['state_uid']
                    assert raw['future_index']==case['future_index']
                    assert eta_identity(raw['eta'])[0]==case['eta_uid']
                    if case['scene'] in ('four_way_intersection','ring_exchange'):
                        assert raw['controller_uid']==case['controller_uid']
                    ctl=protocol['controllers'][case['scene']]
                    assert case['controller_uid']==ctl['safety_uid' if case['eta_index']==16 else 'intervention_uid']
                    identity=dict(state=case['state_uid'],eta=case['eta_uid'],
                                  controller=case['controller_uid'],seed=case['seed_key'])
                    assert case['rollout_uid']==uid('roll',identity)
                    assert not(record['success'] and (record['collision'] or record['numerical_failure']))
                    core=tuple(record[k] for k in FLAGS)
                    old=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(case['rollout_uid'],)).fetchone()
                    if old:
                        assert core==tuple(old[k] for k in FLAGS)
                        assert record['episode_length']==old['episode_length']
                        duplicate+=1
                    else:
                        db.execute('INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,outcome,experiment_uid,original_source_file,compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                   (case['rollout_uid'],case['state_uid'],case['eta_uid'],case['controller_uid'],
                                    case['seed_key'],case['seed_key'],*core,record['episode_length'],record['outcome'],
                                    EXPERIMENT,str(path),'EXACT_REUSE',uid('record',record)))
                        added+=1
                    db.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)',
                               (case['rollout_uid'],source,lineno))
                db.execute('UPDATE source_file SET rows_imported=?,rows_duplicate=? WHERE source_uid=?',
                           (added,duplicate,source))
                stats.update(journals=1,inserted=added,duplicates=duplicate)
        count=db.execute('SELECT count(*) FROM rollout WHERE experiment_uid=?',(EXPERIMENT,)).fetchone()[0]
        db.execute('UPDATE experiment SET new_rollout_count=? WHERE experiment_uid=?',(count,EXPERIMENT))
        db.commit()
    (HERE/'loso_merge_audit.json').write_text(json.dumps(dict(stats,total_new_rows=count),indent=2)+'\n')
    print(json.dumps(dict(stats,total_new_rows=count)))


def collect(scene,split):
    panel=HERE/'loso'/scene/split
    cases=read(panel/'cases.json')
    n={'train':512,'validation':32,'test':128}[split]
    k=16 if split=='train' else 17
    trials=4 if split=='train' else 16
    assert len(cases)==n*k*trials
    values=np.full((n,k,trials,5),-1,np.int8)
    seen=np.zeros((n,k,trials),bool)
    with connect(True) as db:
        for c in cases:
            i,j,t=c['state_index'],c['eta_index'],c['future_index']
            row=db.execute('SELECT * FROM rollout WHERE rollout_uid=?',(c['rollout_uid'],)).fetchone()
            if row is None:continue
            assert row['state_uid']==c['state_uid'] and row['eta_uid']==c['eta_uid']
            assert row['controller_uid']==c['controller_uid'] and row['seed_key']==c['seed_key']
            assert row['compatibility_quality']=='EXACT_REUSE' and not row['conflict_quarantined']
            assert not seen[i,j,t]
            values[i,j,t]=[row[f] for f in FLAGS];seen[i,j,t]=True
    path=panel/'outcomes.npz'
    np.savez_compressed(path,values=values,seen=seen,
                        state_uids=np.asarray([cases[i*k*trials]['state_uid'] for i in range(n)]),
                        eta=np.asarray([cases[i*k*trials+j*trials]['eta']
                                        for i in range(n) for j in range(k)]).reshape(n,k,3))
    audit=dict(scene=scene,split=split,slots=len(cases),observed=int(seen.sum()),
               missing=int((~seen).sum()),numerical=int(np.sum(values[...,4]==1)),
               outcome_sha256=sha(path))
    (panel/'collection_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('merge','collect'))
    p.add_argument('--scene',choices=('toy_give_way','double_bottleneck','four_way_intersection','ring_exchange'))
    p.add_argument('--split',choices=('train','validation','test'))
    a=p.parse_args()
    if a.action=='merge':merge()
    else:
        assert a.scene and a.split
        collect(a.scene,a.split)
