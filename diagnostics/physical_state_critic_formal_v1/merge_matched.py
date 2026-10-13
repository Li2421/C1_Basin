"""Single-writer exact-identity merge and tensor collection for matched panels."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from shared_rollout_db.src.rollout_db import connect, transaction, uid, eta_identity, ROOT as DB_ROOT
from diagnostics.physical_state_critic_formal_v1.matched_worker import EXPERIMENT

HERE = Path(__file__).resolve().parent
FLAGS = ('success', 'deadlock', 'timeout', 'collision', 'numerical_failure')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def merge():
    protocol = read(HERE / 'matched_protocol.json')
    expected = sha(HERE / 'matched_protocol.json')
    states = {s['state_uid']: s for s in protocol['states']}
    stats = Counter()
    journal_dir = DB_ROOT / 'journals' / EXPERIMENT
    with connect() as db:
        for path in sorted(journal_dir.glob('*.jsonl')):
            source = uid('source', dict(path=str(path), sha256=sha(path)))
            if db.execute('SELECT 1 FROM source_file WHERE source_uid=?', (source,)).fetchone():
                stats['already_merged'] += 1
                continue
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            with transaction(db):
                db.execute('INSERT INTO source_file(source_uid,experiment_uid,path,sha256,file_type,classification,rows_seen) VALUES(?,?,?,?,?,?,?)',
                           (source, EXPERIMENT, str(path), sha(path), '.jsonl', 'SEED_EXACT', len(rows)))
                added = duplicate = 0
                for lineno, record in enumerate(rows, 1):
                    assert record['experiment_uid'] == EXPERIMENT
                    assert record['protocol_sha256'] == expected
                    case, raw = record['case'], record['raw']
                    state = states[case['state_uid']]
                    assert state['scenario'] == case['scene']
                    assert raw['state_uid'] == state['uid']
                    if case['chain'] == 'FF':
                        assert raw['controller_uid'] == case['controller_uid']
                    assert eta_identity(case['eta'])[0] == case['eta_uid']
                    assert int(raw['seed']) == case['seed']
                    assert case['controller_uid'] == protocol['controllers'][case['scene']][case['chain']]['controller_uid']
                    ident = dict(state=case['state_uid'], eta=case['eta_uid'],
                                 controller=case['controller_uid'], seed=case['seed_key'])
                    assert case['rollout_uid'] == uid('roll', ident)
                    assert not (record['success'] and (record['collision'] or record['numerical_failure']))
                    core = tuple(record[k] for k in FLAGS)
                    old = db.execute('SELECT * FROM rollout WHERE rollout_uid=?', (case['rollout_uid'],)).fetchone()
                    if old:
                        assert core == tuple(old[k] for k in FLAGS)
                        assert record['episode_length'] == old['episode_length']
                        duplicate += 1
                    else:
                        db.execute('INSERT INTO rollout(rollout_uid,state_uid,eta_uid,controller_uid,seed_key,continuation_seed_json,success,deadlock,timeout,collision,numerical_failure,episode_length,outcome,experiment_uid,original_source_file,compatibility_quality,raw_record_hash) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                   (case['rollout_uid'], case['state_uid'], case['eta_uid'],
                                    case['controller_uid'], case['seed_key'], case['seed_key'],
                                    *core, record['episode_length'], record['outcome'], EXPERIMENT,
                                    str(path), 'EXACT_REUSE', uid('record', record)))
                        added += 1
                    db.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)',
                               (case['rollout_uid'], source, lineno))
                db.execute('UPDATE source_file SET rows_imported=?,rows_duplicate=? WHERE source_uid=?',
                           (added, duplicate, source))
                stats.update(journals=1, inserted=added, duplicates=duplicate)
        count = db.execute('SELECT count(*) FROM rollout WHERE experiment_uid=?', (EXPERIMENT,)).fetchone()[0]
        db.execute('UPDATE experiment SET new_rollout_count=? WHERE experiment_uid=?', (count, EXPERIMENT))
        db.commit()
    (HERE / 'matched_merge_audit.json').write_text(json.dumps(dict(stats, total_new_rows=count), indent=2) + '\n')
    print(json.dumps(dict(stats, total_new_rows=count)))


def collect(scene, chain, split):
    panel = HERE / 'matched' / scene / chain / split
    cases = read(panel / 'cases.json')
    n = 512 if split == 'train' else 32 if split == 'validation' else 128
    k = 16 if split == 'train' else 17
    trials = 4 if split == 'train' else 16
    assert len(cases) == n * k * trials
    values = np.full((n, k, trials, 5), -1, dtype=np.int8)
    seen = np.zeros((n, k, trials), bool)
    with connect(True) as db:
        for c in cases:
            i,j,t = c['state_index'],c['eta_index'],c['seed']
            row = db.execute('SELECT * FROM rollout WHERE rollout_uid=?', (c['rollout_uid'],)).fetchone()
            if row is None:
                continue
            assert row['state_uid'] == c['state_uid']
            assert row['eta_uid'] == c['eta_uid']
            assert row['controller_uid'] == c['controller_uid']
            assert row['seed_key'] == c['seed_key']
            assert row['compatibility_quality'] == 'EXACT_REUSE' and not row['conflict_quarantined']
            assert not seen[i,j,t]
            values[i,j,t] = [row[f] for f in FLAGS]
            seen[i,j,t] = True
    target = panel / 'outcomes.npz'
    np.savez_compressed(target, values=values, seen=seen,
                        state_uids=np.asarray([cases[i*k*trials]['state_uid'] for i in range(n)]),
                        eta=np.asarray([cases[j*trials]['eta'] for j in range(k)]))
    result = dict(scene=scene, chain=chain, split=split,
                  slots=len(cases), observed=int(seen.sum()), missing=int((~seen).sum()),
                  numerical=int(np.sum(values[...,4]==1)),
                  outcome_sha256=sha(target))
    (panel / 'collection_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('merge','collect'))
    parser.add_argument('--scene', choices=('toy_give_way','ring_exchange'))
    parser.add_argument('--chain', choices=('TT','FF'))
    parser.add_argument('--split', choices=('train','validation','test'))
    args = parser.parse_args()
    if args.action == 'merge':
        merge()
    else:
        assert args.scene and args.chain and args.split
        collect(args.scene, args.chain, args.split)
