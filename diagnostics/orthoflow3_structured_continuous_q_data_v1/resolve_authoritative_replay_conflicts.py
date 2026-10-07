#!/usr/bin/env python3
"""Resolve the narrowly scoped ep0187 cache conflict using the new authoritative replay.

The pre-existing seed-0..7 rows for this state had already been quarantined after
the historical restore-semantics mismatch.  The present experiment replayed the
same exact keys with the corrected true-t0 driver.  This script promotes only
those newly replayed rows, retains the historical conflict records, and records
an auditable resolution status.  It refuses to touch any non-quarantined key.
"""
from __future__ import annotations

import csv
import glob
import json
import sqlite3
from pathlib import Path

from shared_rollout_db.src.rollout_db import canonical
from shared_rollout_db.src.historical_ingest import jhash

ROOT = Path('/home/zhihan/research/Basin_C1')
H = Path(__file__).resolve().parent
DB = ROOT / 'shared_rollout_db' / 'rollout.sqlite'
EXP = 'exp_ed6a06643facebc8ad7e62b2d1ce53a9df2471add896b59e39e5284fdf9395d9'
STATE_ID = 'T0_POINT_train_000_ep0187'
STATE_UID = 'state_6b5257035d424ca139261cda1cd8f79fc2ce8f68d5f32d4ca1ad67ddeb8bbd18'
CTL = 'ctl_df736b67f6410260d87812e0a76946af7152d147c9a0a25189d1908a92567b34'
RESOLUTION = 'RESOLVED_AUTHORITATIVE_TRUE_T0_REPLAY_2026_10_01'


def outcome_values(r):
    out = str(r.get('outcome') or '').lower()
    return {
        'success': int(bool(r.get('success', out == 'success'))),
        'deadlock': int(bool(r.get('deadlock', out in ('deadlock', 'safe_deadlock')))),
        'timeout': int(bool(r.get('timeout', out == 'timeout'))),
        'collision': int(bool(r.get('collision', out == 'collision'))),
        'numerical_failure': int(bool(r.get('numerical_failure', False)) or bool(r.get('execution_error')) or 'numerical' in out),
        'outcome': out,
    }


def main():
    index = json.load(open(H / 'request_index.json'))
    requests = json.load(open(H / 'planned_rollouts.json'))['requests']
    wanted = {}
    for meta, req in zip(index, requests):
        if meta['state_id'] != STATE_ID:
            continue
        for sk in req['seed_keys']:
            wanted[(meta['eta_uid'], sk)] = (meta, req)

    raw = {}
    for p in sorted(glob.glob(str(H / 'raw' / 'shard*.jsonl'))):
        for line_no, line in enumerate(open(p), 1):
            r = json.loads(line)
            if r.get('state_id') != STATE_ID:
                continue
            sk = canonical({'future_index': int(r['future_index'])})
            raw[(r['eta_uid'], sk)] = (r, p, line_no)

    con = sqlite3.connect(DB, timeout=120)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA busy_timeout=120000')

    # Locate the corresponding append-only journal source for every replay row.
    journal_source = {}
    for sf in con.execute('SELECT source_uid,path FROM source_file WHERE experiment_uid=?', (EXP,)):
        path = Path(sf['path'])
        if not path.exists():
            continue
        for line_no, line in enumerate(open(path), 1):
            r = json.loads(line)
            if r.get('state_id') != STATE_ID:
                continue
            sk = canonical({'future_index': int(r['future_index'])})
            journal_source[(r['eta_uid'], sk)] = (sf['source_uid'], str(path), line_no)

    missing = []
    rows = []
    for key, (meta, req) in wanted.items():
        eta_uid, sk = key
        current = con.execute(
            '''SELECT * FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=?''',
            (STATE_UID, eta_uid, CTL, sk),
        ).fetchone()
        if current is None or current['conflict_quarantined'] == 0:
            continue
        if key not in raw or key not in journal_source:
            missing.append(key)
            continue
        r, raw_path, raw_line = raw[key]
        if r.get('execution_error') or not r.get('first_step', {}).get('feature_sha256'):
            raise RuntimeError(('invalid authoritative replay', key, r.get('execution_error')))
        val = outcome_values(r)
        source_uid, journal_path, journal_line = journal_source[key]
        rows.append((key, current, r, val, source_uid, journal_path, journal_line, raw_path, raw_line))

    if missing:
        raise RuntimeError(('missing authoritative source rows', missing[:3]))
    if len(rows) != 180:
        raise RuntimeError(('expected exactly 180 quarantined keys', len(rows)))

    audit = []
    try:
        con.execute('BEGIN IMMEDIATE')
        for (eta_uid, sk), old, r, val, source_uid, journal_path, journal_line, raw_path, raw_line in rows:
            con.execute(
                '''UPDATE rollout SET success=?,deadlock=?,timeout=?,collision=?,numerical_failure=?,
                   episode_length=?,j_def=?,outcome=?,experiment_uid=?,original_source_file=?,
                   compatibility_quality='EXACT_REUSE',raw_record_hash=?,conflict_quarantined=0
                   WHERE rollout_uid=? AND conflict_quarantined=1''',
                (val['success'], val['deadlock'], val['timeout'], val['collision'], val['numerical_failure'],
                 r.get('episode_length', r.get('continuation_steps')), r.get('J_def'), val['outcome'], EXP,
                 journal_path, jhash(r), old['rollout_uid']),
            )
            if con.total_changes <= 0:
                raise RuntimeError(('failed update', old['rollout_uid']))
            con.execute('INSERT OR IGNORE INTO rollout_source VALUES(?,?,?)', (old['rollout_uid'], source_uid, journal_line))
            identity = canonical({'state': STATE_UID, 'eta': eta_uid, 'controller': CTL, 'seed': sk})
            con.execute('UPDATE conflict SET status=? WHERE identity_key=?', (RESOLUTION, identity))
            audit.append({
                'rollout_uid': old['rollout_uid'], 'state_uid': STATE_UID, 'eta_uid': eta_uid, 'seed_key': sk,
                'old_success': old['success'], 'old_outcome': old['outcome'],
                'authoritative_success': val['success'], 'authoritative_outcome': val['outcome'],
                'journal': journal_path, 'journal_line': journal_line,
                'raw_artifact': raw_path, 'raw_line': raw_line,
                'resolution': RESOLUTION,
            })
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    (H / 'conflict_resolution.json').write_text(json.dumps({
        'scope': STATE_ID,
        'reason': 'Pre-existing rows were quarantined by the known historical restore-semantics mismatch; current corrected true-t0 replay is authoritative.',
        'resolved_keys': len(audit),
        'success_changed': sum(x['old_success'] != x['authoritative_success'] for x in audit),
        'outcome_changed': sum(x['old_outcome'] != x['authoritative_outcome'] for x in audit),
        'resolution': RESOLUTION,
    }, indent=2) + '\n')
    with open(H / 'conflict_resolution_rows.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=audit[0].keys()); w.writeheader(); w.writerows(audit)
    print(json.dumps({'resolved': len(audit), 'success_changed': sum(x['old_success'] != x['authoritative_success'] for x in audit)}))


if __name__ == '__main__':
    main()
