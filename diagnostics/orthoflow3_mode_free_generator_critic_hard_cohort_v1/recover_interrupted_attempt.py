#!/usr/bin/env python3
"""Quarantine invalid GPU feature checks; recover valid records only."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path('/home/zhihan/research/Basin_C1')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from shared_rollout_db.src.cache_writer import append_journal


def main():
    quarantine = OUT / 'quarantine_job1261'
    if quarantine.exists():
        raise RuntimeError('Recovery already run; refusing to overwrite archive')
    quarantine.mkdir()
    journal_dir = ROOT / 'shared_rollout_db/journals/exp_mode_free_generator_critic_hard_cohort_v1'
    discarded_journals = 0
    for path in sorted(journal_dir.glob('*.jsonl')):
        shutil.move(str(path), str(quarantine / path.name))
        discarded_journals += 1
    valid, invalid = [], []
    for path in sorted((OUT / 'raw').glob('shard[0-5].jsonl')):
        archived = quarantine / path.name
        shutil.move(str(path), str(archived))
        for line in archived.read_text().splitlines():
            row = json.loads(line)
            (valid if row.get('scientific_outcome_valid') is True else invalid).append(row)
    by_shard = {i: [] for i in range(6)}
    from diagnostics.orthoflow3_mode_free_generator_critic_hard_cohort_v1.run_rollouts import tasks
    task_shards = {(int(t['state']['episode_index']), t['kind'], t['future_index']): i % 6
                   for i, t in enumerate(tasks())}
    for row in valid:
        key = (row['episode_index'], row['kind'], row['future_index'])
        by_shard[task_shards[key]].append(row)
    for i, rows in by_shard.items():
        with (OUT / 'raw' / f'shard{i}.jsonl').open('w') as sink:
            for row in rows:
                sink.write(json.dumps(row, sort_keys=True) + '\n')
    # The merger's unique-key handling makes already-merged smoke/recovery rows idempotent.
    for i in range(0, len(valid), 250):
        append_journal(valid[i:i+250], 'exp_mode_free_generator_critic_hard_cohort_v1', 'recovery_job1261')
    (OUT / 'recovery_job1261.json').write_text(json.dumps({
        'valid_recovered': len(valid), 'invalid_quarantined': len(invalid),
        'unmerged_journals_quarantined': discarded_journals,
        'invalid_by_reason': sorted(set(str(x.get('execution_error')) for x in invalid)),
    }, indent=2) + '\n')
    print(json.dumps({'valid_recovered': len(valid), 'invalid_quarantined': len(invalid),
                      'unmerged_journals_quarantined': discarded_journals}))


if __name__ == '__main__':
    main()
