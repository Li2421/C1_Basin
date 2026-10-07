#!/usr/bin/env python3
"""Journal valid local records still missing from global DB after job cancellation."""
from __future__ import annotations

import csv
import argparse
import json
import sys
from pathlib import Path

ROOT = Path('/home/zhihan/research/Basin_C1')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src.rollout_db import canonical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', type=Path, default=OUT / 'cache_after_cancel.json')
    args = parser.parse_args()
    preflight = json.loads(args.preflight.read_text())
    missing = {(d['state_uid'], d['eta_uid'], d['controller_uid'], seed)
               for d in preflight['details'] for seed in d['missing_seeds']}
    with (OUT / 'candidate_cache_keys.csv').open(newline='') as f:
        lookup = {(int(r['episode_index']), r['kind']):
                  (r['state_uid'], r['eta_uid'], r['controller_uid'])
                  for r in csv.DictReader(f)}
    recovered = []
    for path in sorted((OUT / 'raw').glob('shard[0-5].jsonl')):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if not r['scientific_outcome_valid']:
                raise RuntimeError(f'Invalid raw record: {path}')
            key = (*lookup[(r['episode_index'], r['kind'])], canonical({'future_index': r['future_index']}))
            if key in missing:
                recovered.append(r)
                missing.remove(key)
    for i in range(0, len(recovered), 250):
        append_journal(recovered[i:i+250], 'exp_mode_free_generator_critic_hard_cohort_v1', 'recovery_after_cancel')
    print(json.dumps({'unjournaled_recovered': len(recovered)}))


if __name__ == '__main__':
    main()
