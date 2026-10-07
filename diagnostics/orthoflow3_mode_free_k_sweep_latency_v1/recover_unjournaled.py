#!/usr/bin/env python3
"""Recover only locally valid records still absent from the global cache."""
import csv
import json
import sys
from pathlib import Path

ROOT=Path('/home/zhihan/research/Basin_C1')
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from shared_rollout_db.src.cache_writer import append_journal
from shared_rollout_db.src.rollout_db import canonical

def main():
    p=json.loads((OUT/'cache_postflight.json').read_text())
    missing={(d['state_uid'],d['eta_uid'],d['controller_uid'],s)
             for d in p['details'] for s in d['missing_seeds']}
    with (OUT/'candidate_cache_keys.csv').open(newline='') as f:
        keys={(int(r['episode_index']),r['kind']):(r['state_uid'],r['eta_uid'],r['controller_uid'])
              for r in csv.DictReader(f)}
    found=[]
    for path in sorted((OUT/'raw').glob('shard*.jsonl')):
        for line in path.read_text().splitlines():
            r=json.loads(line)
            if not r['scientific_outcome_valid']: raise RuntimeError(f'Invalid local record in {path}')
            key=(*keys[(r['episode_index'],r['kind'])],canonical({'future_index':r['future_index']}))
            if key in missing:
                found.append(r); missing.remove(key)
    if missing: raise RuntimeError(f'{len(missing)} cache-missing records not present locally')
    for i in range(0,len(found),250):
        append_journal(found[i:i+250],'exp_mode_free_k_sweep_latency_v1','recovery_unjournaled')
    print(json.dumps({'recovered':len(found)}))

if __name__=='__main__': main()
