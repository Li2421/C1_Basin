#!/usr/bin/env python3
"""Quarantine the known pre-transition history-adapter errors, preserving bytes."""

import json
import os
from pathlib import Path

HERE=Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_q_learnability_v2/raw/base_rollout_plan')
total=0
for path in sorted(HERE.glob('shard*.jsonl')):
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    invalid=[row for row in rows if (row.get('execution_error') or {}).get('message')=='goal-error history contains NaN/Inf' and int(row.get('continuation_steps',-1))==0]
    valid=[row for row in rows if row not in invalid]
    if invalid:
        quarantine=path.with_name(path.stem+'_quarantined_pretransition.jsonl')
        quarantine.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in invalid))
        tmp=path.with_suffix('.jsonl.tmp'); tmp.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in valid)); os.replace(tmp,path)
    print(path.name,'valid',len(valid),'quarantined',len(invalid))
    total+=len(invalid)
print('total_quarantined',total)
