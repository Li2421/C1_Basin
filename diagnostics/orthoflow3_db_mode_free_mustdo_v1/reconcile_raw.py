#!/usr/bin/env python3
"""Journal complete raw records not yet indexed, after an intentional worker stop."""
import json,sqlite3,sys
from pathlib import Path
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from shared_rollout_db.src.rollout_db import eta_identity
from shared_rollout_db.src.cache_writer import append_journal
f=json.loads((HERE/'db_frozen_proposals.json').read_text())['states']
uid={r['state_id']:r['state_uid'] for r in f}
ctl=json.loads((ROOT/'diagnostics/orthoflow3_pipeline_dataset_evidence_audit_v1/db_hard_frozen_generator_mean.json').read_text())['hard_panel_controller_uid']
con=sqlite3.connect(f'file:{ROOT}/shared_rollout_db/rollout.sqlite?mode=ro',uri=True)
pending=[];seen=set();total=0
for path in sorted((HERE/'raw').glob('proposals_shard*.jsonl')):
    for line in path.open():
        r=json.loads(line);total+=1
        key=(uid[r['state_id']],eta_identity(r['eta'])[0],ctl,json.dumps({'future_index':r['future_index']},sort_keys=True,separators=(',',':')))
        if key in seen:continue
        seen.add(key)
        if con.execute('SELECT 1 FROM rollout WHERE state_uid=? AND eta_uid=? AND controller_uid=? AND seed_key=? AND conflict_quarantined=0',key).fetchone():continue
        pending.append(r)
for i in range(0,len(pending),64):append_journal(pending[i:i+64],'orthoflow3_db_mode_free_mustdo_v1','reconcile_raw')
print(json.dumps({'raw_records':total,'unique_keys':len(seen),'journaled_unindexed':len(pending)}))
