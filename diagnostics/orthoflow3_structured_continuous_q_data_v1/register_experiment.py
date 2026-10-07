#!/usr/bin/env python3
import hashlib,json,sqlite3
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1];DB=ROOT/'shared_rollout_db/rollout.sqlite'
sys_path=str(ROOT)
import sys;sys.path.insert(0,sys_path)
from shared_rollout_db.src.rollout_db import uid,canonical
exp=uid('exp',{'path':str(H)})
pre=json.load(open(H/'cache_preflight.json'))['summary'];runt=[json.load(open(p)) for p in sorted((H/'raw').glob('shard*_runtime.json'))]
# The legacy runner counter includes rows loaded into its local cache.  The
# authoritative number physically requested by this experiment is the global
# DB preflight's genuinely-missing continuation count.
new=pre['genuinely_missing']
protocol=hashlib.sha256((H/'data_design.md').read_bytes()).hexdigest()
with sqlite3.connect(DB) as c:
 c.execute('''UPDATE experiment SET name=?,path=?,protocol_hash=?,end_time=CURRENT_TIMESTAMP,reused_rollout_count=?,new_rollout_count=?,metadata_json=? WHERE experiment_uid=?''',
  ('ORTHOFLOW3_STRUCTURED_CONTINUOUS_Q_DATA_V1',str(H),protocol,pre['exact_reusable']+pre['partial_reusable']+pre['aggregate_reusable'],new,
   canonical({'matrix':'structured state x eta','state_split':'48/12/16','eta_split':'40/12/32','rollout_shards':6,'db_new_rollout':0}),exp))
print(json.dumps({'experiment_uid':exp,'reused':pre['exact_reusable']+pre['partial_reusable']+pre['aggregate_reusable'],'new':new}))
