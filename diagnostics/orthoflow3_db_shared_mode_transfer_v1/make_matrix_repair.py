#!/usr/bin/env python3
"""Resume failed matrix shard and replace one invalid VAL seed as a matched block."""
import json
from pathlib import Path

H=Path(__file__).parent
SRC=H/'plans'/'matrix'/'shard3.jsonl'
OUT=H/'plans'/'matrix_repair'
STATE='DB_MODE_val_008'
BAD=31
REPLACEMENT=32

tasks=[json.loads(x) for x in open(SRC) if x.strip()]
eta=json.load(open(H/'selected_transform.json'))['transform']['eta']
# Replace future 31 by future 32 for every compared controller on this state,
# preserving matched-continuation comparability.  Original invalid and valid
# future-31 records remain in raw/ as immutable audit evidence but are excluded
# by the frozen aggregation rule.
for m,e in enumerate(eta):
    tasks.append(dict(state_id=STATE,eta=e,future_index=REPLACEMENT,
        phase='transformed_matrix_repair',probe_id=f'{STATE}_transformed_{m}_{REPLACEMENT}',
        controller='transformed',mode_id=m,split='val'))
tasks.append(dict(state_id=STATE,eta=[0.,0.,0.],future_index=REPLACEMENT,
    phase='transformed_matrix_repair',probe_id=f'{STATE}_safety_{REPLACEMENT}',
    controller='safety',mode_id=-1,split='val'))
OUT.mkdir(parents=True,exist_ok=True)
with open(OUT/'shard0.jsonl','w') as f:
    for t in tasks:f.write(json.dumps(t)+'\n')
(H/'numerical_repair_manifest.json').write_text(json.dumps({
    'reason':'NUMERICAL_SOLVER_FAILURE after authoritative projection retries',
    'invalid_record':{'state_id':STATE,'controller':'transformed','mode_id':9,'future_index':BAD},
    'matched_seed_block_removed':{'state_id':STATE,'future_index':BAD},
    'matched_seed_block_replacement':{'state_id':STATE,'future_index':REPLACEMENT,
      'controllers':['safety']+[f'transformed_mode_{m}' for m in range(12)]},
    'control_semantics_changed':False,
    'aggregation_rule_frozen_before_replacement_outcomes':True
},indent=2,sort_keys=True)+'\n')
print(json.dumps({'source_tasks':len(tasks)-13,'replacement_tasks':13,'plan_tasks':len(tasks)},indent=2))
