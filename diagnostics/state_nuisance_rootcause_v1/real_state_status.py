"""Read-only progress inspection (plus a replaceable local status snapshot)."""
import collections
import datetime
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent/'real_state_expansion'
counts=collections.Counter();outcomes=collections.Counter();numerical=collision=records=0
for path in (OUT/'raw').glob('*.json'):
    r=json.loads(path.read_text());c=r['case'];records+=1
    counts[c['split'],c['state_index']]+=1;outcomes[r['outcome']]+=1
    numerical+=r['numerical_failure'];collision+=r['collision']
status=json.loads((OUT/'batch_status.json').read_text())
r=dict(time=datetime.datetime.now().isoformat(),new_rollouts=records,new_total=49152,
    train_states_complete=32+sum(k[0]=='train' and n==64 for k,n in counts.items()),
    val_states_complete=8+sum(k[0]=='validation' and n==256 for k,n in counts.items()),
    fresh_test_states_complete=sum(k[0]=='test' and n==256 for k,n in counts.items()),
    numerical=numerical,collision=collision,phase=status['phase'],
    fits_complete=len(list((OUT/'fits').glob('*/seed*/complete.json'))),fits_total=54,
    models_frozen=(OUT/'models_frozen.json').exists(),results_ready=(OUT/'research_results.json').exists())
print(json.dumps(r))
(OUT/'live_progress.json').write_text(json.dumps(r,indent=2)+'\n')
