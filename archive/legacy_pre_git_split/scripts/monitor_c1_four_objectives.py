"""Read-only progress reporting; summarize only after every evaluation finishes."""
import json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/c1_four_objectives_multiseed'
while True:
    runs={}
    for f in OUT.rglob('history.json'):
        try:h=json.loads(f.read_text())
        except (OSError,json.JSONDecodeError):continue
        r=dict(attempts=len(h),accepted=sum(x['accepted'] for x in h),errors=sum(x['base_error'] is not None for x in h))
        if f.parent.name not in runs or r['attempts']>runs[f.parent.name]['attempts']:runs[f.parent.name]=r
    evaluations=len(list((OUT/'evaluation').glob('*/complete.json')))
    print(json.dumps(dict(attempts=sum(r['attempts'] for r in runs.values()),accepted=sum(r['accepted'] for r in runs.values()),base_errors=sum(r['errors'] for r in runs.values()),evaluations_complete=evaluations,runs=runs)),flush=True)
    if evaluations==13:
        with (OUT/'aggregate.log').open('w') as f:
            subprocess.run([str(ROOT/'.venv-c1/bin/python'),str(ROOT/'scripts/summarize_c1_four_objectives.py')],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        print('AGGREGATION COMPLETE',flush=True);break
    time.sleep(45)
