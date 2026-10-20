"""Outcome-blind progress telemetry; never opens rollout labels."""
from pathlib import Path
import json,datetime,subprocess
P=Path(__file__).resolve().parent
now=datetime.datetime.now(datetime.timezone.utc)
x={'utc':now.isoformat()}
for n in ('ADVANCE_COLLECTION_STATUS','ADVANCE_MODELS_STATUS'):
 p=P/(n+'.json')
 if p.exists():x[n]=json.loads(p.read_text())
for phase in ('pilot','remaining','new_evaluation'):
 folder=P/phase/'raw'
 if folder.exists():
  fs=list(folder.glob('*.json'));x[phase]={'raw_count':len(fs),'last60s':sum(now.timestamp()-f.stat().st_mtime<60 for f in fs)}
x['completed_three_seed_fits']=len(list((P/'critic_fits').glob('*/*/COMPLETE.receipt.json')))
x['failures']={p.name:json.loads(p.read_text()) for p in P.glob('ADVANCE_*FAILURE.json')}
print(json.dumps(x,ensure_ascii=False))
