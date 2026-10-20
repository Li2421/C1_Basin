"""Small read-only snapshot for the primary agent's foreground supervision."""
import json
from pathlib import Path
p=Path(__file__).resolve().parent
r=json.loads((p/'MONITOR_STATUS.json').read_text())
print(json.dumps(dict(utc=r['utc'],phase=r['phase'],records=r['progress'],cpu=r['queue']['cpu_shards'],gpu=r['queue']['gpu_shards'],critic_fits=len(list((p/'critic_fits').glob('*/*/COMPLETE.receipt.json'))),evaluation_local_reuse=len(list((p/'evaluation/dynamic_aliases').glob('*.json'))),errors=[x.name for x in p.glob('*NEEDS_IMPLEMENTATION_FIX.json')],exception=(p/'COORDINATOR_EXCEPTION.json').exists()),ensure_ascii=False))
