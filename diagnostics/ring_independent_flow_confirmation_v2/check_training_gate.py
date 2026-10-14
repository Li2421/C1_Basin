import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import json,pickle,hashlib,datetime
import numpy as np
H=Path(__file__).resolve().parent;R=H.parents[1]
def ph(p):
 x=pickle.loads(p.read_bytes())['agent']['network']['params'];h=hashlib.sha256()
 def walk(x,path=()):
  if isinstance(x,dict):
   for k in sorted(x):walk(x[k],path+(str(k),))
  else:
   a=np.asarray(x);h.update('/'.join(path).encode());h.update(str(a.dtype).encode());h.update(json.dumps(a.shape).encode());h.update(a.tobytes())
 walk(x);return h.hexdigest()
old=json.loads((R/'diagnostics/ring_exchange_stage1/base_u_v2_macflow/training_summary.json').read_text());rows=[]
for dev in ('cpu','gpu'):
 d=H/('reproduction_'+dev);new=json.loads((d/'training_summary.json').read_text());err={k:abs(new[k]-old[k])/abs(old[k]) for k in ('best_fixed_dev_loss','final_fixed_dev_loss')};rows.append(dict(device=dev,old=old,new=new,relative_error=err,passed=all(v<=.01 for v in err.values()),best_step_equal=new['best_step']==old['best_step'],parameter_hashes={kind:dict(old=ph(R/'diagnostics/ring_exchange_stage1/base_u_v2_macflow'/f'{kind}.pkl'),new=ph(d/f'{kind}.pkl')) for kind in ('best','final')},runtime=json.loads((H/f'reproduction_{dev}_START.receipt.json').read_text())))
obj=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),primary_passed=any(x['passed'] for x in rows),rows=rows,code_drift_alone_explains_failure=False)
(H/'PRIMARY_TRAINING_GATE.json').write_text(json.dumps(obj,indent=2));print([(x['device'],x['passed'],x['relative_error']) for x in rows])
