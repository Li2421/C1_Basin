import os,sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];DBROOT=ROOT/'shared_rollout_db';EXP='exp_ring_eta_pair_generality_v1';R9=ROOT/'diagnostics/controller_eta_switch_time_v1';R10=ROOT/'diagnostics/controller_universal_eta_schedule_v1';P1=ROOT/'diagnostics/controller_q_transfer_round3_v1';R4=ROOT/'diagnostics/controller_loco_real_flow_v1';FIELD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['JAX_PLATFORMS']='cpu';os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
for k,v in [('TMPDIR','tmp'),('MPLCONFIGDIR','mplconfig'),('XDG_CACHE_HOME','cache')]:
 p=HERE/v;p.mkdir(exist_ok=True);os.environ[k]=str(p)
sys.path[:0]=[str(ROOT),str(FIELD),str(FIELD/'source')]
import numpy as np
def path(p):
 p=Path(p)
 return HERE/'gate_protocol.json' if p==HERE/'protocol.json' and os.environ.get('ROUND22_PHASE')=='gate' else p
def read(p):return json.loads(path(p).read_text())
def sha(p):return hashlib.sha256(path(p).read_bytes()).hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def write(p,x):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(clean(x),indent=2,allow_nan=False)+'\n');t.replace(p)
def freeze(p,x):
 if Path(p).exists():assert read(p)==clean(x)
 else:write(p,x)
def prereg():assert sha(HERE/'PREREGISTRATION.md')==read(HERE/'PREREGISTRATION.receipt.json')['sha256']
def gate():prereg();assert read(HERE/'CONSISTENCY_GATE.json')['passed']
from shared_rollout_db.src import cache_writer
cache_writer.ROOT=HERE
