import os,sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];D=ROOT/'diagnostics';EXP='exp_axis1_state_v1'
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
for k,v in [('TMPDIR','tmp'),('MPLCONFIGDIR','mplconfig'),('XDG_CACHE_HOME','cache')]:
 p=HERE/v;p.mkdir(exist_ok=True);os.environ[k]=str(p)
os.environ['JAX_PLATFORMS']='cpu';os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
sys.path.insert(0,str(ROOT))
import numpy as np
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def clean(x):
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 if isinstance(x,np.ndarray):return clean(x.tolist())
 if isinstance(x,np.generic):return clean(x.item())
 if isinstance(x,float) and not np.isfinite(x):return None
 return x
def write(p,x):
 p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def seal(name,paths,**kw):write(HERE/name,dict(utc=utc(),files={str(p):sha(p) for p in paths},**kw))
def prereg():assert sha(HERE/'PREREGISTRATION.md')==read(HERE/'PREREGISTRATION.receipt.json')['sha256']
