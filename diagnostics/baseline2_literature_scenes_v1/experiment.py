"""Experiment boundary, serialization, and immutable receipts."""
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
for name,folder in [('TMPDIR','tmp'),('MPLCONFIGDIR','mplconfig'),('XDG_CACHE_HOME','cache'),('CUDA_CACHE_PATH','cuda_cache')]:
    p=HERE/folder;p.mkdir(exist_ok=True);os.environ[name]=str(p)
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['PYTHONDONTWRITEBYTECODE']='1'

def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def clean(x):
    import numpy as np
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not __import__('math').isfinite(x):return None
    return x
def canonical(obj):return json.dumps(clean(obj),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
def uid(prefix,obj):return prefix+'_'+hashlib.sha256(canonical(obj).encode()).hexdigest()
def write(p,obj):
    p=Path(p);assert p.resolve().is_relative_to(HERE)
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp')
    q.write_text(json.dumps(clean(obj),indent=2,ensure_ascii=False,allow_nan=False)+'\n');q.replace(p)
def seal(name,paths,**kw):
    p=HERE/name;assert not p.exists()
    write(p,dict(utc=utc(),files={str(Path(f).resolve()):sha(f) for f in paths},**kw))
def prereg():
    r=read(HERE/'PREREGISTRATION.receipt.json')
    for p,h in r['files'].items():assert sha(p)==h,p

NAMES={'lit_narrow_passage':'窄通道场景','lit_crowded_random':'二维拥挤随机场景','lit_circle_five':'圆形围场五机随机场景','lit_doorway':'门口场景'}
