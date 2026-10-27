"""Experiment-owned utilities; no mutation of imported project modules."""
import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FIELD=ROOT.parent/'Basin_C1_flow_field_poc_20261004'
D=ROOT/'diagnostics'
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
for k,d in [('TMPDIR','tmp'),('MPLCONFIGDIR','mplconfig'),('XDG_CACHE_HOME','cache')]:
    p=HERE/d;p.mkdir(exist_ok=True);os.environ[k]=str(p)
sys.path[:0]=[str(ROOT),str(FIELD),str(FIELD/'source')]
import numpy as np

def utc():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    return x

def write(path,x):
    p=Path(path);assert p.resolve().is_relative_to(HERE)
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean(x),indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def seal(name,paths,**kw):
    p=HERE/name;assert not p.exists()
    write(p,dict(utc=utc(),sha256={str(Path(f).resolve()):sha(f) for f in paths},**kw))
def source_inventory(name):
    files=set()
    for m in list(sys.modules.values()):
        p=getattr(m,'__file__',None)
        if p and Path(p).suffix=='.py' and Path(p).is_file():files.add(str(Path(p).resolve()))
    write(HERE/name,dict(utc=utc(),imported_sources={p:sha(p) for p in sorted(files)}))
