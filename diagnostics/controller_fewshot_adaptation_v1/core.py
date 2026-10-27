"""Round8 artifact-only execution and immutable input utilities."""
import os,sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
R4=ROOT/'diagnostics/controller_loco_real_flow_v1';R5=ROOT/'diagnostics/controller_flow_blend_v1';R6=ROOT/'diagnostics/controller_pathway_fix_v1';R7=ROOT/'diagnostics/controller_behavior_repr_v1';P1=ROOT/'diagnostics/controller_q_transfer_round3_v1'
PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
sys.dont_write_bytecode=True
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
for k,v in [('TMPDIR','tmp'),('MPLCONFIGDIR','mplconfig'),('XDG_CACHE_HOME','cache')]:
    p=HERE/v;p.mkdir(exist_ok=True);os.environ[k]=str(p)
sys.path.insert(0,str(ROOT))
def guard(event,args):
    if event=='open':
        path,mode,flags=args
        if isinstance(path,(str,bytes,os.PathLike)) and ((isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)):
            p=Path(os.fsdecode(path)).resolve()
            if not p.is_relative_to(HERE) and str(p) not in ('/dev/null','/dev/tty'):raise PermissionError('Round8 external write forbidden: '+str(p))
    if event=='sqlite3.connect':raise PermissionError('Round8 has no database connection')
sys.addaudithook(guard)
import numpy as np
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def utc():return datetime.now(timezone.utc).isoformat()
def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def write(p,x):
    p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def prereg():assert sha(HERE/'PREREGISTRATION.md')==read(HERE/'PREREGISTRATION.receipt.json')['sha256']
def gate():prereg();assert read(HERE/'POOL_AUDIT.json')['passed']
BUDGETS=np.array([0,4,8,16,32,64,128]);STRATEGIES=['U','R','A'];REPEATS=200
METHODS=['P0','E','E-safe','H','H-safe','C-R0','C-R0-safe','C-ctl-R0','C-ctl-R0-safe','C-F2','C-F2-safe','C-ctl-F2','C-ctl-F2-safe']
def picked(y,a):return np.take_along_axis(y,a[...,None],axis=-1)[...,0]
def contrast(y,u,a,b):
    different=a!=b
    return picked(y,a)-picked(y,b)-picked(u,b)*different,picked(y,a)+picked(u,a)*different-picked(y,b)
