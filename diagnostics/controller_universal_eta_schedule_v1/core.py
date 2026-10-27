"""Round10 immutable artifacts and explicitly authorized DB/journal lifecycle."""
import os,sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];R4=ROOT/'diagnostics/controller_loco_real_flow_v1';DBROOT=ROOT/'shared_rollout_db';EXP='exp_controller_universal_eta_schedule_v1'
PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
sys.dont_write_bytecode=True
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
for k,v in [('TMPDIR','tmp'),('MPLCONFIGDIR','mplconfig'),('XDG_CACHE_HOME','cache')]:
    p=HERE/v;p.mkdir(exist_ok=True);os.environ[k]=str(p)
sys.path.insert(0,str(ROOT))
def guard(event,args):
    if event=='open':
        path,mode,flags=args
        if isinstance(path,(str,bytes,os.PathLike)) and ((isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)):
            p=Path(os.fsdecode(path)).resolve()
            permitted=p.is_relative_to(HERE) or p.is_relative_to(DBROOT/'journals'/EXP) or str(p) in ('/dev/null','/dev/tty')
            if not permitted:raise PermissionError('Round10 external artifact write forbidden: '+str(p))
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
    p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def freeze(p,x):
    p=Path(p)
    if p.exists():assert read(p)==clean(x),('Immutable artifact',str(p))
    else:write(p,x)
def prereg():
    assert sha(HERE/'PREREGISTRATION.md')==read(HERE/'PREREGISTRATION.receipt.json')['sha256']
    assert sha(HERE/'PREREGISTRATION_V2.md')==read(HERE/'PREREGISTRATION_V2.receipt.json')['sha256']
def gate():prereg();assert read(HERE/'CONSISTENCY_GATE.json')['passed'],'Consistency gate has not passed'
R9=ROOT/'diagnostics/controller_eta_switch_time_v1'
P1=ROOT/'diagnostics/controller_q_transfer_round3_v1'
FIELD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
DEV=['base_u_macflow','base_u_v10_local_macflow','base_u_v9_local_macflow','base_u_plus_w_v4_wide_macflow']
CONTROLLERS=read(P1/'P1_EXECUTION_PROTOCOL.json')['controller_order']
FREG={'base_u_plus_w_v3_macflow':2,'base_u_v3_macflow':2,'base_u_v8_local_macflow':4}
ARMS=['F10','F13','Freg','S300','S400','S500']
TIMES=[300,400,500]
