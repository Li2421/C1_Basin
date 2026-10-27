"""Round11 immutable IO; label access denied until R6 receipt exists."""
import os,sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=Path('/home/zhihan/research/Basin_C1');DBROOT=ROOT/'shared_rollout_db';EXP='exp_four_controller_census_v1'
R5=ROOT/'diagnostics/controller_flow_blend_v1';R7=ROOT/'diagnostics/controller_behavior_repr_v1';R9=ROOT/'diagnostics/controller_eta_switch_time_v1';R10=ROOT/'diagnostics/controller_universal_eta_schedule_v1';R4=ROOT/'diagnostics/controller_loco_real_flow_v1';P1=ROOT/'diagnostics/controller_q_transfer_round3_v1';FIELD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004')
PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['JAX_PLATFORMS']='cpu';os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
for k,v in [('TMPDIR','tmp'),('MPLCONFIGDIR','mplconfig'),('XDG_CACHE_HOME','cache')]:
    p=HERE/v;p.mkdir(exist_ok=True);os.environ[k]=str(p)
sys.path[:0]=[str(ROOT),str(FIELD),str(FIELD/'source')]
def guard(event,args):
    if event=='open':
        path,mode,flags=args
        if not isinstance(path,(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(path)).resolve();writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if writing:
            if not(p.is_relative_to(HERE) or p.name in ('rollout.sqlite','rollout.sqlite-wal','rollout.sqlite-shm') and p.parent==DBROOT or str(p) in ('/dev/null','/dev/tty')):raise PermissionError('External round11 write: '+str(p))
        elif not (HERE/'PREREGISTRATION.receipt.json').exists() and p.is_relative_to(HERE/'main') and p.suffix in ('.json','.npz'):
            raise PermissionError('Final success evidence forbidden before probe/selector freeze: '+str(p))
sys.addaudithook(guard)
import numpy as np
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
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
    if Path(p).exists():assert read(p)==clean(x),str(p)
    else:write(p,x)
def prereg():assert sha(HERE/'PREREGISTRATION.md')==read(HERE/'PREREGISTRATION.receipt.json')['sha256']
def gate():prereg();assert read(HERE/'CONSISTENCY_GATE.json')['passed']
def labels_allowed():prereg()
# Worker journals are kept within the allowed experiment directory.
from shared_rollout_db.src import cache_writer
cache_writer.ROOT=HERE

E8=ROOT/'diagnostics/controller_ff_factorial_repair_v1'
SCENES=['toy_give_way','double_bottleneck','four_way_intersection','ring_exchange']
NAMES=['flow_isotropic','flow_anisotropic','flow_coupled']
ARMS=['F10','F13','F15','S13_t1','S13_t2','S15_t1','S15_t2']
