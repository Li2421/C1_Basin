"""Round7 artifact-only utilities. No simulator or database execution."""
import os, sys, json, hashlib
from pathlib import Path
from datetime import datetime, timezone
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
R4=ROOT/'diagnostics/controller_loco_real_flow_v1'
R5=ROOT/'diagnostics/controller_flow_blend_v1'
R6=ROOT/'diagnostics/controller_pathway_fix_v1'
P1=ROOT/'diagnostics/controller_q_transfer_round3_v1'
PY='/home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python'
sys.dont_write_bytecode=True
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
for k,d in [('TMPDIR','tmp'),('MPLCONFIGDIR','mplconfig'),('XDG_CACHE_HOME','cache')]:
    p=HERE/d;p.mkdir(exist_ok=True);os.environ[k]=str(p)
sys.path.insert(0,str(ROOT))
def guard(event,args):
    if event=='open':
        path,mode,flags=args
        if isinstance(path,(str,bytes,os.PathLike)) and ((isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)):
            p=Path(os.fsdecode(path)).resolve()
            if not p.is_relative_to(HERE) and str(p) not in ('/dev/null','/dev/tty'):
                raise PermissionError('Round7 write outside artifact directory: '+str(p))
    if event=='sqlite3.connect':raise PermissionError('Round7 does not need a DB connection')
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
    p=Path(p);assert p.resolve().is_relative_to(HERE);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def check_prereg():assert sha(HERE/'PREREGISTRATION.md')==read(HERE/'PREREGISTRATION.receipt.json')['sha256']
def gate():
    check_prereg();assert read(HERE/'STAGE0.json')['passed'],'Stage0 is mandatory'
def curve(l):
    s=(l==1).sum(-1);v=((l==1)|(l==0)).sum(-1)
    return np.nanmean(np.divide(s,v,out=np.full_like(s,np.nan,dtype=float),where=v>0),axis=0)
REGIMES=np.array([10,13,2,4])
def tasks():
    real=read(R4/'protocol.json')['controller_order'];p5=read(R5/'stage_2/protocol.json')
    ts=[dict(bed='real',coverage='LOCO',fold=i,sources=[n for n in real if n!=h],held=[h]) for i,h in enumerate(real)]
    ts += [dict(bed='blend',coverage=t['coverage'],fold=t['fold'],sources=t['sources'],held=t['held']) for t in read(R5/'stage_2/FIT_TASKS.json') if t['split']=='LOPO' and t['arm']=='R0']
    return ts
def old_name(t,arm):return f"{t['bed']}_{t['coverage']}_fold{t['fold']}_{arm}"
def r0_folder(t):return R4/'fits'/f"fold{t['fold']}_R0" if t['bed']=='real' else R5/'stage_2/fits'/f"LOPO_fold{t['fold']}_{t['coverage']}_R0"
def choose(logits,policy,src):
    from scipy.special import expit
    if policy=='argmax':return logits.argmax(-1)
    q=expit(logits);mu=q.mean(0)
    if policy.startswith('pessimistic'):a=(mu-float(policy[len('pessimistic'):])*q.std(0)).argmax(-1)
    else:
        c=mu.argmax(-1);ok=(q.argmax(-1)==c[None]).all(0)
        a=np.where(ok & (np.take_along_axis(mu,c[...,None],-1)[...,0]-mu[...,src]>=.10),c,src)
    return np.tile(a[None],(3,1,1))
def picked(x,a):return np.take_along_axis(x,a[...,None],-1)[...,0]
def contrast(y,u,a,b):
    diff=a!=b
    return picked(y,a)-picked(y,b)-picked(u,b)*diff,picked(y,a)+picked(u,a)*diff-picked(y,b)
