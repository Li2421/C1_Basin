"""Round18 artifact IO and phased pre-label gates."""
import os,sys,json,hashlib
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
R14=ROOT/'diagnostics/ring_independent_flow_confirmation_v2';R15=ROOT/'diagnostics/ring_controller_population_v1';R16=ROOT/'diagnostics/ring_controller_critic_v1';R17=ROOT/'diagnostics/ring_critic_new_recipe_confirmation_v1';R11=ROOT/'diagnostics/controller_late_phase_probe_v1';R9=ROOT/'diagnostics/controller_eta_switch_time_v1';R10=ROOT/'diagnostics/controller_universal_eta_schedule_v1';R5=ROOT/'diagnostics/controller_flow_blend_v1';R7=ROOT/'diagnostics/controller_behavior_repr_v1';P1=ROOT/'diagnostics/controller_q_transfer_round3_v1';FIELD=Path('/home/zhihan/research/Basin_C1_flow_field_poc_20261004');EXP='exp_ring_eta2_gap_v1'
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
for k,v in [('TMPDIR','tmp'),('MPLCONFIGDIR','mplconfig'),('XDG_CACHE_HOME','cache')]:
 p=HERE/v;p.mkdir(exist_ok=True);os.environ[k]=str(p)
sys.path[:0]=[str(ROOT),str(FIELD),str(FIELD/'source')]
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
def features_frozen():
 prereg()
 f=HERE/'FEATURES_FROZEN.receipt.json'
 assert f.exists(),'No new success labels before all R5/R5eta2 probes and features freeze'
 r=read(f)
 assert r['status']=='FROZEN_BEFORE_ANY_NEW_SUCCESS_LABELS'
 for path,digest in r['files'].items():assert sha(HERE/path)==digest,path
def train_labels_allowed():features_frozen();assert read(HERE/'CONSISTENCY_GATE.json')['passed']
def test_labels_allowed():
 train_labels_allowed()
 f=HERE/'TEST_FREEZE.receipt.json'
 assert f.exists(),'No TEST labels before critic and choices freeze'
 r=read(f);assert r['status']=='FROZEN_BEFORE_TEST_LABELS'
 for path,digest in r['files'].items():assert sha(HERE/path)==digest,path
from shared_rollout_db.src import cache_writer
cache_writer.ROOT=HERE
