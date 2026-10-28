"""Private production imports and canonical experiment helpers."""
import guard
from guard import HERE,ROOT
import sys
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(HERE/'generated'))
import bootstrap
import runtime_new as RN
import hashlib,json,math
import numpy as np
from shared_rollout_db.src import rollout_db as db,planner,cache_writer
import shared_rollout_db
from pathlib import Path
assert Path(shared_rollout_db.__file__).resolve().is_relative_to(ROOT)
assert cache_writer.ROOT.resolve()==ROOT/'shared_rollout_db'
EXP='exp_scene_types_remaining_20261028'

def clean(x):
    if isinstance(x,dict):return {str(k):clean(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v)for v in x]
    if hasattr(x,'tolist'):return clean(x.tolist())
    if isinstance(x,float)and not math.isfinite(x):return None
    return x

def dump(path,x):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)

def read(path):return json.loads(Path(path).read_text())
def digest(x):return hashlib.sha256(db.canonical(clean(x)).encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def controllers():
    v=read(ROOT/'diagnostics/cross_scene_correction_v1/SELECTED_CONTROLLERS.json')['xsc_swap_n6']
    return [v[k]for k in ('centralized_sampling','gradient_mpc')]
