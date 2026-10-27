"""v2 output boundary; never import the v1 environment module."""
import hashlib
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V1 = ROOT / 'diagnostics/baseline2_literature_scenes_v1'
ROUND28 = ROOT / 'diagnostics/decentralized_pipeline_v1'
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
for key, folder in (('TMPDIR','tmp'), ('MPLCONFIGDIR','mplconfig'),
                    ('XDG_CACHE_HOME','cache'), ('CUDA_CACHE_PATH','cuda_cache')):
    path = HERE / folder
    path.mkdir(exist_ok=True)
    os.environ[key] = str(path)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

NAMES = {'lit_narrow_passage':'窄通道场景', 'lit_crowded_random':'二维拥挤随机场景',
         'lit_circle_five':'圆形围场五机随机场景', 'lit_doorway':'门口场景'}

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def clean(value):
    if isinstance(value, dict): return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [clean(v) for v in value]
    if hasattr(value, 'tolist'): return clean(value.tolist())
    if isinstance(value, float) and not __import__('math').isfinite(value): return None
    return value

def canonical(value):
    return json.dumps(clean(value),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)

def uid(prefix, value):
    return prefix + '_' + hashlib.sha256(canonical(value).encode()).hexdigest()

def write(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(HERE), path
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)

def seal(name, paths, **metadata):
    path = HERE / name
    assert not path.exists(), path
    write(path,dict(utc=utc(),files={str(Path(p).resolve()):sha(p) for p in paths},**metadata))

def prereg():
    receipt=read(HERE/'PREREGISTRATION.receipt.json')
    for path,expected in receipt['files'].items():
        assert sha(path)==expected,path
