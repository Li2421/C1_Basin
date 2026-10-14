import os,sys,json,inspect,hashlib,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['TMPDIR']=str(HERE/'tmp');(HERE/'tmp').mkdir(exist_ok=True)
audit=json.loads((HERE/'HISTORICAL_CODE_AUDIT.json').read_text());sys.path.insert(0,str(HERE/audit['chosen']))
import jax
from new_benchmark_common.training import train_stage1
run=sys.argv[1];seed=int(sys.argv[2]);dest=HERE/sys.argv[3]
cfg=json.loads((ROOT/'diagnostics/ring_exchange_stage1'/run/'config.json').read_text());plan=json.loads((HERE/'TRAINING_PLAN.json').read_text());item=next(x for x in plan if x['run']==run)
assert hashlib.sha256((ROOT/item['dataset']/'manifest.json').read_bytes()).hexdigest()==item['dataset_manifest_sha256']
kwargs={k:cfg[k] for k in inspect.signature(train_stage1).parameters if k in cfg};kwargs.update(seed=seed)
receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),jax_version=jax.__version__,devices=[str(x) for x in jax.devices()],training_module=inspect.getfile(train_stage1),kwargs=kwargs,prereg_sha256=hashlib.sha256((HERE/'PREREGISTRATION.md').read_bytes()).hexdigest())
(HERE/(sys.argv[3].replace('/','_')+'_START.receipt.json')).write_text(json.dumps(receipt,indent=2))
print(json.dumps(train_stage1(ROOT/item['dataset'],dest,**kwargs)),flush=True)
