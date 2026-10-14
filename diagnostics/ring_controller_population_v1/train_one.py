import os,sys,json,inspect,hashlib,datetime,pickle
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['TMPDIR']=str(HERE/'tmp');(HERE/'tmp').mkdir(exist_ok=True)
R14=ROOT/'diagnostics/ring_independent_flow_confirmation_v2'
audit=json.loads((R14/'HISTORICAL_CODE_AUDIT.json').read_text())
assert all(x['equal'] and hashlib.sha256((R14/audit['chosen']/x['file']).read_bytes()).hexdigest()==x['git_sha256'] for x in audit['files'])
reg=json.loads((HERE/'PREREGISTRATION.receipt.json').read_text());assert hashlib.sha256((HERE/'PREREGISTRATION.md').read_bytes()).hexdigest()==reg['sha256']
sys.path.insert(0,str(R14/audit['chosen']))
import jax
from new_benchmark_common.training import train_stage1
plan=json.loads((HERE/'TRAINING_PLAN.json').read_text());item=plan[int(sys.argv[1])];run=item['run'];seed=item['new_seed'];dest=HERE/'checkpoints'/run/str(seed)
cfg=json.loads((ROOT/'diagnostics/ring_exchange_stage1'/run/'config.json').read_text())
assert hashlib.sha256((ROOT/item['dataset']/'manifest.json').read_bytes()).hexdigest()==item['dataset_manifest_sha256']
kwargs={k:cfg[k] for k in inspect.signature(train_stage1).parameters if k in cfg};kwargs.update(seed=seed)
original=pickle.loads((ROOT/item['original_best']).read_bytes());kwargs['actor_hidden_dims']=tuple(original['config']['actor_hidden_dims'])
assert str(jax.devices()[0]).startswith('cuda')
receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),jax_version=jax.__version__,devices=[str(x) for x in jax.devices()],historical_package=str(R14/audit['chosen']),training_module=inspect.getfile(train_stage1),kwargs=kwargs,prereg_sha256=reg['sha256'])
(HERE/'training_receipts').mkdir(parents=True,exist_ok=True);(HERE/'training_receipts'/f'{run}_{seed}.json').write_text(json.dumps(receipt,indent=2)+'\n')
result=train_stage1(ROOT/item['dataset'],dest,**kwargs)
(dest/'ROUND15_TRAINING_RESULT.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
print(json.dumps(dict(run=run,seed=seed,dest=str(dest),result=result),default=str),flush=True)
