import os,sys,json,inspect,hashlib,datetime,pickle
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['JAX_PLATFORMS']='cpu';sys.path.insert(0,str(ROOT/'diagnostics/ring_independent_flow_confirmation_v2/git_43b5264'))
import numpy as np,jax
from new_benchmark_common import training
old=ROOT/'diagnostics/ring_exchange_stage1/base_u_v2_macflow';cfg=json.loads((old/'config.json').read_text());cp=pickle.loads((old/'best.pkl').read_bytes());source=inspect.getsource(training.train_stage1);prefix=source[:source.index('    run = {')]+'    return agent, fixed_train, fixed_dev\n';ns=dict(training.__dict__);exec(prefix,ns);agent,ft,fd=ns['train_stage1'](ROOT/'diagnostics/ring_exchange_stage1/base_u_v2_dataset',HERE/'initial_batch_grid',seed=cfg['seed'],steps=cfg['steps'],batch_size=cfg['batch_size'],validation_batches=64,actor_hidden_dims=tuple(cp['config']['actor_hidden_dims']))
loss={}
for name,batches,salt in [('train',ft,1001),('dev',fd,1002)]:loss[name]=[float(agent.total_loss(batch,agent.network.params,rng=jax.random.fold_in(jax.random.PRNGKey(salt),i))[0]) for i,batch in enumerate(batches)]
original=json.loads((old/'metrics.jsonl').read_text().splitlines()[0]);out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'numpy':np.__version__,'batch_losses':loss,'prefix_means':{str(n):{k:float(np.mean(v[:n])) for k,v in loss.items()} for n in (1,2,4,8,16,32,64)},'original':original};(HERE/'VALIDATION_BATCH_GRID.json').write_text(json.dumps(out,indent=2)+'\n');print(out['prefix_means'],flush=True)
