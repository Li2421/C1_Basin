import os,sys,json,inspect,pickle,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['JAX_PLATFORMS']='cpu';sys.path.insert(0,str(ROOT/'diagnostics/ring_independent_flow_confirmation_v2/git_43b5264'))
from new_benchmark_common import training
import numpy as np,jax
plan=json.loads((ROOT/'diagnostics/ring_independent_flow_confirmation_v2/TRAINING_PLAN.json').read_text())
for run in ('base_u_v2_macflow','base_u_v9_local_macflow'):
 item=next(x for x in plan if x['run']==run);old=ROOT/'diagnostics/ring_exchange_stage1'/run;cfg=json.loads((old/'config.json').read_text());cp=pickle.loads((old/'best.pkl').read_bytes());kwargs={k:cfg[k] for k in inspect.signature(training.train_stage1).parameters if k in cfg};kwargs['actor_hidden_dims']=tuple(cp['config']['actor_hidden_dims']);kwargs['validation_batches']=4;kwargs['log_interval']=1000 if run=='base_u_v2_macflow' else 250;kwargs['steps']=12000 if run=='base_u_v2_macflow' else 1000
 out=training.train_stage1(ROOT/item['dataset'],HERE/(run+'_validation4'),**kwargs);orig=json.loads((old/'training_summary.json').read_text());metrics=[json.loads(l) for l in (old/'metrics.jsonl').read_text().splitlines()];newmetrics=[json.loads(l) for l in (HERE/(run+'_validation4')/'metrics.jsonl').read_text().splitlines()];comparisons=[]
 for r in newmetrics:
  match=next((m for m in metrics if m['step']==r['step']),None)
  if match:comparisons.append(dict(step=r['step'],new=r,original=match,dev_relative_error=abs(r['fixed_dev_loss']-match['fixed_dev_loss'])/abs(match['fixed_dev_loss']),train_relative_error=abs(r['fixed_train_loss']-match['fixed_train_loss'])/abs(match['fixed_train_loss'])))
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=run,kwargs=kwargs,dataset=item['dataset'],summary=out,original_summary=orig,comparisons=comparisons,final_relative_error=abs(out['final_fixed_dev_loss']-orig['final_fixed_dev_loss'])/abs(orig['final_fixed_dev_loss']) if kwargs['steps']==cfg['steps'] else None,best_relative_error=abs(out['best_fixed_dev_loss']-orig['best_fixed_dev_loss'])/abs(orig['best_fixed_dev_loss']) if kwargs['steps']==cfg['steps'] else None)
 (HERE/(run+'_VALIDATION4_RESULT.json')).write_text(json.dumps(result,indent=2)+'\n');print(run,result,flush=True)
