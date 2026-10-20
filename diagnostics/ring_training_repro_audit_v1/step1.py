import os,sys,json,inspect,hashlib,datetime,pickle
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
(HERE/'tmp').mkdir(exist_ok=True);os.environ['TMPDIR']=str(HERE/'tmp');os.environ['JAX_PLATFORMS']='cpu'
HIST=ROOT/'diagnostics/ring_independent_flow_confirmation_v2/git_43b5264';sys.path.insert(0,str(HIST))
import numpy as np,jax
from new_benchmark_common import training
from new_benchmark_common.macflow import load_checkpoint
run=sys.argv[1] if len(sys.argv)>1 else 'base_u_v2_macflow';old=ROOT/'diagnostics/ring_exchange_stage1'/run;cfg=json.loads((old/'config.json').read_text());dataset=ROOT/'diagnostics/ring_exchange_stage1'/run.replace('_macflow','_dataset');rows=[json.loads(l) for l in (old/'metrics.jsonl').read_text().splitlines()];checkpoint=pickle.loads((old/'best.pkl').read_bytes());kwargs={k:cfg[k] for k in inspect.signature(training.train_stage1).parameters if k in cfg};kwargs['actor_hidden_dims']=tuple(checkpoint['config']['actor_hidden_dims']);kwargs['steps']=cfg['steps']
# Copy only the exact historical initialization prefix; exit before any update.
source=inspect.getsource(training.train_stage1);prefix=source[:source.index('    run = {')];prefix+='    return agent, train, dev, initial_train, initial_dev, fixed_train, fixed_dev\n';ns=dict(training.__dict__);exec(prefix,ns)
dest=HERE/(run+'_initial');agent,train,dev,tl,dl,ft,fd=ns['train_stage1'](dataset,dest,**kwargs)
arrays={};comparisons={}
for k in ('obs_mean','obs_scale','act_mean','act_scale'):
 current=np.asarray(agent.config[k]);reference=np.asarray(checkpoint['config'][k]);arrays[k]=current;arrays['original_'+k]=reference;comparisons[k]={'same_bytes':current.dtype==reference.dtype and current.tobytes()==reference.tobytes(),'max_abs_difference':float(np.max(abs(current-reference))),'dtype':str(current.dtype),'original_dtype':str(reference.dtype)}
params=jax.tree_util.tree_leaves(agent.network.params);ph=hashlib.sha256(b''.join(np.asarray(x).tobytes() for x in params)).hexdigest();np.savez_compressed(HERE/(run+'_normalization.npz'),**arrays)
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':run,'historical_module':training.__file__,'historical_module_sha256':hashlib.sha256(Path(training.__file__).read_bytes()).hexdigest(),'original_step0':rows[0],'current_step0':{'fixed_train_loss':tl,'fixed_dev_loss':dl},'relative_dev_difference':(dl-rows[0]['fixed_dev_loss'])/rows[0]['fixed_dev_loss'],'relative_train_difference':(tl-rows[0]['fixed_train_loss'])/rows[0]['fixed_train_loss'],'normalization':comparisons,'initial_parameter_hash':ph,'original_initial_parameter_hash_available':False,'train_samples':len(train.observations),'dev_samples':len(dev.observations),'jax_x64':bool(jax.config.x64_enabled),'devices':[str(x) for x in jax.devices()]};(HERE/(run+'_STEP1.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if abs(result['relative_dev_difference'])<=1e-7 and abs(result['relative_train_difference'])<=1e-7:
 kwargs['steps']=1000;kwargs['log_interval']=250;out=training.train_stage1(dataset,HERE/(run+'_step1000'),**kwargs);original1000=next((r for r in rows if r['step']==1000),None);(HERE/(run+'_STEP1000.json')).write_text(json.dumps({'summary':out,'original1000':original1000},indent=2)+'\n');print('step1000 complete',out,flush=True)
