"""Train one from-scratch seed on the frozen k1-augmented dataset."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1'); HERE=ROOT/'diagnostics/gphi_dagger_k1_diagnostic_v1'; DATA=ROOT/'diagnostics/gphi_training_dataset_dagger_k1_v1'; BASE=ROOT/'diagnostics/gphi_training_dataset_startup_complete_v1'
EXPECTED='83dad86c9eb18147dcc348071034f0522655f188671354fa8d744e6c688df547'; V3=18816; STARTUP_BASE=26432; COVERAGE_BASE=27136; TOTAL=27840
sys.path.insert(0,str(ROOT)); import diagnostics.gphi_pilot_training_v2.train_and_evaluate as core
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def cohort(pred,target,sid,mask):
 v=core.subset_metrics(pred[mask],target[mask],sid[mask]); return v['state_grouped_mean_l2'],v['state_grouped_mse']
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,choices=(17,23,41),required=True); a=ap.parse_args(); out=HERE/f'seed{a.seed}'
 if out.exists(): raise SystemExit(f'existing {out}')
 out.mkdir(); (out/'checkpoints').mkdir()
 if sha(DATA/'samples.npz')!=EXPECTED: raise RuntimeError('dataset hash')
 with np.load(DATA/'samples.npz',allow_pickle=False) as z: x={k:np.asarray(z[k]).copy() for k in z.files}
 if len(x['features'])!=TOTAL or set(x['split'][COVERAGE_BASE:])!={'train'}: raise RuntimeError('dataset shape/split')
 schema=json.load(open(BASE/'feature_schema.json')); train=np.flatnonzero(x['split']=='train'); val=np.flatnonzero(x['split']=='validation')
 mean,scale,binary=core.fit_normalization(x['features'][train],schema); xn=core.normalize(x['features'],mean,scale).astype(np.float32); y=x['targets'].astype(np.float32); sid=x['state_id']
 startup=(val>=V3)&(val<STARTUP_BASE); warm=val<V3
 import jax,jax.numpy as jnp,optax
 cfg=json.load(open(HERE/'config.json')); params=core.init_mlp(jax,[214,128,128,4],a.seed); opt=optax.adamw(cfg['learning_rate'],weight_decay=cfg['weight_decay']); ost=opt.init(params)
 @jax.jit
 def update(p,s,xb,yb):
  def loss(q): return jnp.mean((core.mlp_apply(jax,q,xb)-yb)**2)
  l,g=jax.value_and_grad(loss)(p); u,s=opt.update(g,s,p); return optax.apply_updates(p,u),s,l
 @jax.jit
 def predict(p,x): return core.mlp_apply(jax,p,x)
 xt=jnp.asarray(xn[train]); yt=jnp.asarray(y[train]); xv=jnp.asarray(xn[val]); rng=np.random.default_rng(a.seed); rows=[]; started=time.monotonic()
 modelcfg={'name':f'dagger_k1_seed{a.seed}','seed':a.seed,'hidden':[128,128],'learning_rate':cfg['learning_rate'],'weight_decay':cfg['weight_decay'],'batch_size':cfg['batch_size'],'max_epochs':1200}
 for epoch in range(1,1201):
  order=rng.permutation(len(train)); losses=[]
  for b in range(0,len(order),cfg['batch_size']):
   ix=order[b:b+cfg['batch_size']]; params,ost,l=update(params,ost,xt[ix],yt[ix]); losses.append(float(l))
  pred=np.asarray(predict(params,xv)); sm,ss=cohort(pred,y[val],sid[val],startup); wm,ws=cohort(pred,y[val],sid[val],warm); am,aas=cohort(pred,y[val],sid[val],np.ones(len(val),bool))
  rows.append({'seed':a.seed,'epoch':epoch,'training_minibatch_mse':float(np.mean(losses)),'startup_validation_state_grouped_mean_l2':sm,'startup_validation_state_grouped_mse':ss,'warm_validation_state_grouped_mean_l2':wm,'warm_validation_state_grouped_mse':ws,'all_validation_state_grouped_mean_l2':am,'all_validation_state_grouped_mse':aas,'elapsed_s':time.monotonic()-started})
  core.save_checkpoint(out/'checkpoints'/f'epoch_{epoch:04d}.npz',core.tree_to_numpy(jax,params),modelcfg,mean,scale,binary)
  if epoch==1 or epoch%10==0 or epoch==1200:
   with open(out/'epoch_metrics.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
   (out/'progress.json').write_text(json.dumps({'status':'COMPLETED' if epoch==1200 else 'IN_PROGRESS','seed':a.seed,'epoch':epoch,'elapsed_s':time.monotonic()-started},indent=2)+'\n')
 (out/'runtime.json').write_text(json.dumps({'seed':a.seed,'wall_seconds':time.monotonic()-started,'device':str(jax.devices()[0]),'train_samples':len(train),'validation_samples':len(val)},indent=2)+'\n')
if __name__=='__main__': main()
