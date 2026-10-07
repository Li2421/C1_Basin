#!/usr/bin/env python3
"""Train the frozen simple Direct-eta baseline and freeze TEST predictions."""
import csv,hashlib,json,time
from pathlib import Path
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np,optax

ROOT=Path('/home/zhihan/research/Basin_C1'); HERE=ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1'
LOW=np.array([0.,-.5,0.],np.float32); HIGH=np.array([1.25,.5,.75],np.float32); SCALE=np.array([.75,1.,.75],np.float32)
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class G(nn.Module):
 @nn.compact
 def __call__(self,x):
  x=nn.silu(nn.Dense(128)(x));x=nn.silu(nn.Dense(128)(x));y=nn.sigmoid(nn.Dense(3)(x));return jnp.asarray(LOW)+y*jnp.asarray(HIGH-LOW)
def loss(model,p,x,y):return jnp.mean(jnp.sum(((model.apply(p,x)-y)/jnp.asarray(SCALE))**2,axis=1))
def train(seed,x,y,xv,yv):
 m=G();p=m.init(jax.random.PRNGKey(seed),jnp.asarray(x[:1]));o=optax.adamw(1e-3,weight_decay=1e-5);s=o.init(p);best=None;bv=1e99;be=0;wait=0;hist=[];rng=np.random.default_rng(seed)
 @jax.jit
 def step(p,s,xb,yb):
  v,g=jax.value_and_grad(lambda z:loss(m,z,xb,yb))(p);u,s=o.update(g,s,p);return optax.apply_updates(p,u),s,v
 for e in range(3000):
  order=rng.permutation(len(x));vals=[]
  for b in range(0,len(x),32):p,s,v=step(p,s,jnp.asarray(x[order[b:b+32]]),jnp.asarray(y[order[b:b+32]]));vals.append(float(v))
  v=float(loss(m,p,jnp.asarray(xv),jnp.asarray(yv)));hist.append({'seed':seed,'epoch':e,'train_mse':float(np.mean(vals)),'val_mse':v})
  if v<bv-1e-8:bv=v;be=e;best=jax.tree_util.tree_map(lambda z:np.asarray(z),p);wait=0
  else:wait+=1
  if wait>=180:break
 return m,best,bv,be,hist
def main():
 targets=list(csv.DictReader(open(HERE/'robust_lowj_targets.csv'))); resolved=[r for r in targets if r['target_status']!='TARGET_UNRESOLVED']
 counts={sp:sum(r['split']==sp for r in resolved) for sp in ('train','val','test')}
 if counts['train']<72:raise RuntimeError(('DIRECT_ETA_TARGET_DATA_UNDERRESOLVED',counts))
 states=json.loads((HERE/'eligible_state_manifest.json').read_text())['selected_states']; sm={r['state_id']:r for r in states};h=np.asarray(np.load(HERE/'conditioning_features.npz')['features'],float)
 tridx=[sm[r['state_id']]['feature_index'] for r in resolved if r['split']=='train'];mean=h[tridx].mean(0);std=h[tridx].std(0);std=np.where(std<1e-8,1.,std)
 dump(HERE/'normalization.json',{'h_mean':mean.tolist(),'h_std':std.tolist(),'eta_loss_scales':SCALE.tolist(),'physical_output_low':LOW.tolist(),'physical_output_high':HIGH.tolist(),'note':'envelope includes explicit legal zero plus active OrthoFlow3 box; no snapping/gate','fit_train_only':True})
 def data(sp):
  rr=[r for r in resolved if r['split']==sp];x=np.stack([(h[sm[r['state_id']]['feature_index']]-mean)/std for r in rr]).astype(np.float32);y=np.array([[r['eta1'],r['eta2'],r['eta3']] for r in rr],np.float32);return rr,x,y
 datasets={sp:data(sp) for sp in ('train','val','test')};hist=[];results=[];models={};t0=time.monotonic()
 for seed in (17,23,41):
  m,p,v,e,hh=train(seed,datasets['train'][1],datasets['train'][2],datasets['val'][1],datasets['val'][2]);path=HERE/f'direct_eta_seed{seed}.msgpack';path.write_bytes(serialization.to_bytes(p));models[seed]=(m,p);hist+=hh;results.append({'seed':seed,'best_val_normalized_mse':v,'best_epoch':e,'checkpoint':str(path),'checkpoint_sha256':sha(path)})
 sel=min(results,key=lambda r:(r['best_val_normalized_mse'],r['seed']));m,p=models[sel['seed']]
 with open(HERE/'training_history.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(hist[0]));w.writeheader();w.writerows(hist)
 dump(HERE/'training_seed_results.json',{'seeds':results});dump(HERE/'selected_checkpoint.json',sel)
 params=sum(v.size for v in jax.tree_util.tree_leaves(p));dump(HERE/'model_config.json',{'architecture':'214->128->128->3','activation':'SiLU','output':'sigmoid affine to [0,-.5,0]..[1.25,.5,.75]','parameter_count':int(params),'loss':'normalized eta MSE only','optimizer':'AdamW lr1e-3 wd1e-5','seeds':[17,23,41]})
 # baselines and pointwise frozen TEST output.
 trr,xtr,ytr=datasets['train'];ter,xt,yt=datasets['test'];pred=np.asarray(m.apply(p,jnp.asarray(xt)));const=np.broadcast_to(ytr.mean(0),yt.shape);ridge=np.linalg.solve(xtr.T@xtr+1e-3*np.eye(xtr.shape[1]),xtr.T@ytr);lin=xt@ridge
 rows=[]
 for r,y,yp in zip(ter,yt,pred):rows.append({'state_id':r['state_id'],'split':'test','target_kind':r['target_kind'],'target_status':r['target_status'],'target_eta':json.dumps(y.tolist()),'pred_eta':json.dumps(yp.tolist()),'normalized_l2':float(np.linalg.norm((yp-y)/SCALE)),'physical_l2':float(np.linalg.norm(yp-y)),'pred_eta_norm':float(np.linalg.norm(yp))})
 def metrics(mask):
  a=pred[mask];b=yt[mask];return {'states':int(mask.sum()),'normalized_l2_mean':float(np.mean(np.linalg.norm((a-b)/SCALE,axis=1))),'normalized_l2_median':float(np.median(np.linalg.norm((a-b)/SCALE,axis=1))),'physical_l2_mean':float(np.mean(np.linalg.norm(a-b,axis=1))),'physical_l2_median':float(np.median(np.linalg.norm(a-b,axis=1))),'per_dimension_mae':np.mean(np.abs(a-b),axis=0).tolist()}
 zero=np.array([r['target_kind']=='ZERO' for r in ter]);point={'all':metrics(np.ones(len(ter),bool)),'zero':metrics(zero),'active':metrics(~zero),'constant_normalized_mse':float(np.mean(np.sum(((const-yt)/SCALE)**2,axis=1))),'ridge_normalized_mse':float(np.mean(np.sum(((lin-yt)/SCALE)**2,axis=1)))};dump(HERE/'heldout_pointwise_metrics.json',point)
 with open(HERE/'heldout_zero_outputs.csv','w',newline='') as f:
  fields=list(rows[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([r for r in rows if r['target_kind']=='ZERO'])
 tasks=[]
 for r in rows:
  eta=json.loads(r['pred_eta'])
  for fi in range(64):tasks.append({'task_id':f"pred64__{r['state_id']}__f{fi:02d}",'candidate_id':f"{r['state_id']}__direct_pred",'state_id':r['state_id'],'split':'test','probe_id':'direct_pred','eta':eta,'future_index':fi})
 dump(HERE/'test_prediction_plan.json',{'schema':'direct_eta_test_prediction64_v1','orthoflow3_sha256':'51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38','future_root_seed':2026092702,'conditioning':'exact Q-v2 h conditioning','checkpoint_sha256':sel['checkpoint_sha256'],'tasks':tasks,'maximum_new_continuations':len(tasks)})
 dump(HERE/'frozen_direct_eta_manifest.json',{'checkpoint':sel,'normalization_sha256':sha(HERE/'normalization.json'),'test_predictions':rows,'frozen_before_test_closed_loop':True,'training_seconds':time.monotonic()-t0})
 print(json.dumps({'selected':sel,'pointwise':point,'test_plan':len(tasks),'training_seconds':time.monotonic()-t0},indent=2))
if __name__=='__main__':main()
