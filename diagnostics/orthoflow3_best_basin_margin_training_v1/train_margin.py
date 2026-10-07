#!/usr/bin/env python3
"""Train exactly the frozen set-valued margin model; no point-MSE term."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,time
from pathlib import Path
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np
import optax
ROOT=Path('/home/zhihan/research/Basin_C1');D=ROOT/'diagnostics';H=Path(__file__).parent;POINT=D/'orthoflow3_true_t0_point_learning_v1';SHARED=D/'orthoflow3_shared_conservative_family_v1'
HS=np.unique(np.array(json.load(open(D/'orthoflow3_t0_multiball_basin_learning_v1/geometry_constants.json'))['halfspaces']),axis=0)
SEEDS=(17,23,41)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,x):
 p=H/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def write(n,rows,fields=None):
 p=H/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),extrasaction='ignore');w.writeheader();w.writerows(rows)
class G(nn.Module):
 @nn.compact
 def __call__(self,x):x=nn.silu(nn.Dense(128)(x));x=nn.silu(nn.Dense(128)(x));return nn.Dense(3)(x)
def prepare_labels():
 # Fresh validation is descriptive; shrink globally if it observed any failure.
 fresh=json.load(open(H/'fresh_retained_summary.json'));models=json.load(open(H/'fitted_models.json'));split=json.load(open(POINT/'final_source_split.json'));gamma_ret=.75 if fresh['non_B63']==0 else .50;gamma_inner=.8*gamma_ret
 labels=[];params=[]
 for st in split['states']:
  sid=st['state_id']
  if sid in models:m=json.loads(json.dumps(models[sid]))
  else:
   # TEST geometry is deliberately quarantined until after controller freeze.
   # This placeholder is never indexed by the TRAIN/VAL loss.
   target=next(r for r in __import__('csv').DictReader(open(POINT/'selected_eta_targets.csv')) if r['state_id']==sid)
   a=(np.array([float(target[f'target_eta{i}']) for i in (1,2,3)])-np.array([.875,0.,.375]))/np.array([.75,1.,.75])
   m=dict(family='superbody_with_cuts',p=4,center=a.tolist(),R=np.eye(3).tolist(),axes=[.1,.1,.1],anchor=a.tolist(),cuts=[])
  m['gamma']=gamma_inner
  cuts=m.get('cuts',[]);q=np.zeros((3,3));b=np.zeros(3);mask=np.zeros(3)
  for j,c in enumerate(cuts):q[j]=c['q'];b[j]=c['b'];mask[j]=1
  params.append(dict(center=m['center'],R=m['R'],axes=m['axes'],anchor=m['anchor'],gamma=m['gamma'],p=m['p'],q=q.tolist(),b=b.tolist(),cut_mask=mask.tolist()))
  labels.append(dict(state_id=sid,split=st['split'],source_group=st['source_group'],family=m['family'],p=m['p'],gamma_retained=gamma_ret,gamma_inner_relative=.8,gamma_inner=gamma_inner,center=m['center'],R=m['R'],axes=m['axes'],anchor=m['anchor'],cuts=cuts,quality_class=fresh['quality'],small_margin_state=False,test_placeholder=sid not in models))
 dump('train_set_labels.json',[x for x in labels if x['split']=='train']);dump('val_set_labels.json',[x for x in labels if x['split']=='val']);dump('all_set_labels.json',labels);np.savez_compressed(H/'margin_label_params.npz',center=np.array([x['center'] for x in params],np.float32),R=np.array([x['R'] for x in params],np.float32),axes=np.array([x['axes'] for x in params],np.float32),anchor=np.array([x['anchor'] for x in params],np.float32),gamma=np.array([x['gamma'] for x in params],np.float32),p=np.array([x['p'] for x in params],np.float32),q=np.array([x['q'] for x in params],np.float32),b=np.array([x['b'] for x in params],np.float32),cut_mask=np.array([x['cut_mask'] for x in params],np.float32))
 a=np.load(POINT/'point_learning_arrays.npz');np.savez_compressed(H/'margin_arrays.npz',x=a['x'],splits=a['splits'],state_ids=a['state_ids'],physical_targets=a['physical_target'])
 config=dict(architecture=[214,128,128,3],activation='SiLU',seeds=list(SEEDS),optimizer='AdamW',learning_rate=.001,weight_decay=1e-5,batch_size=24,max_epochs=4000,early_stop_patience=250,loss='only normalized analytic inner-set violation squared',no_point_MSE=True,no_auxiliary_losses=True,eta_coordinates='frozen normalized E_bridge coordinates',selected_family=json.load(open(H/'selected_family.json'))['family'],gamma_retained=gamma_ret,gamma_inner=gamma_inner,quality_class=fresh['quality'])
 dump('common_model_config.json',config);dump('dataset_manifest.json',dict(source_dataset=str(POINT/'dataset_manifest.json'),source_hash=sha(POINT/'dataset_manifest.json'),set_label_hash=sha(H/'all_set_labels.json'),param_hash=sha(H/'margin_label_params.npz'),source_leakage=False,split_counts={s:int(np.sum(a['splits']==s)) for s in ('train','val','test')},frozen_before_training=True))
 print(json.dumps({'gamma_retained':gamma_ret,'gamma_inner':gamma_inner,'quality':fresh['quality']}))
def loss_fn(model,p,xb,idx,lab):
 z=model.apply(p,xb);c=lab['center'][idx];R=lab['R'][idx];axes=lab['axes'][idx];anchor=lab['anchor'][idx];g=lab['gamma'][idx,None];pre=anchor+(z-anchor)/g
 u=jnp.einsum('bi,bij->bj',pre-c,R)/axes
 # Rooted p-norm excess preserves the exact zero set but keeps narrow-axis
 # gradients numerically normalized.
 vs=[jnp.maximum(jnp.sum(jnp.abs(u)**lab['p'][idx,None],axis=1)**(1.0/lab['p'][idx])-1,0)]
 # E_bridge is part of every retained region.
 vs.extend([jnp.maximum(pre@jnp.asarray(HS[k,:3])+HS[k,3],0) for k in range(len(HS))])
 q=lab['q'][idx];b=lab['b'][idx];mask=lab['cut_mask'][idx]
 cv=jnp.einsum('bki,bi->bk',q,u)-b
 vs.append(jnp.maximum(cv,0)*mask)
 V=jnp.sqrt(jnp.sum(jnp.stack([jnp.sum(x*x,axis=1) if x.ndim==2 else x*x for x in vs],axis=1)+1e-16,axis=1))
 return jnp.mean(V*V),z,V
def train(seed):
 a=np.load(H/'margin_arrays.npz');lab0=np.load(H/'margin_label_params.npz');lab={k:jnp.asarray(lab0[k]) for k in lab0.files};x=a['x'];sp=a['splits'];tr=np.where(sp=='train')[0];va=np.where(sp=='val')[0];model=G();p=model.init(jax.random.PRNGKey(seed),jnp.zeros((1,214)));opt=optax.chain(optax.clip_by_global_norm(1.0),optax.adamw(.001,weight_decay=1e-5));ost=opt.init(p);rng=np.random.default_rng(seed)
 @jax.jit
 def step(p,ost,xb,ii):
  v,grad=jax.value_and_grad(lambda pp:loss_fn(model,pp,xb,ii,lab)[0])(p);upd,ost=opt.update(grad,ost,p);return optax.apply_updates(p,upd),ost,v
 @jax.jit
 def ev(p,xb,ii):return loss_fn(model,p,xb,ii,lab)
 best=None;bv=np.inf;be=0;wait=0;hist=[];start=time.time()
 for epoch in range(4000):
  ii=tr[rng.permutation(len(tr))];p,ost,_=step(p,ost,jnp.asarray(x[ii]),jnp.asarray(ii));lt,pt,vt=ev(p,jnp.asarray(x[tr]),jnp.asarray(tr));lv,pv,vv=ev(p,jnp.asarray(x[va]),jnp.asarray(va))
  hist.append(dict(seed=seed,epoch=epoch,train_loss=float(lt),val_loss=float(lv),train_inner_membership=float(np.mean(np.asarray(vt)<=1e-7)),val_inner_membership=float(np.mean(np.asarray(vv)<=1e-7)),prediction_variance=float(np.mean(np.var(np.asarray(pt),axis=0)))))
  if float(lv)<bv-1e-10:bv=float(lv);be=epoch;best=jax.tree_util.tree_map(lambda z:np.asarray(z),p);wait=0
  else:wait+=1
  if wait>=250:break
 d=H/f'seed{seed}';d.mkdir(exist_ok=True);ck=d/'checkpoint.msgpack';ck.write_bytes(serialization.to_bytes(best));write(f'seed{seed}/training_history.csv',hist);dump(f'seed{seed}/summary.json',dict(seed=seed,best_epoch=be,best_val_loss=bv,epochs=len(hist),training_seconds=time.time()-start,checkpoint=str(ck),checkpoint_sha256=sha(ck),train_inner_membership=hist[be]['train_inner_membership'],val_inner_membership=hist[be]['val_inner_membership'],prediction_variance=hist[be]['prediction_variance']));print(json.dumps(json.load(open(d/'summary.json'))))
def aggregate():
 rows=[json.load(open(H/f'seed{s}/summary.json')) for s in SEEDS];write('training_summary.csv',rows);dump('retained_checkpoints.json',{'selection':'minimum VAL analytic inner-set margin loss before true VAL closed loop','seeds':rows});print(json.dumps({'seeds':len(rows)}))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','train','aggregate']);ap.add_argument('--seed',type=int);a=ap.parse_args();{'prepare':prepare_labels,'train':lambda:train(a.seed),'aggregate':aggregate}[a.stage]()
if __name__=='__main__':main()
