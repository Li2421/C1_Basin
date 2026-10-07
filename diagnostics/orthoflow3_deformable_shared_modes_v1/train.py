#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,math,time
from pathlib import Path
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp
import numpy as np
import optax
H=Path(__file__).parent;SEEDS=(17,23,41);TAU=.0025;MU=.05
def write(name,rows):
 with (H/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(name,x):(H/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Residual(nn.Module):
 @nn.compact
 def __call__(self,x,m):
  e=nn.Embed(12,8)(m);z=jnp.concatenate([x,e],axis=-1);z=nn.silu(nn.Dense(64)(z));z=nn.silu(nn.Dense(64)(z));return nn.Dense(3)(z)
def bounded(raw,anchor,A,b):
 r=jnp.linalg.norm(raw,axis=-1,keepdims=True);d=.20*jnp.tanh(r)*raw/jnp.maximum(r,1e-8);den=jnp.einsum('kj,bj->bk',A,d);num=-(jnp.einsum('kj,bj->bk',A,anchor)+b);cand=jnp.where(den>1e-9,num/den,jnp.inf);t=jnp.minimum(1.,jnp.min(cand,axis=1));t=jnp.clip(.999999*t,0.,1.);return d*t[:,None]
def loss_and_aux(model,p,batch,A,b):
 x,m,a,P,PM,N,NM=batch;raw=model.apply(p,x,m);d=bounded(raw,a,A,b);pred=a+d;d2=jnp.sum((pred[:,None,:]-P)**2,axis=-1);lse=jax.scipy.special.logsumexp(jnp.where(PM>0,-d2/TAU,-jnp.inf),axis=1)-jnp.log(jnp.sum(PM,axis=1));lp=-TAU*lse;nd2=jnp.min(jnp.where(NM>0,jnp.sum((pred[:,None,:]-N)**2,axis=-1),jnp.inf),axis=1);has=jnp.sum(NM,axis=1)>0;ln=jnp.where(has,jnp.maximum(0.,MU-jnp.sqrt(jnp.maximum(nd2,1e-12)))**2,0.);return jnp.mean(lp+ln),(jnp.mean(lp),jnp.mean(ln),pred,d)
def metrics(model,p,data,mask,A,b):
 z=[jnp.asarray(q[mask]) for q in data];val,aux=loss_and_aux(model,p,z,jnp.asarray(A),jnp.asarray(b));pred=np.asarray(aux[2]);P=np.asarray(z[3]);PM=np.asarray(z[4]);dist=np.sqrt(np.min(np.where(PM>0,np.sum((pred[:,None,:]-P)**2,axis=-1),np.inf),axis=1));d=np.asarray(aux[3]);return dict(loss=float(val),local_set_distance=float(dist.mean()),membership_proxy=float(np.mean(dist<=.05)),prediction_variance=float(np.mean(np.var(d,axis=0))),residual_norm_mean=float(np.mean(np.linalg.norm(d,axis=1))),residual_norm_max=float(np.max(np.linalg.norm(d,axis=1)))) ,pred,d,dist
def train(seed):
 z=np.load(H/'local_sets.npz');data=[z[k] for k in ('x','modes','anchors','P','Pmask','N','Nmask')];tr=z['splits']=='train';va=z['splits']=='val';A=z['halfspace_A'];b=z['halfspace_b'];model=Residual();p=model.init(jax.random.PRNGKey(seed),jnp.zeros((1,214)),jnp.zeros((1,),jnp.int32));opt=optax.adamw(1e-3,weight_decay=1e-4);os=opt.init(p)
 @jax.jit
 def step(p,os,batch):
  def objective(params):return loss_and_aux(model,params,batch,jnp.asarray(A),jnp.asarray(b))
  (v,_),g=jax.value_and_grad(objective,has_aux=True)(p);u,os=opt.update(g,os,p);return optax.apply_updates(p,u),os,v
 rng=np.random.default_rng(seed);best=None;bv=math.inf;wait=0;hist=[];start=time.time();ti=np.flatnonzero(tr)
 for ep in range(2200):
  order=rng.permutation(ti)
  for j in range(0,len(order),32):idx=order[j:j+32];p,os,_=step(p,os,[jnp.asarray(q[idx]) for q in data])
  mt,_,_,_=metrics(model,p,data,tr,A,b);mv,_,_,_=metrics(model,p,data,va,A,b);hist.append(dict(epoch=ep,train_loss=mt['loss'],val_loss=mv['loss'],train_distance=mt['local_set_distance'],val_distance=mv['local_set_distance'],train_membership=mt['membership_proxy'],val_membership=mv['membership_proxy']))
  if mv['loss']<bv-1e-7:bv=mv['loss'];best=jax.tree_util.tree_map(lambda x:np.array(x),p);be=ep;wait=0
  else:wait+=1
  if wait>=220:break
 name=f'seed{seed}';d=H/name;d.mkdir(exist_ok=True);ck=d/'checkpoint.msgpack';ck.write_bytes(serialization.to_bytes(best));write(f'{name}/training_history.csv',hist);M={}
 for split,mask in [('train',tr),('val',va)]:M[split],pred,delta,dist=metrics(model,best,data,mask,A,b);np.savez_compressed(d/f'{split}_predictions.npz',state_ids=z['state_ids'][mask],modes=z['modes'][mask],pred_z=pred,delta_z=delta,distance=dist)
 s=dict(seed=seed,best_epoch=be,epochs=len(hist),best_val_loss=bv,training_seconds=time.time()-start,checkpoint=str(ck),checkpoint_sha256=sha(ck),metrics=M);dump(f'{name}/summary.json',s);print(json.dumps(s))
def aggregate():
 s=[json.load(open(H/f'seed{x}/summary.json')) for x in SEEDS];win=min(s,key=lambda x:(x['metrics']['val']['loss'],x['seed']));rows=[]
 for q in s:
  for split,m in q['metrics'].items():rows.append(dict(seed=q['seed'],split=split,**m,best_epoch=q['best_epoch']))
 write('training_summary.csv',rows);dump('selected_checkpoint.json',win)
 # Emit exact VAL predictions and per-mode residual distribution.
 z=np.load(H/f"seed{win['seed']}/val_predictions.npz");rows=[]
 for sid,m,p,d,di in zip(z['state_ids'],z['modes'],z['pred_z'],z['delta_z'],z['distance']):rows.append(dict(state_id=str(sid),mode_id=int(m),pred_z1=p[0],pred_z2=p[1],pred_z3=p[2],delta_z1=d[0],delta_z2=d[1],delta_z3=d[2],residual_norm=np.linalg.norm(d),local_set_distance=di,membership_proxy=di<=.05))
 write('val_predictions.csv',rows);print(json.dumps({'selected_seed':win['seed'],'val':win['metrics']['val']}))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['train','aggregate']);ap.add_argument('--seed',type=int);a=ap.parse_args();train(a.seed) if a.stage=='train' else aggregate()
if __name__=='__main__':main()
