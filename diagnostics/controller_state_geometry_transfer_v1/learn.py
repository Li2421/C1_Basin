"""Existing entity critic, source-only normalization, immutable per-seed artifacts."""
from pathlib import Path
import json,time
import numpy as np
SEEDS=[17,23,41]

def fit(D,tr,y,arm,seed,outdir,steps=None):
 import jax,jax.numpy as jnp,optax,flax.linen as nn
 from flax import serialization
 from diagnostics.controller_q_diversity_v1.train import Critic
 outdir=Path(outdir);outdir.mkdir(exist_ok=True,parents=True);name=f'{arm}_{seed}'
 cells=np.indices(D['Y'].shape[:4]).reshape(4,-1).T.astype(int);S=D['Y'].shape[1];C=D['Y'].shape[2]
 if (outdir/(name+'.npz')).exists():
  f=np.load(outdir/(name+'.npz'));assert np.array_equal(f['tr'],tr) and np.array_equal(f['y'],y);return f['pred']
 started=time.monotonic();eta=jnp.asarray(D['eta'],jnp.float32);onehot=jnp.eye(C,dtype=jnp.float32)
 X={k:jnp.asarray(v,jnp.float32) for k,v in D['X'].items()};na=X['agents'].shape[1]
 if arm in ('ENT_raw','ENT_mech','ENT_mech_no_margin','ENT_mech_phys'):
  cols=(D['raw'] if arm=='ENT_raw' else D['mech']).copy()
  if arm=='ENT_mech_no_margin':cols[:,1]=0.
  train_g=tr[:,0]*S+tr[:,1];mu=cols[train_g].mean(0);sd=np.maximum(cols[train_g].std(0),.01)
  if arm=='ENT_mech_phys':sd=np.array([1.,60.,1.,1.,1.,4.,10.,1.])
  X['globals']=jnp.concatenate([X['globals'],jnp.asarray((cols-mu)/sd,jnp.float32)],-1)
 if arm in ('MLP','MECH'):
  cols=D['flat'] if arm=='MLP' else D['mech'];g=tr[:,0]*S+tr[:,1];mu=cols[g].mean(0);sd=np.maximum(cols[g].std(0),.01);cont=jnp.asarray((cols-mu)/sd,jnp.float32)
  class MLP(nn.Module):
   @nn.compact
   def __call__(self,x,c,e):
    z=jnp.concatenate([x,c,e],-1)
    for _ in range(3):z=nn.silu(nn.Dense(128)(z))
    return nn.Dense(1)(z)[:,0]
  net=MLP()
  def inputs(ix):return cont[ix[:,0]*S+ix[:,1]],onehot[ix[:,2]],eta[ix[:,3]]
  nsteps=steps or 3000;batch=256;lr=1e-3
 else:
  net=Critic('state_field_compact')
  def inputs(ix):
   g=ix[:,0]*S+ix[:,1];x={k:v[g] for k,v in X.items()};c=jnp.broadcast_to(onehot[ix[:,2]][:,None,:],(len(g),na,C))*x['agent_mask'][...,None]
   if arm=='STATE':c=jnp.zeros_like(c)
   return x,c,eta[ix[:,3]]
  nsteps=steps or 6000;batch=120;lr=8e-4
 emp=np.zeros(D['Y'].shape[:4]);counts=np.zeros_like(emp)
 np.add.at(emp,tuple(tr.T),y);np.add.at(counts,tuple(tr.T),1);emp=jnp.asarray(emp/np.maximum(counts,1),jnp.float32)
 tj=jnp.asarray(tr,jnp.int32);yy=jnp.asarray(y,jnp.float32);params=net.init(jax.random.PRNGKey(seed),*inputs(tj[:2]));opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(lr,weight_decay=1e-4));ost=opt.init(params)
 @jax.jit
 def block(params,ost,draws):
  def step(carry,idx):
   pp,oo=carry
   def objective(par):
    batchix=tj[idx];z=net.apply(par,*inputs(batchix));loss=optax.sigmoid_binary_cross_entropy(z,yy[idx]).mean()
    if arm=='RANK':
     other=batchix.at[:,3].set((batchix[:,3]+1+idx%(D['Y'].shape[3]-1))%D['Y'].shape[3]);z2=net.apply(par,*inputs(other));delta=emp[tuple(batchix.T)]-emp[tuple(other.T)]
     loss=loss+jnp.mean(jnp.abs(delta)*optax.sigmoid_binary_cross_entropy(z-z2,(1+delta)/2))
    return loss
   loss,grad=jax.value_and_grad(objective)(pp)
   updates,oo=opt.update(grad,oo,pp);return (optax.apply_updates(pp,updates),oo),loss
  return jax.lax.scan(step,(params,ost),draws)
 rng=np.random.default_rng(seed);losses=[]
 for start in range(0,nsteps,500):
  (params,ost),ll=block(params,ost,jnp.asarray(rng.integers(len(y),size=(min(500,nsteps-start),batch)),jnp.int32));losses.append(float(np.mean(ll)))
 predict=jax.jit(lambda p,ix:jax.nn.sigmoid(net.apply(p,*inputs(ix))))
 pred=np.concatenate([np.asarray(predict(params,jnp.asarray(cells[i:i+512],jnp.int32))) for i in range(0,len(cells),512)]).reshape(D['Y'].shape[:4])
 (outdir/(name+'.msgpack')).write_bytes(serialization.to_bytes(params));np.savez_compressed(outdir/(name+'.npz'),pred=pred,tr=tr,y=y,losses=losses)
 (outdir/(name+'.json')).write_text(json.dumps({'arm':arm,'seed':seed,'steps':nsteps,'batch':batch,'lr':lr,'weight_decay':1e-4,'n_labels':len(y),'seconds':time.monotonic()-started,'losses':losses,'devices':[str(x) for x in jax.devices()]},indent=2)+'\n')
 print(name,len(y),losses[-1],time.monotonic()-started,flush=True);return pred

def rows(D,spec,allow_unknown=False):
 ix=[];ys=[];seeds=[];unknown=[]
 for g,states,rands in spec:
  for s in states:
   for c in range(D['Y'].shape[2]):
    for e in range(D['Y'].shape[3]):
     for r in rands:
      v=D['Y'][g,s,c,e,r]
      if not np.isfinite(v):unknown.append([g,s,c,e,r]);continue
      ix.append([g,s,c,e]);ys.append(v);seeds.append(r)
 if not allow_unknown:assert not unknown,('missing/unknown training labels',unknown[:3],len(unknown))
 elif unknown:print('Retained unknown rows omitted from Bernoulli likelihood:',len(unknown),flush=True)
 return np.asarray(ix,int),np.asarray(ys,float)

def prior(tr,y,C,E):return np.asarray([[np.mean(y[(tr[:,2]==c)&(tr[:,3]==e)]) for e in range(E)] for c in range(C)])
