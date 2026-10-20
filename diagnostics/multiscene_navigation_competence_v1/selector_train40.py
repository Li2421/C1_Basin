from common import *
from selector_features40 import fit_transform,apply_transform
import flax.linen as nn
from flax import serialization
import jax,jax.numpy as jnp,optax
class Critic(nn.Module):
 @nn.compact
 def __call__(self,x):
  for width in [64,64,16]:x=nn.silu(nn.Dense(width)(x))
  return nn.Dense(1)(x)[...,0]

def fit(X,y,n,trainmask,heldmask,seed,folder,targets):
 cfg=read(HERE/'SELECTOR40_CONFIG_FROZEN.json')
 if (folder/'fit.json').exists():
  old=read(folder/'fit.json');assert old['seed']==seed and old['targets']==targets and old['train_rows']==int(trainmask.sum()) and old['held_rows']==int(heldmask.sum());assert all((folder/f'step{t}.msgpack').exists() for t in targets);tr0=fit_transform(X[trainmask]);saved=np.load(folder/'transform.npz');assert all(np.array_equal(saved[k],v) for k,v in tr0.items());return old['validation_nll']
 if folder.exists():
  import shutil
  archive=HERE/'incomplete_selector_fits40'/folder.relative_to(HERE/'fits40');archive.parent.mkdir(parents=True,exist_ok=True);assert not archive.exists();shutil.move(str(folder),str(archive))
 folder.mkdir(parents=True,exist_ok=True);tr=fit_transform(X[trainmask]);np.savez_compressed(folder/'transform.npz',**tr);net=Critic();params=net.init(jax.random.PRNGKey(seed),jnp.zeros((1,33),jnp.float32));opt=optax.chain(optax.clip_by_global_norm(cfg['clip']),optax.adamw(cfg['lr'],weight_decay=cfg['wd']));state=opt.init(params);xx=jnp.asarray(apply_transform(X[trainmask],tr));yy=jnp.asarray(y[trainmask]);nnn=jnp.asarray(n[trainmask]);valid=np.flatnonzero(n[trainmask]>0);hx=jnp.asarray(apply_transform(X[heldmask],tr));hy=jnp.asarray(y[heldmask]);hn=jnp.asarray(n[heldmask]);rng=np.random.default_rng(seed);previous=0;history=[]
 @jax.jit
 def block(params,state,ids):
  def step(carry,idx):
   p,st=carry
   def loss(pp):
    logits=net.apply(pp,xx[idx]);return jnp.mean((yy[idx]*jax.nn.softplus(-logits)+(nnn[idx]-yy[idx])*jax.nn.softplus(logits))/4)
   grad=jax.grad(loss)(p);updates,st=opt.update(grad,st,p);return (optax.apply_updates(p,updates),st),None
  return jax.lax.scan(step,(params,state),ids)[0]
 @jax.jit
 def validation(pa):
  t=net.apply(pa,hx);return jnp.sum(hy*jax.nn.softplus(-t)+(hn-hy)*jax.nn.softplus(t))/jnp.sum(hn)
 for target in targets:
  ids=rng.choice(valid,size=(target-previous,cfg['batch']));params,state=block(params,state,jnp.asarray(ids));jax.block_until_ready(params);history.append(float(validation(params)) if heldmask.any() else None);(folder/f'step{target}.msgpack').write_bytes(serialization.to_bytes(params));previous=target
 write(folder/'fit.json',dict(seed=seed,train_rows=int(trainmask.sum()),held_rows=int(heldmask.sum()),targets=targets,validation_nll=history,jax_enable_x64=bool(jax.config.x64_enabled)));return history
