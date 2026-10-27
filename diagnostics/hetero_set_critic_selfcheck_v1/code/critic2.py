"""Ranking critic trained on menu + off-menu (generator proposal) labels. Candidate sets differ per scene."""
import os,json,pickle,sys
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np,jax,jax.numpy as jnp,optax
import critic as CR
def Xe(c,h,k,nm,E):
 m=len(E);return np.column_stack([np.repeat(((c-nm['cm'])/nm['cs'])[None],m,0),np.repeat(((h-nm['hm'])/nm['hs'])[None],m,0),np.repeat(np.eye(3)[k][None],m,0),(E-CR.CENTER)/CR.RADIUS]).astype(np.float32)
def build(scenes,nm):
 G=[];Y=[];M=[];W=[];K=max(len(s['E']) for s in scenes.values())
 for n,s in scenes.items():
  t=s['t'];idx=[(k,st) for k in range(3) for st in range(t.shape[1]) if np.isfinite(t[k,st]).sum()>=10]
  for k,st in idx:
   x=Xe(s['c'],s['h'][st],k,nm,s['E']);y=np.nan_to_num(t[k,st]);m=np.isfinite(t[k,st]);pad=K-len(x)
   G.append(np.pad(x,((0,pad),(0,0))));Y.append(np.pad(y,(0,pad)));M.append(np.pad(m,(0,pad)));W.append(1/len(idx)/len(scenes))
 return np.array(G),np.array(Y,np.float32),np.array(M),np.array(W)
def train(scenes,nm,seed,steps=3000):
 G,Y,M,W=build(scenes,nm);s=(Y*M).sum(1);c=M.sum(1);mix=(s>0)&(s<c);P=np.where(mix,W,0);P/=P.sum()
 net=CR.MLP();key=jax.random.PRNGKey(seed);params=net.init(key,jnp.zeros((1,24)));opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(optax.cosine_decay_schedule(1e-3,steps),weight_decay=1e-4));st=opt.init(params)
 def loss(p,x,y,m,dk):
  f=net.apply(p,x,training=True,rngs={'dropout':dk});lp=jax.nn.log_softmax(jnp.where(m,f,-1e9),-1);return -jnp.mean(jnp.sum(jnp.where(m,y*lp,0),-1)/jnp.maximum(jnp.sum(y*m,-1),1))
 @jax.jit
 def step(p,st,x,y,m,dk):
  l,g=jax.value_and_grad(loss)(p,x,y,m,dk);u,st=opt.update(g,st,p);return optax.apply_updates(p,u),st,l
 rng=np.random.default_rng(seed)
 for i in range(steps):
  ids=rng.choice(len(G),size=24,p=P);params,st,_=step(params,st,G[ids],Y[ids],M[ids],jax.random.fold_in(key,i+1))
 return net,params
