"""Critic fits on hetset data. data: {name: dict(t=(3,S,49) success/nan, c=(12,), h=(S,6))}.
kinds: bce (per-cell success, scene-equal) | list (within (scene,state,controller) softmax ranking)."""
import os,sys,json
os.environ['JAX_PLATFORMS']='cpu';os.environ['XLA_FLAGS']='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
import numpy as np,jax,jax.numpy as jnp,optax,flax.linen as nn,pickle
MENU=np.array(json.load(open('/home/zhihan/research/Basin_C1/diagnostics/critic_pool_supplement_v1/VALIDATION_INPUTS.json'))['etas'],float)
CENTER=np.array([1.,0.,.75]);RADIUS=np.array([1.,1.5,.75]);EN=(MENU-CENTER)/RADIUS
class MLP(nn.Module):
 @nn.compact
 def __call__(self,x,training=False):
  x=nn.silu(nn.Dense(128)(x));x=nn.Dropout(.1)(x,deterministic=not training)
  x=nn.silu(nn.Dense(128)(x));x=nn.Dropout(.1)(x,deterministic=not training)
  return nn.Dense(1)(x)[...,0]
def norm_of(data,names):
 C=np.array([data[n]['c'] for n in names]);H=np.concatenate([data[n]['h'] for n in names])
 cs=C.std(0);cs[cs<1e-6]=1;hs=H.std(0);hs[hs<1e-6]=1;return dict(cm=C.mean(0),cs=cs,hm=H.mean(0),hs=hs)
def X(c,h,k,nm):
 return np.column_stack([np.repeat(((c-nm['cm'])/nm['cs'])[None],49,0),np.repeat(((h-nm['hm'])/nm['hs'])[None],49,0),np.repeat(np.eye(3)[k][None],49,0),EN]).astype(np.float32)
def groups(data,names,nm):
 G=[];Y=[];M=[];W=[]
 for n in names:
  t=data[n]['t'];idx=[(k,s) for k in range(3) for s in range(t.shape[1]) if np.isfinite(t[k,s]).sum()>0]
  for k,s in idx:G.append(X(data[n]['c'],data[n]['h'][s],k,nm));Y.append(np.nan_to_num(t[k,s]));M.append(np.isfinite(t[k,s]));W.append(1/len(idx)/len(names))
 return np.array(G),np.array(Y,np.float32),np.array(M),np.array(W)
def train(data,names,kind,seed,steps=3000):
 nm=norm_of(data,names);G,Y,M,W=groups(data,names,nm)
 if kind=='list':
  s=(Y*M).sum(1);c=M.sum(1);mix=(s>0)&(s<c)&(c>=10);P=np.where(mix,W,0);P/=P.sum()
 else:P=W/W.sum()
 net=MLP();key=jax.random.PRNGKey(seed);params=net.init(key,jnp.zeros((1,24)));opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(optax.cosine_decay_schedule(1e-3,steps),weight_decay=1e-4));st=opt.init(params)
 def loss(p,x,y,m,dk):
  f=net.apply(p,x,training=True,rngs={'dropout':dk})
  if kind=='bce':return jnp.sum(optax.sigmoid_binary_cross_entropy(f,y)*m)/jnp.sum(m)
  lp=jax.nn.log_softmax(jnp.where(m,f,-1e9),-1);return -jnp.mean(jnp.sum(jnp.where(m,y*lp,0),-1)/jnp.maximum(jnp.sum(y*m,-1),1))
 @jax.jit
 def step(p,st,x,y,m,dk):
  l,g=jax.value_and_grad(loss)(p,x,y,m,dk);u,st=opt.update(g,st,p);return optax.apply_updates(p,u),st,l
 rng=np.random.default_rng(seed)
 for i in range(steps):
  ids=rng.choice(len(G),size=24,p=P);params,st,_=step(params,st,G[ids],Y[ids],M[ids],jax.random.fold_in(key,i+1))
 return net,params,nm
def predict(net,params,nm,c,H):
 ap=jax.jit(lambda p,x:net.apply(p,x));return np.array([[np.asarray(ap(params,X(c,H[s],k,nm))) for s in range(len(H))] for k in range(3)])
def score(f,kind):
 if kind=='bce':return (1/(1+np.exp(-f))).mean((0,1))
 p=np.exp(f-f.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True);return p.mean((0,1))
if __name__=='__main__':
 job=json.loads(sys.argv[1]);data=pickle.load(open(job['data'],'rb'))
 if os.path.exists(job['out']):sys.exit()
 net,params,nm=train(data,job['names'],job['kind'],job['seed'])
 np.savez_compressed(job['out'],**{n:predict(net,params,nm,data[n]['c'],data[n]['h_pred']) for n in job['targets']})
