"""Diagnostic critic fits (job tmp only). kinds: bce (per-cell success), list (within (scene,state,controller) ranking), hybrid."""
import os,sys,json
os.environ.setdefault('XLA_FLAGS','--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
from load import *
import jax,jax.numpy as jnp,optax,flax.linen as nn
E_N=(MENU-CENTER)/RADIUS
def mixed_frac(n):
 t=cell_table(n);s=np.nansum(t,-1);c=np.isfinite(t).sum(-1);ok=c>=10
 return float(((s>0)&(s<c)&ok).sum()/max(ok.sum(),1)) if ok.any() else 0.
def build(names):
 C=np.array([cprime(n) for n in names]);cm=C.mean(0);cs=C.std(0);cs[cs<1e-6]=1
 H=np.concatenate([rows(n)['h'] for n in names]);hm=H.mean(0);hs=H.std(0);hs[hs<1e-6]=1
 norm=dict(cm=cm,cs=cs,hm=hm,hs=hs)
 G=[];Y=[];M=[];W=[]
 for n in names:
  t=cell_table(n);h=rows(n)['h'];c=(cprime(n)-cm)/cs;grp=[]
  for k in range(3):
   for s in range(t.shape[1]):
    m=np.isfinite(t[k,s]);
    if m.sum()==0:continue
    x=np.column_stack([np.repeat(c[None],49,0),np.repeat(((h[s]-hm)/hs)[None],49,0),np.repeat(np.eye(3)[k][None],49,0),E_N])
    grp.append((x,np.nan_to_num(t[k,s]),m))
  for x,y,m in grp:G.append(x);Y.append(y);M.append(m);W.append(1/len(grp)/len(names))
 G=np.array(G,np.float32);Y=np.array(Y,np.float32);M=np.array(M,bool);W=np.array(W);W/=W.sum()
 return G,Y,M,W,norm
def target_x(n,norm):
 h=vh(n) if n in VIN['scene_inputs'] else rows(n)['h'];c=(cprime(n)-norm['cm'])/norm['cs'];S=len(h)
 X=np.zeros((3,S,49,24),np.float32)
 for k in range(3):
  for s in range(S):X[k,s]=np.column_stack([np.repeat(c[None],49,0),np.repeat(((h[s]-norm['hm'])/norm['hs'])[None],49,0),np.repeat(np.eye(3)[k][None],49,0),E_N])
 return X
class MLP(nn.Module):
 @nn.compact
 def __call__(self,x,training=False):
  x=nn.silu(nn.Dense(128)(x));x=nn.Dropout(.1)(x,deterministic=not training)
  x=nn.silu(nn.Dense(128)(x));x=nn.Dropout(.1)(x,deterministic=not training)
  return nn.Dense(1)(x)[...,0]
def train(names,kind,seed,targets,steps=3000):
 G,Y,M,W,norm=build(names)
 if kind in('list','hybrid'):
  s=(Y*M).sum(1);c=M.sum(1);mixed=(s>0)&(s<c)&(c>=10)
  # re-weight scene-equal among mixed groups only
  Wl=np.where(mixed,W,0.);
  # renormalise per scene: groups of a scene share weight equally
  Wl=Wl/Wl.sum()
 net=MLP();key=jax.random.PRNGKey(seed);params=net.init(key,jnp.zeros((1,24)));opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(optax.cosine_decay_schedule(1e-3,steps),weight_decay=1e-4));st=opt.init(params)
 def loss_fn(p,x,y,m,dk):
  f=net.apply(p,x,training=True,rngs={'dropout':dk})  # (B,49)
  bce=jnp.sum(optax.sigmoid_binary_cross_entropy(f,y)*m)/jnp.sum(m)
  lf=jnp.where(m,f,-1e9);lp=jax.nn.log_softmax(lf,-1);lst=-jnp.mean(jnp.sum(jnp.where(m,y*lp,0),-1)/jnp.maximum(jnp.sum(y*m,-1),1))
  return {'bce':bce,'list':lst,'hybrid':bce+lst}[kind]
 @jax.jit
 def step(p,st,x,y,m,dk):
  l,g=jax.value_and_grad(loss_fn)(p,x,y,m,dk);u,st=opt.update(g,st,p);return optax.apply_updates(p,u),st,l
 rng=np.random.default_rng(seed);B=24 if kind!='bce' else 24
 P=Wl if kind in('list','hybrid') else W
 for i in range(steps):
  ids=rng.choice(len(G),size=B,p=P)
  if kind=='hybrid':
   ids2=rng.choice(len(G),size=B,p=W);ids=np.concatenate([ids,ids2])
  params,st,l=step(params,st,G[ids],Y[ids],M[ids],jax.random.fold_in(key,i+1))
 ap=jax.jit(lambda p,x:net.apply(p,x))
 return {n:np.asarray(ap(params,jnp.asarray(target_x(n,norm)))) for n in targets}
if __name__=='__main__':
 job=json.loads(sys.argv[1]);out=job['out']
 if os.path.exists(out):sys.exit()
 r=train(job['names'],job['kind'],job['seed'],job['targets'],job.get('steps',3000))
 np.savez_compressed(out,**r)
