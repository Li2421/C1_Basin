import json,pickle,numpy as np,jax,jax.numpy as jnp,optax,flax.linen as nn
import critic as CR,gen as GN
from tab import KEYS
sp=json.load(open('SPLIT.json'));data=pickle.load(open('critic_data.pkl','rb'));TR=sp['train']
nm=CR.norm_of(data,TR)
best={n:int(data[n]['t'][:,:16].mean((0,1)).argmax()) for n in TR}
# 1) generator, top-1 target, track loss and in-sample fit
per={n:dict(h=((data[n]['h'][:16]-nm['hm'])/nm['hs']).astype(np.float32),c=((data[n]['c']-nm['cm'])/nm['cs']).astype(np.float32)) for n in TR}
net=GN.Generator();params=net.init(jax.random.PRNGKey(17),jnp.zeros((1,6)),jnp.zeros((1,12)))
opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(5e-4,weight_decay=1e-4));st=opt.init(params)
H=jnp.asarray(np.concatenate([per[n]['h'] for n in TR]));C=jnp.asarray(np.concatenate([np.repeat(per[n]['c'][None],16,0) for n in TR]));E=jnp.asarray(np.concatenate([np.repeat(CR.MENU[best[n]][None],16,0) for n in TR]),jnp.float32)
@jax.jit
def step(p,st):
 l,g=jax.value_and_grad(lambda q:-jnp.mean(GN.log_prob(net.apply(q,H,C),E)))(p);u,st=opt.update(g,st,p);return optax.apply_updates(p,u),st,l
for i in range(3000):
 params,st,l=step(params,st)
 if i%1000==0 or i==2999:
  m=GN.mean_eta(net.apply(params,H,C));print('step',i,'nll',round(float(l),2),'median dist mean->best',round(float(np.median(np.linalg.norm((np.asarray(m)-np.asarray(E))/CR.RADIUS,axis=1))),3))
raw=np.asarray(net.apply(params,H,C));mu,sig=GN.dist(raw);print('sigma median',np.round(np.median(np.asarray(sig),0),3))
# check normalization of c: duplicated/constant dims?
Cn=np.array([per[n]['c'] for n in TR]);print('c std per dim',np.round(Cn.std(0),2))
