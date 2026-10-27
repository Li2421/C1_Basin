"""Generator q(eta|h,c) (structure + loss copied from cross_scene_pipeline_v1 networks/train_model; fixed 3000 steps)
-> 16 proposals per validation scene -> ranking critic (10 seeds) picks one. Everything frozen to GEN_TOP1_FROZEN.json before rollouts."""
import os,json,pickle,hashlib,math
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np,jax,jax.numpy as jnp,optax,flax.linen as nn
import critic as CR
LOW=np.array([0.,-1.5,0.],np.float32);HIGH=np.array([2.,1.5,1.5],np.float32);GC=(LOW+HIGH)/2;GR=(HIGH-LOW)/2
class Generator(nn.Module):
 def setup(self):
  self.state_adapter=nn.Dense(96);self.context=nn.Dense(32);self.shared1=nn.Dense(128);self.shared2=nn.Dense(64);self.out=nn.Dense(6)
 def __call__(self,h,c):
  x=jnp.concatenate((nn.silu(self.state_adapter(h)),nn.silu(self.context(c))),-1);x=nn.silu(self.shared1(x));x=nn.silu(self.shared2(x));return self.out(x)
def dist(raw):return raw[...,:3],.025+.275*jax.nn.sigmoid(raw[...,3:])
def mean_eta(raw):return GC+GR*np.tanh(np.asarray(raw)[...,:3])
def from_noise(raw,z):mu,s=dist(raw);return GC+GR*np.tanh(np.asarray(mu)+np.asarray(s)*z)
def log_prob(raw,eta):
 u=jnp.clip((eta-GC)/GR,-.999999,.999999);z=jnp.arctanh(u);mu,s=dist(raw)
 return jnp.sum(-.5*(((z-mu)/s)**2+2*jnp.log(s)+math.log(2*math.pi))-jnp.log(GR*(1-u*u)+1e-9),-1)
def stable_int(*p):return int(hashlib.sha256("\0".join(map(str,p)).encode()).hexdigest()[:16],16)
def train_generator(data,names,nm,seed=17,steps=3000,target='state'):
 per={}
 for n in names:
  t=data[n]['t'];S=[s for s in range(t.shape[1]) if np.isfinite(t[:,s]).all()];q=t[:,S].mean(0)
  pos=q==q.max(1,keepdims=True);neg=q==0;el=q.max(1)>0
  if target in('scene','top1'):
   qs=q.mean(0);top=np.zeros(49,bool);top[np.argsort(-qs,kind='stable')[:3 if target=='scene' else 1]]=True
   pos=np.repeat(top[None],len(q),0);neg=np.repeat((qs<.5*qs.max())[None],len(q),0);el=np.ones(len(q),bool)
  per[n]=dict(h=((data[n]['h'][S]-nm['hm'])/nm['hs'])[el].astype(np.float32),c=((data[n]['c']-nm['cm'])/nm['cs']).astype(np.float32),pos=pos[el],neg=neg[el])
 net=Generator();key=jax.random.PRNGKey(seed);params=net.init(key,jnp.zeros((1,6)),jnp.zeros((1,12)))
 opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(5e-4,weight_decay=1e-4));st=opt.init(params)
 E=CR.MENU.astype(np.float32)
 def obj(p,h,c,ep,en,has):
  raw=net.apply(p,h,c);lp=log_prob(raw,ep);ln=log_prob(raw,en)
  return jnp.mean(-lp.mean(1)+.25*jnp.sum(has*jax.nn.softplus(.5+ln-lp),1)/jnp.maximum(has.sum(1),1))
 @jax.jit
 def step(p,st,*a):
  l,g=jax.value_and_grad(obj)(p,*a);u,st=opt.update(g,st,p);return optax.apply_updates(p,u),st,l
 rngs={n:np.random.default_rng(stable_int('generator',seed,n)) for n in names}
 def draw(m,r):c=m.sum(-1);return np.argmax(np.cumsum(m,-1)>r.integers(c)[:,None],-1)
 for it in range(steps):
  B=[]
  for n in names:
   d=per[n];r=rngs[n];i=r.integers(len(d['h']),size=64);p=draw(d['pos'][i],r);has=d['neg'][i].any(1);ng=draw(np.where(has[:,None],d['neg'][i],d['pos'][i]),r)
   B.append((d['h'][i],np.repeat(d['c'][None],64,0),E[p],E[ng],has.astype(np.float32)))
  a=[jnp.asarray(np.stack([b[k] for b in B])) for k in range(5)];params,st,l=step(params,st,*a)
  if it%500==0:print('gen step',it,float(l),flush=True)
 return net,params
def Xe(c,h,k,nm,E):
 m=len(E);return np.column_stack([np.repeat(((c-nm['cm'])/nm['cs'])[None],m,0),np.repeat(((h-nm['hm'])/nm['hs'])[None],m,0),np.repeat(np.eye(3)[k][None],m,0),(E-CR.CENTER)/CR.RADIUS]).astype(np.float32)
if __name__=='__main__':
 sp=json.load(open('SPLIT.json'));data=pickle.load(open('critic_data.pkl','rb'));TR=sp['train'];VA=sp['val']
 nm=CR.norm_of(data,TR);gnet,gp=train_generator(data,TR,nm,target='top1')
 props={}
 for v in VA:
  H=data[v]['h'];c=(data[v]['c']-nm['cm'])/nm['cs'];P=[]
  for s in range(8):   # proposal states 0-7 (inputs only)
   raw=np.asarray(gnet.apply(gp,jnp.asarray(((H[s]-nm['hm'])/nm['hs'])[None],jnp.float32),jnp.asarray(c[None],jnp.float32)))[0]
   z=np.random.default_rng(stable_int('hetset-proposals-top1',17,v,s)).standard_normal(3)
   P.append(mean_eta(raw));P.append(from_noise(raw,z))
  props[v]=np.array(P)
 # ranking critic, 10 seeds, scores proposals over all 24 states x 3 controllers (inputs only)
 logits={v:[] for v in VA}
 for seed in range(10):
  net,params,nmc=CR.train(data,TR,'list',seed);ap=jax.jit(lambda p,x:net.apply(p,x))
  for v in VA:logits[v].append(np.array([[np.asarray(ap(params,Xe(data[v]['c'],data[v]['h'][s],k,nmc,props[v]))) for s in range(24)] for k in range(3)]))
 out={}
 for v in VA:
  f=np.mean(logits[v],0)
  def pick(K):
   g=f[...,:K];p=np.exp(g-g.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True);return int(p.mean((0,1)).argmax())
  out[v]=dict(proposals=props[v].tolist(),pick16=pick(16),pick8=pick(8))
  print(v,'pick16',np.round(props[v][out[v]['pick16']],2).tolist(),'pick8',np.round(props[v][out[v]['pick8']],2).tolist(),'means',np.round(props[v][0::2][:4],2).tolist())
 json.dump(out,open('GEN_TOP1_FROZEN.json','w'),indent=1)
