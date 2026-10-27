"""Scene-level-target generator: in-sample fit check on training scenes + proposals for validation (no labels)."""
import json,pickle,numpy as np,jax.numpy as jnp
import critic as CR,gen as GN
from tab import KEYS
sp=json.load(open('SPLIT.json'));data=pickle.load(open('critic_data.pkl','rb'));TR=sp['train'];VA=sp['val']
nm=CR.norm_of(data,TR);gnet,gp=GN.train_generator(data,TR,nm,target='scene')
def props(v):
 H=data[v]['h'];c=(data[v]['c']-nm['cm'])/nm['cs'];P=[]
 for s in range(8):
  raw=np.asarray(gnet.apply(gp,jnp.asarray(((H[s]-nm['hm'])/nm['hs'])[None],jnp.float32),jnp.asarray(c[None],jnp.float32)))[0]
  z=np.random.default_rng(GN.stable_int('hetset-proposals-scene',17,v,s)).standard_normal(3);P.append(GN.mean_eta(raw));P.append(GN.from_noise(raw,z))
 return np.array(P)
dist=[]
for n in TR:
 q=data[n]['t'][:,:16].mean((0,1));b=int(q.argmax());P=props(n);d=np.linalg.norm((P-CR.MENU[b])/CR.RADIUS,axis=1).min();dist.append(d)
 if n.startswith('hs_obst'):print(f'{n:24s} best {KEYS[b]} nearest proposal {d:.2f} mean side {P[:,1].mean():+.2f}')
print('training scenes: median nearest-proposal distance to scene best',round(float(np.median(dist)),3))
out={v:dict(proposals=props(v).tolist()) for v in VA}
for v in VA:print(v,'proposal means',np.round(np.array(out[v]['proposals'])[0::2][:3],2).tolist())
json.dump(out,open('GEN_SCENE_PROPOSALS.json','w'))
