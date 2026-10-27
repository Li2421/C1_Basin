"""Re-train the same generator (seed 17), check it reproduces frozen validation proposals, then make 16 proposals per TRAINING scene."""
import json,pickle,numpy as np,jax.numpy as jnp
import critic as CR,gen as GN
sp=json.load(open('SPLIT.json'));data=pickle.load(open('critic_data.pkl','rb'));TR=sp['train'];VA=sp['val'];G=json.load(open('GEN_FROZEN.json'))
nm=CR.norm_of(data,TR);gnet,gp=GN.train_generator(data,TR,nm)
def props(v):
 H=data[v]['h'];c=(data[v]['c']-nm['cm'])/nm['cs'];P=[]
 for s in range(8):
  raw=np.asarray(gnet.apply(gp,jnp.asarray(((H[s]-nm['hm'])/nm['hs'])[None],jnp.float32),jnp.asarray(c[None],jnp.float32)))[0]
  z=np.random.default_rng(GN.stable_int('hetset-proposals',17,v,s)).standard_normal(3);P.append(GN.mean_eta(raw));P.append(GN.from_noise(raw,z))
 return np.array(P)
err=max(np.abs(props(v)-np.array(G[v]['proposals'])).max() for v in VA);print('max diff vs frozen val proposals',err)
json.dump({n:props(n).tolist() for n in TR},open('GEN_TRAIN_PROPOSALS.json','w'))
