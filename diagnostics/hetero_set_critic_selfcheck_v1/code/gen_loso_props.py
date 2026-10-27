"""For one training scene: train top-1 generator on the other 27, make 16 proposals (same scheme as validation)."""
import json,pickle,sys,numpy as np,jax.numpy as jnp
import critic as CR,gen as GN
sp=json.load(open('SPLIT.json'));data=pickle.load(open('critic_data.pkl','rb'));held=sys.argv[1]
src=[n for n in sp['train'] if n!=held];nm=CR.norm_of(data,src);gnet,gp=GN.train_generator(data,src,nm,target='top1')
H=data[held]['h'];c=(data[held]['c']-nm['cm'])/nm['cs'];P=[]
for s in range(8):
 raw=np.asarray(gnet.apply(gp,jnp.asarray(((H[s]-nm['hm'])/nm['hs'])[None],jnp.float32),jnp.asarray(c[None],jnp.float32)))[0]
 z=np.random.default_rng(GN.stable_int('hetset-proposals-top1-loso',17,held,s)).standard_normal(3);P.append(GN.mean_eta(raw));P.append(GN.from_noise(raw,z))
print('PROPS',json.dumps({held:np.array(P).tolist()}))
