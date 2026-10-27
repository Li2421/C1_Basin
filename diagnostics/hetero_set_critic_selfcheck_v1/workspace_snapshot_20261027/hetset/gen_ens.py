"""3-seed top-1 generator ensemble; 16 proposals = 8 states x (mean, sample), slot (s,j) uses seed SEEDS[(2s+j)%3]."""
import json,pickle,sys,numpy as np,jax.numpy as jnp
import critic as CR,gen as GN
SEEDS=(17,23,41)
def train_ens(data,names):
 nm=CR.norm_of(data,names);return nm,[GN.train_generator(data,names,nm,seed=s,target='top1') for s in SEEDS]
def props(data,nm,models,v,tag):
 H=data[v]['h'];c=(data[v]['c']-nm['cm'])/nm['cs'];P=[]
 for s in range(8):
  for j in range(2):
   m=(2*s+j)%3;gnet,gp=models[m]
   raw=np.asarray(gnet.apply(gp,jnp.asarray(((H[s]-nm['hm'])/nm['hs'])[None],jnp.float32),jnp.asarray(c[None],jnp.float32)))[0]
   if j==0:P.append(GN.mean_eta(raw))
   else:P.append(GN.from_noise(raw,np.random.default_rng(GN.stable_int(tag,SEEDS[m],v,s)).standard_normal(3)))
 return np.array(P)
if __name__=='__main__':
 sp=json.load(open('SPLIT.json'));data=pickle.load(open(sys.argv[2] if len(sys.argv)>2 else 'critic_data.pkl','rb'))
 if sys.argv[1]=='all':   # trained on all training scenes -> proposals for given targets file
  nm,models=train_ens(data,sp['train']);tg=json.load(open('ENS_TARGETS.json'))
  print('PROPS',json.dumps({v:props(data,nm,models,v,'hetset-ens').tolist() for v in tg}))
 else:                    # leave-one-out for one training scene
  held=sys.argv[1];nm,models=train_ens(data,[n for n in sp['train'] if n!=held])
  print('PROPS',json.dumps({held:props(data,nm,models,held,'hetset-ens-loso').tolist()}))
