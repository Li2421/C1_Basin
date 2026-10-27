"""Leave-one-training-scene-out check of generator target rules (training data only).
proxy value = held-out scene's menu rate (states 0-15) at the menu point nearest to each proposal mean; report mean over proposals and best."""
import json,pickle,sys,numpy as np,jax.numpy as jnp
import critic as CR,gen as GN
sp=json.load(open('SPLIT.json'));data=pickle.load(open('critic_data.pkl','rb'));TR=sp['train'];target=sys.argv[1];held=sys.argv[2]
src=[n for n in TR if n!=held];nm=CR.norm_of(data,src);gnet,gp=GN.train_generator(data,src,nm,target=target)
H=data[held]['h'];c=(data[held]['c']-nm['cm'])/nm['cs'];M=[]
for s in range(8):M.append(GN.mean_eta(np.asarray(gnet.apply(gp,jnp.asarray(((H[s]-nm['hm'])/nm['hs'])[None],jnp.float32),jnp.asarray(c[None],jnp.float32)))[0]))
M=np.array(M);q=data[held]['t'][:,:16].mean((0,1));nearest=[int(np.linalg.norm((CR.MENU-m)/CR.RADIUS,axis=1).argmin()) for m in M]
b=int(q.argmax());print(json.dumps(dict(target=target,held=held,dist_best=float(np.linalg.norm((M-CR.MENU[b])/CR.RADIUS,axis=1).min()),proxy_mean=float(np.mean(q[nearest])),proxy_first=float(q[nearest[0]]),scene_best=float(q.max()),global22=float(q[39]))))
