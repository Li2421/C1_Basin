from core import *
from guard import guard
from student import network,inputs,encoding,target,public_key
from data import teacher,dataset
from audit_utils import source_inventory
import jax,jax.numpy as jnp,optax,pickle,time
G=guard()
def train(index,variant,rd):
 assert jax.default_backend()=='gpu','Training must use GPU'
 assert read(HERE/'UNIT_TESTS.json')['status']=='PASS'
 for p,h in read(HERE/'IMPLEMENTATION_FROZEN.receipt.json')['files'].items():assert sha(p)==h,p
 out=HERE/'students'/variant/str(index);out.mkdir(parents=True,exist_ok=True);stem='offline' if rd==0 else f'round{rd}';dest=out/f'{stem}.pkl';complete=out/f'{stem}.json'
 if complete.exists():assert sha(dest)==read(complete)['checkpoint_sha256'];return
 assert not dest.exists(),'Preserve partial checkpoint; do not retrain or overwrite'
 rt=teacher(index);x,o,norm=dataset(index,'train');dx,do,_=dataset(index,'dev');mean=jnp.asarray(norm['mean']);scale=jnp.asarray(norm['scale']);model=network();seed=read(HERE/'TEACHERS.json')[index]['student_seed']+100000*int(variant[1:])+rd
 if rd==0:
  assert variant=='V3';params=model.init(jax.random.PRNGKey(seed),jnp.zeros((4,27),jnp.float32),jnp.zeros((4,8),jnp.float32))['params'];initial=None
 else:
  from collect import checkpoint
  initial=checkpoint(index,variant,rd)
  with initial.open('rb') as f:ck=pickle.load(f)
  params=ck['params'];np.testing.assert_array_equal(ck['mean'],np.asarray(mean));np.testing.assert_array_equal(ck['scale'],np.asarray(scale))
 tx=optax.adam(3e-4);opt=tx.init(params)
 native=jax.jit(jax.vmap(rt.native_fn));noise=jax.jit(jax.vmap(lambda o,k:rt.trace_fn(o,k)[0].reshape(4,2)))
 # Closed-loop labels preserve the exact sampled teacher action and noise.
 extras=[];provenance=[]
 for r in range(1,rd+1):
  folder=HERE/'dagger'/variant/str(index)/f'round{r}';info=read(folder/'COMPLETE.json');xx=[];oo=[];kk=[]
  for s in info['sources']:
   assert sha(s['trace_path'])==s['trace_sha256'];z=np.load(s['trace_path']);xx.extend(z['local']);oo.extend(z['teacher']);kk.extend(z['keys'])
  xx=np.asarray(xx,np.float32);oo=np.asarray(oo,np.float32);kk=jnp.asarray(kk,np.uint32)
  if variant=='V3':kk=jax.vmap(public_key)(kk)
  label=folder/'labels.npz'
  if label.exists():
   old=np.load(label);uu=old['target'];zz=old['noise'];np.testing.assert_array_equal(old['keys'],np.asarray(kk))
  else:
   us=[];zs=[]
   for start in range(0,len(xx),256):
    ob=jnp.asarray(oo[start:start+256]);ks=kk[start:start+256];us.append(np.asarray(native(ob,ks)));zs.append(np.asarray(noise(ob,ks)))
   uu=np.concatenate(us);zz=np.concatenate(zs);np.savez_compressed(label,target=uu,noise=zz,keys=np.asarray(kk))
  extras.append((xx,oo,uu,zz));provenance.append(dict(round=r,samples=len(xx),collection_sha256=sha(folder/'COMPLETE.json'),labels_sha256=sha(label)))
 base_count=len(x)
 if extras:
  ax=np.concatenate([e[0] for e in extras]);ao=np.concatenate([e[1] for e in extras]);au=np.concatenate([e[2] for e in extras]);az=np.concatenate([e[3] for e in extras]);extra_count=len(ax)
 else:
  ax=x[:1];ao=o[:1];au=np.zeros((1,4,2),np.float32);az=au.copy();extra_count=0
 x=jnp.asarray(x);o=jnp.asarray(o);ax=jnp.asarray(ax);ao=jnp.asarray(ao);au=jnp.asarray(au);az=jnp.asarray(az)
 def loss(p,xx,ob,z,u):return jnp.mean((model.apply({'params':p},inputs(xx,mean,scale),encoding(index,variant,xx,ob,z))-target(index,u,ob))**2)
 @jax.jit
 def step(carry,_):
  p,opt,key=carry;key,ik,nk=jax.random.split(key,3);ix=jax.random.randint(ik,(256,),0,base_count+extra_count);mask=ix>=base_count;bi=jnp.minimum(ix,base_count-1);ai=jnp.clip(ix-base_count,0,len(ax)-1);keys=jax.random.split(nk,256);ob=o[bi];xx=x[bi];u=native(ob,keys);z=noise(ob,keys);ob=jnp.where(mask[:,None,None],ao[ai],ob);xx=jnp.where(mask[:,None,None],ax[ai],xx);u=jnp.where(mask[:,None,None],au[ai],u);z=jnp.where(mask[:,None,None],az[ai],z)
  val,grad=jax.value_and_grad(loss)(p,xx,ob,z,u);updates,opt=tx.update(grad,opt,p);p=optax.apply_updates(p,updates);return (p,opt,key),val
 @jax.jit
 def chunk(carry):return jax.lax.scan(step,carry,None,length=500)
 dk=jax.random.PRNGKey(2026103031);dk,ik,nk=jax.random.split(dk,3);di=jax.random.randint(ik,(2048,),0,len(dx));dkeys=jax.random.split(nk,2048);devx=jnp.asarray(dx)[di];devo=jnp.asarray(do)[di];devu=native(devo,dkeys);devz=noise(devo,dkeys);evaluate=jax.jit(lambda p:loss(p,devx,devo,devz,devu))
 history=[];best=float('inf');beststep=None;bestparams=None;carry=(params,opt,jax.random.PRNGKey(seed));t0=time.time();nsteps=80000 if rd==0 else 20000
 for t in range(0,nsteps+1,500):
  if t:carry,values=chunk(carry);trainloss=float(values.mean())
  else:trainloss=None
  value=float(evaluate(carry[0]));history.append(dict(step=t,dev_mse=value,train_mse=trainloss,elapsed=time.time()-t0))
  if value<best:best=value;beststep=t;bestparams=jax.device_get(carry[0])
  write(out/f'{stem}_history.json',history)
  if t%5000==0:print(variant,index,rd,t,value,time.time()-t0,flush=True)
 selected=bestparams if rd==0 else jax.device_get(carry[0]);selectedstep=beststep if rd==0 else nsteps
 with dest.open('xb') as f:pickle.dump(dict(params=selected,mean=np.asarray(mean),scale=np.asarray(scale),step=selectedstep,round=rd,seed=seed,teacher_sha256=read(HERE/'TEACHERS.json')[index]['checkpoint_sha256'],variant=variant),f)
 write(complete,dict(utc=utc(),checkpoint_sha256=sha(dest),checkpoint=str(dest),index=index,variant=variant,round=rd,steps=nsteps,selected_step=selectedstep,best_dev_step=beststep,initial_checkpoint=str(initial) if initial else None,initial_sha256=sha(initial) if initial else None,base_samples=base_count,dagger_samples=extra_count,provenance=provenance,guard=guard(),backend=jax.default_backend(),evaluation_labels_read=False,elapsed=time.time()-t0));source_inventory(f'imports_train_{variant}_{index}_{rd}.json')
if __name__=='__main__':train(int(sys.argv[1]),sys.argv[2],int(sys.argv[3]))
