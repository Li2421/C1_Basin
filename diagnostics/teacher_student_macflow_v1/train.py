from core import *
from audit_utils import source_inventory
from data import teacher
from student import network,inputs
import jax
import jax.numpy as jnp
import optax,pickle,time
jax.config.update('jax_enable_x64',False)

def train(index):
    assert read(HERE/'NOISE_GATE.json')['status']=='PASS'
    r=read(HERE/'TEACHERS.json')[index];cfg=read(HERE/'TRAINING_CONFIG.json');seed=r['student_seed'];out=HERE/'students'/r['name'];out.mkdir(parents=True,exist_ok=True)
    assert not (out/'COMPLETE.json').exists(),'Do not retrain frozen student'
    data=HERE/'data'/r['name'];assert (data/'READY.json').exists();rt=teacher(index)
    train_o=jnp.asarray(np.load(data/'train_teacher.npy'));train_x=jnp.asarray(np.load(data/'train_local.npy'))
    dev_o=jnp.asarray(np.load(data/'dev_teacher.npy'));dev_x=jnp.asarray(np.load(data/'dev_local.npy'));norm=np.load(data/'normalization.npz');mean=jnp.asarray(norm['mean']);scale=jnp.asarray(norm['scale'])
    model=network();params=model.init(jax.random.PRNGKey(seed),jnp.zeros((4,27),jnp.float32),jnp.zeros((4,2),jnp.float32))['params'];tx=optax.adam(3e-4);opt=tx.init(params)
    target=jax.vmap(rt.native_fn)
    noise=jax.vmap(lambda o,k:rt.trace_fn(o,k)[0].reshape(4,2))
    def loss(p,x,z,u):return jnp.mean((model.apply({'params':p},inputs(x,mean,scale),z)-u)**2)
    @jax.jit
    def step(carry,_):
        p,opt,key=carry;key,ixk,nk=jax.random.split(key,3);ix=jax.random.randint(ixk,(256,),0,len(train_x));keys=jax.random.split(nk,256);o=train_o[ix];x=train_x[ix];u=target(o,keys);z=noise(o,keys);value,grad=jax.value_and_grad(loss)(p,x,z,u);updates,opt=tx.update(grad,opt,p);p=optax.apply_updates(p,updates);return (p,opt,key),value
    @jax.jit
    def chunk(carry):return jax.lax.scan(step,carry,None,length=500)
    # Fixed dev draws are identical at every validation checkpoint.
    dk=jax.random.PRNGKey(cfg['dev_batch_seed']);dk,ik,nk=jax.random.split(dk,3);ix=jax.random.randint(ik,(8,256),0,len(dev_x));keys=jax.random.split(nk,8*256).reshape(8,256,2)
    dx=dev_x[ix];du=jax.vmap(target)(dev_o[ix],keys);dz=jax.vmap(noise)(dev_o[ix],keys)
    evaluate=jax.jit(lambda p:jnp.mean(jax.vmap(lambda x,z,u:loss(p,x,z,u))(dx,dz,du)))
    best=float('inf');beststep=None;history=[];carry=(params,opt,jax.random.PRNGKey(seed));t0=time.time()
    for t in range(0,80001,500):
        if t:
            carry,values=chunk(carry);trainloss=float(values.mean())
        else:trainloss=None
        params=carry[0];dev=float(evaluate(params));row=dict(step=t,dev_mse=dev,train_mse=trainloss,elapsed=time.time()-t0);history.append(row)
        if dev<best:
            best=dev;beststep=t
            with (out/'best.pkl').open('wb') as f:pickle.dump(dict(params=jax.device_get(params),mean=np.asarray(mean),scale=np.asarray(scale),step=t,dev_mse=dev,seed=seed,teacher_sha256=r['checkpoint_sha256']),f)
        if t%5000==0:print(json.dumps(dict(teacher=r['name'],**row)),flush=True)
        write(out/'history.json',history)
    write(out/'COMPLETE.json',dict(utc=utc(),steps=80000,best_step=beststep,dev_mse=best,checkpoint_sha256=sha(out/'best.pkl'),teacher=r,config_sha256=sha(HERE/'TRAINING_CONFIG.json'),jax_enable_x64=bool(jax.config.x64_enabled),backend=jax.default_backend(),elapsed=time.time()-t0,closed_loop_results_read=False));source_inventory(f'imports_train_{index}.json')
if __name__=='__main__':
    start=int(sys.argv[1]) if len(sys.argv)>1 else 0
    for i in range(start,14):train(i)
