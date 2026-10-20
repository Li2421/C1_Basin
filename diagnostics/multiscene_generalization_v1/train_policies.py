"""Fixed dev-loss checkpoint selection; no navigation or multi-agent labels read."""
from common import *
from policies import Policy
import jax,jax.numpy as jnp,optax,time
from flax import serialization

def train(family,kind,seed):
    cfg=frozen_config();assert all(d.platform=='gpu' for d in jax.devices());jax.config.update('jax_enable_x64',False)
    name=f'{family}_{kind}_{seed}';folder=HERE/'policies'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'COMPLETE.receipt.json').exists():return
    z=np.load(HERE/'datasets'/f'{family}.npz');train_x=z['train_X'];train_y=z['train_Y'];dev_x=z['dev_X'];dev_y=z['dev_Y']
    normal=dict(obs_mean=train_x.mean(0),obs_scale=np.maximum(train_x.std(0),1e-6),act_mean=train_y.mean(0),act_scale=np.maximum(train_y.std(0),1e-6))
    # Constant columns are imputed with unit scale, as in the source preprocessing.
    normal['obs_scale']=np.where(train_x.std(0)<1e-6,1.,normal['obs_scale'])
    config=dict(name=name,scene_family=family,family=kind,seed=seed,steps=cfg['training_steps'],batch=cfg['batch'],lr=cfg['lr'],hidden=[256,256,256],output_chart='world velocity per robot',observation_dimensions=35,selection='earliest minimum fixed dev loss',source_dataset_sha256=sha(HERE/'datasets'/f'{family}.npz'))
    write(folder/'CONFIG.json',config);write(folder/'NORMALIZATION.json',normal)
    rng=np.random.default_rng(202610173900+seed);vi=rng.integers(len(dev_x),size=2048);np.save(folder/'DEV_INDICES.npy',vi)
    seal(str(folder.relative_to(HERE)/'CONFIG_FROZEN.receipt.json'),[folder/'CONFIG.json',folder/'NORMALIZATION.json',folder/'DEV_INDICES.npy',HERE/'train_policies.py',HERE/'policies.py'],before_training=True)
    norm=lambda x,k:(x-np.asarray(normal[k+'_mean']))/np.asarray(normal[k+'_scale'])
    X=jnp.asarray(norm(train_x,'obs'));Y=jnp.asarray(norm(train_y,'act'));XV=jnp.asarray(norm(dev_x[vi],'obs'));YV=jnp.asarray(norm(dev_y[vi],'act'))
    net=Policy(kind);pa=net.init(jax.random.PRNGKey(seed),jnp.zeros((1,35),jnp.float32));opt=optax.adam(cfg['lr']);st=opt.init(pa)
    def loss(p,x,y):
        if kind=='deterministic':return jnp.mean(jnp.square(net.apply(p,x)-y))
        logits,means,logs=net.apply(p,x);component=-jnp.sum(.5*jnp.square((y[:,None]-means)*jnp.exp(-logs))+logs+.5*jnp.log(2*jnp.pi),-1)
        return -jnp.mean(jax.scipy.special.logsumexp(jax.nn.log_softmax(logits,-1)+component,-1))
    @jax.jit
    def block(pa,st,ids):
        def update(carry,ids):
            p,s=carry;g=jax.grad(loss)(p,X[ids],Y[ids]);delta,s=opt.update(g,s,p);return (optax.apply_updates(p,delta),s),None
        return jax.lax.scan(update,(pa,st),ids)[0]
    validate=jax.jit(lambda p:loss(p,XV,YV));best=float(validate(pa));beststep=0;history=[dict(step=0,dev_loss=best)]
    (folder/'best.msgpack').write_bytes(serialization.to_bytes(pa));rng=np.random.default_rng(seed);start=time.monotonic()
    for target in range(250,cfg['training_steps']+1,250):
        ids=rng.integers(len(train_x),size=(250,cfg['batch']));pa,st=block(pa,st,jnp.asarray(ids));value=float(validate(pa));assert np.isfinite(value)
        history.append(dict(step=target,dev_loss=value))
        if value<best:best=value;beststep=target;(folder/'best.msgpack').write_bytes(serialization.to_bytes(pa))
        if target%5000==0:print(name,target,value,flush=True)
    write(folder/'HISTORY.json',history);write(folder/'SUMMARY.json',dict(best_step=beststep,best_loss=best,train_transitions=len(train_x),dev_transitions=len(dev_x),elapsed_seconds=time.monotonic()-start))
    seal(str(folder.relative_to(HERE)/'COMPLETE.receipt.json'),[folder/'best.msgpack',folder/'CONFIG.json',folder/'NORMALIZATION.json',folder/'HISTORY.json'],no_navigation_labels_read=True,no_evaluation_labels_read=True)

def worker(index):
    eligible=read(HERE/'DATASET_FROZEN.json')['families'];tasks=[(f,k,s) for f in eligible for k in ['deterministic','mixture'] for s in [17,23]]
    for task in tasks[index::2]:train(*task)

if __name__=='__main__':worker(int(sys.argv[1]))
