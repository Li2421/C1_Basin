"""Frozen 39-dimensional round0--3 training, fixed development-loss selection."""
from common import *
from policies39 import Policy,loss
import jax,jax.numpy as jnp,optax,time
from flax import serialization

def normalization(x,y):
 assert x.ndim==2 and x.shape[1]==39 and y.shape==(len(x),2)
 assert len(x) and np.isfinite(x).all() and np.isfinite(y).all()
 scale=x.std(0)
 return dict(obs_mean=x.mean(0),obs_scale=np.where(scale<1e-6,1.,scale),act_mean=y.mean(0),act_scale=np.maximum(y.std(0),1e-6))

def earliest_best(history):return min(history,key=lambda row:(row['dev_loss'],row['step']))['step']

def train(family,kind,seed,round_index):
 from freeze_guard39 import verify
 verify()
 assert round_index in range(4)
 assert all(d.platform=='gpu' for d in jax.devices());jax.config.update('jax_enable_x64',False)
 folder=HERE/'policies39'/f'{family}_{kind}_{seed}'/f'round_{round_index}';folder.mkdir(parents=True,exist_ok=True)
 if (folder/'COMPLETE.receipt.json').exists():return
 dataset=HERE/'datasets39'/f'{family}_{kind}_{seed}_round_{round_index}.npz';manifest=read(dataset.with_suffix('.json'))
 assert manifest['round']==round_index and manifest['heldout_geometry_used'] is False
 assert manifest['dataset_sha256']==sha(dataset)
 z=np.load(dataset);x=z['train_X'];y=z['train_Y'];dev_x=z['dev_X'];dev_y=z['dev_Y'];normal=normalization(x,y)
 assert dev_x.shape[1]==39 and dev_y.shape==(len(dev_x),2)
 config=dict(scene_family=family,family=kind,seed=seed,round=round_index,steps=60000,batch=256,lr=.0003,hidden=[256,256,256],observation_dimensions=39,selection='earliest minimum fixed dev loss',initialization='fresh same policy seed each round',dataset=str(dataset),source_dataset_sha256=sha(dataset),final_round=3)
 write(folder/'CONFIG.json',config);write(folder/'NORMALIZATION.json',normal)
 vi=np.random.default_rng(202610173900+seed).integers(len(dev_x),size=2048);np.save(folder/'DEV_INDICES.npy',vi)
 seal(str(folder.relative_to(HERE)/'CONFIG_FROZEN.receipt.json'),[folder/'CONFIG.json',folder/'NORMALIZATION.json',folder/'DEV_INDICES.npy',HERE/'training39.py',HERE/'policies39.py',dataset.with_suffix('.json')],before_training=True)
 norm=lambda a,k:(a-np.asarray(normal[k+'_mean']))/np.asarray(normal[k+'_scale'])
 X=jnp.asarray(norm(x,'obs'));Y=jnp.asarray(norm(y,'act'));XV=jnp.asarray(norm(dev_x[vi],'obs'));YV=jnp.asarray(norm(dev_y[vi],'act'))
 net=Policy(kind);pa=net.init(jax.random.PRNGKey(seed),jnp.zeros((1,39),jnp.float32));optimizer=optax.adam(.0003);state=optimizer.init(pa)
 @jax.jit
 def block(pa,state,ids):
  def update(carry,idx):
   p,s=carry;gradient=jax.grad(lambda p:loss(net,kind,p,X[idx],Y[idx]))(p);delta,s=optimizer.update(gradient,s,p);return (optax.apply_updates(p,delta),s),None
  return jax.lax.scan(update,(pa,state),ids)[0]
 validate=jax.jit(lambda p:loss(net,kind,p,XV,YV));best=float(validate(pa));history=[dict(step=0,dev_loss=best)];(folder/'best.msgpack').write_bytes(serialization.to_bytes(pa));rng=np.random.default_rng(seed);begin=time.monotonic()
 for step in range(250,60001,250):
  ids=rng.integers(len(x),size=(250,256));pa,state=block(pa,state,jnp.asarray(ids));value=float(validate(pa));assert np.isfinite(value)
  history.append(dict(step=step,dev_loss=value))
  if value<best:best=value;(folder/'best.msgpack').write_bytes(serialization.to_bytes(pa))
  if step%5000==0:print(family,kind,seed,round_index,step,value,flush=True)
 write(folder/'HISTORY.json',history);write(folder/'SUMMARY.json',dict(best_step=earliest_best(history),best_loss=best,train_transitions=len(x),dev_transitions=len(dev_x),elapsed_seconds=time.monotonic()-begin))
 seal(str(folder.relative_to(HERE)/'COMPLETE.receipt.json'),[folder/n for n in ['CONFIG.json','NORMALIZATION.json','DEV_INDICES.npy','best.msgpack','HISTORY.json','SUMMARY.json']],no_navigation_holdout_read=True,final_policy=round_index==3)

if __name__=='__main__':
 tasks=read(HERE/'TRAINING39_TASKS.json');index=int(sys.argv[1])
 for task in tasks[index::2]:train(**task)
