"""Fixed-length scene-equal soft-label training; no validation-label access or tuning."""
from common import *
from critic_models import model
from flax import serialization
import jax,jax.numpy as jnp,optax,numpy as np

def fit(task):
 definition=read(HERE/'datasets'/f"{task['dataset']}.json");assert sha(definition['path'])==definition['sha256'];folder=HERE/'fits'/task['id']/f"seed{task['seed']}";receipt=folder/'FIT.receipt.json'
 if receipt.exists():
  r=read(receipt);assert sha(r['checkpoint'])==r['sha256'];return r
 folder.mkdir(parents=True,exist_ok=True)
 with np.load(definition['path'],allow_pickle=False)as z:X=z['x'];Y=z['y'];prob=z['prob']
 seed=int(task['seed']);net=model(task['kind']);key=jax.random.PRNGKey(seed);params=net.init(key,jnp.zeros((1,27),jnp.float32),training=False);steps=4000 if task['kind']=='original'else 3000
 opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(optax.cosine_decay_schedule(1e-3,steps),weight_decay=1e-4));state=opt.init(params)
 @jax.jit
 def step(p,st,x,y,dropkey):
  def objective(q):return jnp.mean(optax.sigmoid_binary_cross_entropy(net.apply(q,x,training=True,rngs={'dropout':dropkey}),y))
  loss,grads=jax.value_and_grad(objective)(p);updates,st=opt.update(grads,st,p);return optax.apply_updates(p,updates),st,loss
 rng=np.random.default_rng(seed);history=[];begin=time.monotonic();start=utc()
 for i in range(steps):
  ids=rng.choice(len(Y),size=1024,replace=True,p=prob);params,state,loss=step(params,state,jnp.asarray(X[ids]),jnp.asarray(Y[ids]),jax.random.fold_in(key,i+1))
  if(i+1)%300==0 or i+1==steps:
   value=float(loss);assert np.isfinite(value),('NONFINITE_TRAINING',task,i);history.append(dict(step=i+1,loss=value,elapsed_seconds=time.monotonic()-begin));write(folder/'history.json',history)
 checkpoint=folder/'checkpoint.msgpack';checkpoint.write_bytes(serialization.to_bytes(params));r=dict(utc=utc(),started_utc=start,elapsed_seconds=time.monotonic()-begin,task=task,steps=steps,batch_size=1024,checkpoint=str(checkpoint),sha256=sha(checkpoint),dataset_path=definition['path'],dataset_sha256=definition['sha256'],source_scenes=definition['names'],no_early_stopping=True,no_seed_selection=True,validation_labels_used=False,initialization_seed=seed,sampling_seed=seed,logit_ensemble=True,implementation_sources={str(HERE/n):sha(HERE/n)for n in ['train_critic.py','critic_models.py','pool_features.py']})
 write(receipt,r);print('FIT',task['id'],seed,steps,round(r['elapsed_seconds'],2),flush=True);return r
if __name__=='__main__':fit(json.loads(sys.argv[1]))
