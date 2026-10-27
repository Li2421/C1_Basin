"""Preserved v1 fit loop; uniform candidates and anchored bounded critic output."""
from common import *
from networks import Generator,critic,stable_int,log_prob,CENTER,RADIUS
from train_model import dataset,gen_batch,validation_arrays as original_validation_arrays
from anchored import logit
from flax import serialization
import jax,jax.numpy as jnp,optax,numpy as np

def crit_batch(d,rng,etas,bias):
    i=d['train'][rng.integers(len(d['train']),size=96)]
    ctl=rng.integers(d['y'].shape[1],size=96)
    a=rng.integers(49,size=96)
    return d['h'][i],d['c'][i],d['onehot'][ctl],((etas[a]-CENTER)/RADIUS).astype(np.float32),d['y'][i,ctl,a],bias[a]

def validation_arrays(data,kind,etas,bias):
    original=original_validation_arrays(data,kind,etas)
    if kind=='generator':return original
    b=np.concatenate([np.tile(bias,len(d['val'])*d['y'].shape[1])for d in data.values()])
    assert len(b)==len(original[-1])
    return (*original,jnp.asarray(b,jnp.float32))

def fit(task):
    fold=read(HERE/'folds'/task['fold']/'definition.json');kind=task['kind'];mode=task.get('mode','full');seed=task['seed']
    if mode=='raw'and fold['descriptor']!='c_raw':fold=read(HERE/'folds'/task['fold']/'raw_definition.json')
    priors=read(HERE/'folds'/task['fold']/'PRIORS.json');bias=np.asarray(logit(priors['p0']),np.float32)
    folder=HERE/'folds'/task['fold']/mode/f'seed{seed}';receipt=folder/'FIT.receipt.json'
    if receipt.exists():
        row=read(receipt);assert sha(row['checkpoint'])==row['sha256'];return row
    folder.mkdir(parents=True,exist_ok=True);data=dataset(fold);etas=np.asarray([a['eta_a']for a in actions()],np.float32)
    model=Generator()if kind=='generator'else critic('full')
    # The historical train_generator/train_critic merge_initialized use key0;
    # seeds17/23/41 change their independent per-scene sampling streams.
    dummy=[jnp.zeros((1,6)),jnp.zeros((1,12))]
    if kind=='critic':dummy.extend([jnp.zeros((1,8)),jnp.zeros((1,3))])
    params=model.init(jax.random.PRNGKey(0),*dummy)
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(5e-4 if kind=='generator'else 1e-3,weight_decay=1e-4));state=opt.init(params)
    @jax.jit
    def step(p,st,args):
        def objective(q):
            if kind=='generator':
                h,c,ep,en,hn=args;raw=model.apply(q,h,c);lp=log_prob(raw,ep);ln=log_prob(raw,en)
                rank=jnp.sum(hn*jax.nn.softplus(.5+ln-lp),axis=1)/jnp.maximum(hn.sum(1),1)
                return jnp.mean(-lp.mean(1)+.25*rank)
            h,c,ct,e,y,p0bias=args;raw=model.apply(q,h,c,ct,e)
            logits=raw if mode=='no_prior'else p0bias+3*jnp.tanh(raw/3)
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits,y))
        loss,grad=jax.value_and_grad(objective)(p);updates,st=opt.update(grad,st,p)
        return optax.apply_updates(p,updates),st,loss
    va=validation_arrays(data,kind,etas,bias)
    @jax.jit
    def validation(p):
        if kind=='critic':
            h,c,ct,e,y,w,p0bias=va;raw=model.apply(p,h,c,ct,e)
            logits=raw if mode=='no_prior'else p0bias+3*jnp.tanh(raw/3)
            return jnp.sum(w*optax.sigmoid_binary_cross_entropy(logits,y))
        h,c,e,w,rh,rc,rp,rn,rw=va;raw=model.apply(p,h,c);r=model.apply(p,rh,rc)
        return -jnp.sum(w*log_prob(raw,e))+.25*jnp.sum(rw*jax.nn.softplus(.5+log_prob(r,rn)-log_prob(r,rp)))
    rngs={n:np.random.default_rng(stable_int(kind,seed,n))for n in data};best=(float('inf'),None,0);history=[];stale=0
    max_steps=3000 if kind=='generator'else 4000;patience=8 if kind=='generator'else 10;minimum=1200 if kind=='generator'else 1500
    before=time.monotonic()
    for iteration in range(1,max_steps+1):
        batches=[gen_batch(d,rngs[n],etas)if kind=='generator'else crit_batch(d,rngs[n],etas,bias)for n,d in data.items()]
        args=tuple(jnp.asarray(np.stack([v[k]for v in batches]))for k in range(len(batches[0])))
        params,state,loss=step(params,state,args)
        if iteration%100==0:
            val=float(validation(params));assert np.isfinite(val),(task,'NONFINITE_VALIDATION',val)
            history.append(dict(step=iteration,train=float(loss),validation=val,elapsed_seconds=time.monotonic()-before))
            if val<best[0]-1e-5:best=(val,serialization.to_bytes(params),iteration);stale=0
            else:stale+=1
            write(folder/'history.json',history)
            write(folder/'LIVE.json',dict(step=iteration,best_step=best[2],best_validation=best[0],stale=stale,elapsed_seconds=time.monotonic()-before))
            if stale>=patience and iteration>=minimum:break
    checkpoint=folder/'checkpoint.msgpack';checkpoint.write_bytes(best[1])
    row=dict(utc=utc(),fold=task['fold'],kind=kind,mode=mode,seed=seed,validation_loss=best[0],best_step=best[2],steps=iteration,
        elapsed_seconds=time.monotonic()-before,checkpoint=str(checkpoint),sha256=sha(checkpoint),
        training_scenes=fold['sources'],heldout_scenes=fold['targets'],training_data_sha256=fold['training_data_sha256'],
        normalization=fold['normalization'],initialization_key=0,source_only_early_stopping=True,heldout_labels_used=False,
        implementation_sources={str(HERE/n):sha(HERE/n)for n in ('train_model.py','train_v2.py','anchored.py','descriptors_v2.py','networks.py')})
    write(receipt,row);print('FIT',task['fold'],mode,seed,iteration,'steps',round(row['elapsed_seconds'],2),'seconds',flush=True)
    return row

if __name__=='__main__':fit(json.loads(sys.argv[1]))
