"""Original no-controller critic architecture; new physical12 state input.

Geometric coordinates 0:9 feed the 96-unit state adapter; coordinates 9:12
feed the 32-unit context adapter. Action19 and all training hyperparameters
are the frozen cross-type-six rules. No controller feature enters the net.
"""
from experiment import *
import itertools
import time
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import serialization
import optax
jax.config.update('jax_enable_x64',False)
SEEDS=(17,23,41)
STEPS=tuple(range(100,2001,100))

class Critic(nn.Module):
    @nn.compact
    def __call__(self,h,c,a):
        geom=nn.silu(nn.Dense(96,name='ring_adapter')(h));context=nn.silu(nn.Dense(32,name='context')(c));action=nn.silu(nn.Dense(32,name='action_encoder')(a))
        x=jnp.concatenate([geom,context,action],-1);x=nn.silu(nn.Dense(128,name='shared1')(x));x=nn.silu(nn.Dense(64,name='shared2')(x));return nn.Dense(1,name='out')(x)[...,0]

def source_key(source):return hashlib.sha256(json.dumps(sorted(source)).encode()).hexdigest()[:20]
def stable_int(*parts):return int(hashlib.sha256('\0'.join(map(str,parts)).encode()).hexdigest()[:16],16)
def transform(X):
    good=np.isfinite(X);mean=np.where(good,X,0).sum(0)/np.maximum(good.sum(0),1);filled=np.where(good,X,mean);scale=filled.std(0);scale=np.where(scale>1e-6,scale,1.)
    return dict(mean=mean.tolist(),scale=scale.tolist())
def apply(X,tr):
    mean=np.asarray(tr['mean']);return ((np.where(np.isfinite(X),X,mean)-mean)/np.asarray(tr['scale'])).astype(np.float32)

NET=Critic();OPT=optax.chain(optax.clip_by_global_norm(5),optax.adamw(1e-3,weight_decay=1e-4))
@jax.jit
def train_block(parameters,optimizer_states,indices,X,actions,Y,N):
    ns=X.shape[0];na=actions.shape[0]
    def one(p,st,ids):
        si=ids//na%ns;ai=ids%na
        def loss(pp):
            logit=NET.apply(pp,X[si,:9],X[si,9:],actions[ai]);n=N[ids];w=jnp.minimum(n,16.)
            return jnp.sum(w*optax.sigmoid_binary_cross_entropy(logit,Y[ids]/jnp.maximum(n,1)))/jnp.maximum(w.sum(),1)
        value,grad=jax.value_and_grad(loss)(p);delta,st=OPT.update(grad,st,p);return optax.apply_updates(p,delta),st,value
    def step(carry,ids):
        p,st=carry;p,st,value=jax.vmap(one)(p,st,ids);return (p,st),value
    return jax.lax.scan(step,(parameters,optimizer_states),indices)

@jax.jit
def predict(parameters,X,actions):
    ns=len(X);na=len(actions);h=jnp.repeat(X[:,:9],na,axis=0);c=jnp.repeat(X[:,9:],na,axis=0);a=jnp.tile(actions,(ns,1))
    return jax.vmap(lambda p:jax.nn.sigmoid(NET.apply(p,h,c,a)).reshape(ns,na))(parameters)

def weights(labels,types,source):
    w=np.zeros_like(labels,dtype=float)
    for t in source:
        ctls=np.flatnonzero(types==t)
        for ci in ctls:
            valid=labels[ci]>=0
            if valid.any():w[ci]=valid/(len(source)*len(ctls)*valid.sum())
    assert not np.any(w[~np.isin(types,source)])
    return w.reshape(-1)/w.sum()

def fit(scene,source,data,tr):
    folder=HERE/'critic_fits'/scene/source_key(source);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'COMPLETE.receipt.json').exists():return
    X=jnp.asarray(apply(data['physical12'],tr));actions=jnp.asarray(data['action']);labels=data['labels'];w=weights(labels,data['types'],source)
    Y=jnp.asarray(np.maximum(labels,0).reshape(-1),jnp.float32);N=jnp.asarray((labels>=0).astype(np.float32).reshape(-1))
    parameters=jax.tree_util.tree_map(lambda *a:jnp.stack(a),*[NET.init(jax.random.PRNGKey(seed),jnp.zeros((1,9)),jnp.zeros((1,3)),jnp.zeros((1,19))) for seed in SEEDS]);optstates=jax.vmap(OPT.init)(parameters)
    rngs=[np.random.default_rng(stable_int('crosstype1c-critic',source_key(source),'no_controller',s)) for s in SEEDS];history=[];files=[];last=0;start=time.monotonic()
    write(folder/'CONFIG.json',dict(source_types=source,seeds=SEEDS,transform=tr,architecture='cross-type-six no-controller; physical12 split9+3',steps=STEPS,batch=256,learning_rate=1e-3,weight_decay=1e-4,clip=5))
    for step in STEPS:
        ids=np.stack([r.choice(len(w),size=(step-last,256),p=w).astype(np.int32) for r in rngs],axis=1)
        (parameters,optstates),loss=train_block(parameters,optstates,jnp.asarray(ids),X,actions,Y,N)
        probabilities=np.asarray(predict(parameters,X,actions));checkpoint=folder/f'step{step}.msgpack';checkpoint.write_bytes(serialization.to_bytes(parameters));packet=folder/f'step{step}_predictions.npz';np.savez_compressed(packet,probabilities=probabilities)
        files.extend([checkpoint,packet]);history.append(dict(step=step,training_nll_by_seed=np.asarray(loss).mean(0).tolist(),seconds=time.monotonic()-start));last=step
    write(folder/'HISTORY.json',history);seal(str(folder.relative_to(HERE)/'COMPLETE.receipt.json'),[folder/'CONFIG.json',folder/'HISTORY.json',*files],source_only_gradient=True,outer_evaluation_labels_used=False)
    print(scene,len(source),source_key(source),round(time.monotonic()-start,2),flush=True)

def nll(probability,labels):
    p=np.clip(probability,1e-6,1-1e-6);valid=labels>=0;y=np.maximum(labels,0)
    loss=np.where(valid,-y*np.log(p)-(1-y)*np.log1p(-p),0)
    return float(np.mean(loss.sum((-1,-2))/np.maximum(1,valid.sum((-1,-2)))))

def main(scene):
    prereg();assert jax.default_backend()=='gpu';path=HERE/f'LEARNING_BUNDLE_{scene}.npz';receipt=read(HERE/f'LEARNING_BUNDLE_{scene}.receipt.json');assert receipt['files'][str(path)]==sha(path)
    with np.load(path) as z:data={k:z[k] for k in z.files}
    types=sorted(set(data['types'].tolist()));assert len(types)==10;tr=transform(data['physical12'])
    tasks=list(itertools.combinations(types,len(types)-2))+list(itertools.combinations(types,len(types)-1))
    plan=HERE/f'CRITIC_TRAINING_PLAN_{scene}.json'
    if not plan.exists():
        write(plan,dict(utc=utc(),source_sets=tasks,steps=STEPS,seeds=SEEDS,initialization='scratch',source_only_preprocessing=True,fold_validation='For held-out t, select checkpoint by mean NLL on each u!=t using fit excluding t and u; then deploy fit excluding t',no_controller_input=True,unknown_masked=True,geometry_features='physical12[0:9]',context_features='physical12[9:12]',action_dimension=19))
        seal(f'CRITIC_TRAINING_PLAN_{scene}.receipt.json',[plan,HERE/'learn_critic_extended.py',path,HERE/f'LEARNING_BUNDLE_{scene}.receipt.json'],before_training=True)
    with jax.default_matmul_precision('highest'):
        for source in tasks:fit(scene,list(source),data,tr)
    folds={}
    for target in types:
        scores=[]
        for step in STEPS:
            losses=[]
            for validation in types:
                if validation==target:continue
                source=sorted(set(types)-{target,validation});folder=HERE/'critic_fits'/scene/source_key(source)
                with np.load(folder/f'step{step}_predictions.npz') as z:prob=z['probabilities'].mean(0)
                mask=data['types']==validation;losses.append(nll(np.broadcast_to(prob,data['labels'][mask].shape),data['labels'][mask]))
            scores.append(float(np.mean(losses)))
        step=STEPS[int(np.argmin(scores))];source=sorted(set(types)-{target});folder=HERE/'critic_fits'/scene/source_key(source);checkpoint=folder/f'step{step}.msgpack'
        folds[target]=dict(source_types=source,step=step,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),transform=tr,nested_nll_by_step=scores,held_out_type_labels_used=False)
    write(HERE/f'FROZEN_SCENE_CRITIC_{scene}.json',dict(utc=utc(),folds=folds,features=12,controller_input=False,seed_ensemble=SEEDS,evidence='DEVELOPMENT'))
    seal(f'FROZEN_SCENE_CRITIC_{scene}.receipt.json',[HERE/f'FROZEN_SCENE_CRITIC_{scene}.json',plan,HERE/'learn_critic_extended.py'],before_evaluation_rollouts=True)

if __name__=='__main__':main(sys.argv[1])
