"""Identical crossed labels, source-only normalization and Q-focused selection."""
from __future__ import annotations
import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import serialization
import optax
from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
from .experiment import HERE,read,write,freeze,sha

KINDS=('eta_only','state_only','context_only','state_context','field_only','state_field')
SEEDS=(17,23,41)


class Critic(nn.Module):
    kind:str
    @nn.compact
    def __call__(self,x,c,e):
        parts=[]
        if self.kind!='eta_only':
            if self.kind in ('state_only','state_context','state_field'):
                xx=dict(x)
                cc=c if self.kind=='state_field' else c[...,:14] if self.kind=='state_context' else jnp.zeros_like(c[...,:14])
                xx['agents']=jnp.concatenate((xx['agents'],cc),-1)
                parts.append(rep.Encoder(name='physical_encoder')(xx))
            else:
                cc=c if self.kind=='field_only' else c[...,:14]
                h=nn.silu(nn.Dense(64,name='context1')(cc))
                h=nn.silu(nn.Dense(64,name='context2')(h))
                parts.append(rep.pool(h,x['agent_mask'],1))
        parts.append(nn.silu(nn.Dense(32,name='eta')(e)))
        h=nn.silu(nn.Dense(128,name='shared1')(jnp.concatenate(parts,-1)))
        h=nn.silu(nn.Dense(64,name='shared2')(h))
        return nn.Dense(1,name='out')(h)[...,0]


def normalization(d):
    c=np.concatenate((d['nominal'],d['field']),-1)
    train=(d['split']=='train');mask=d['entity_agent_mask'].astype(bool)
    values=c[train][mask[train]]
    mean=values.mean(0);scale=np.maximum(values.std(0),.05)
    return dict(mean=mean.tolist(),scale=scale.tolist(),fit_split='train',clip=10.)


def inputs(d,norm):
    c=np.concatenate((d['nominal'],d['field']),-1)
    c=np.clip((c-np.asarray(norm['mean']))/np.asarray(norm['scale']),-norm['clip'],norm['clip'])
    c*=d['entity_agent_mask'][...,None]
    x={k[7:]:jnp.asarray(d[k],jnp.float32) for k in d.files if k.startswith('entity_')}
    return x,jnp.asarray(c,jnp.float32),jnp.asarray(d['eta'],jnp.float32)


def metrics(z,d,ix):
    q=np.asarray(jax.nn.sigmoid(jnp.asarray(z)),float)
    s=d['success'][ix];f=d['failure'][ix];u=d['unknown'][ix];v=s+f
    y=np.divide(s,v,out=np.zeros_like(s,dtype=float),where=v>0)
    valid=v>0
    mse=float(np.sum((q-y)**2*valid)/valid.sum())
    nll=float(np.sum(s*np.logaddexp(0,-z)+f*np.logaddexp(0,z))/v.sum())
    state=d['state_uid'][ix];ctl=d['controller'][ix]
    contrasts=[]
    for sid in sorted(set(state)):
        ids=np.flatnonzero(state==sid)
        a,b=np.triu_indices(len(ids),1);a=ids[a];b=ids[b]
        good=valid[a]&valid[b]
        contrasts.extend(np.square((q[a]-q[b])-(y[a]-y[b]))[good].tolist())
    cmse=float(np.mean(contrasts))
    lo=s/(s+f+u);sel=q.argmax(1)
    reg=float(np.mean(lo.max(1)-lo[np.arange(len(ix)),sel]))
    a,b=np.triu_indices(16,1);direction=np.sign(y[:,a]-y[:,b]);mask=(np.abs(y[:,a]-y[:,b])>=.375)&valid[:,a]&valid[:,b]
    pred=np.sign(q[:,a]-q[:,b]);accuracy=float(np.mean((pred==direction)[mask]+.5*(pred==0)[mask])) if mask.any() else None
    held=np.isin(ctl,['analytic4','analytic5'])
    return dict(score=mse+.5*cmse,probability_mse=mse,controller_contrast_mse=cmse,nll=nll,
        selection_regret=reg,pairwise_accuracy=accuracy,
        held_controller_mse=float(np.mean(np.square(q[held]-y[held])[valid[held]])) if held.any() else None)


def fit(kind,seed,limit=128):
    assert kind in KINDS and seed in SEEDS
    plan=read(HERE/'training_plan.json');path=HERE/'data'/f'source{limit}'/'dataset.npz'
    manifest=read(path.parent/'manifest.json');assert sha(path)==manifest['dataset_sha256'] and not manifest['final_labels_used']
    d=np.load(path);train=np.flatnonzero(d['split']=='train');val=np.flatnonzero(d['split']=='validation')
    assert not np.isin(d['controller'][train],['analytic4','analytic5','analytic6','analytic7']).any()
    assert not np.isin(d['controller'],['analytic6','analytic7']).any()
    dest=HERE/'fits'/f'source{limit}'/kind/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    norm=normalization(d);freeze(dest/'normalization.json',norm)
    x,c,e=inputs(d,norm);s=jnp.asarray(d['success'],jnp.float32);f=jnp.asarray(d['failure'],jnp.float32)
    net=Critic(kind)
    gather=lambda ii:{k:v[ii] for k,v in x.items()}
    params=net.init(jax.random.PRNGKey(seed),gather(jnp.array([0])),c[:1],e[:1])
    optim=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(plan['learning_rate'],weight_decay=plan['weight_decay']))
    opt=optim.init(params);exposure=float(np.mean(np.asarray(s+f)[train]))
    @jax.jit
    def block(params,opt,draws):
        def step(carry,draw):
            p,o=carry;ii,jj=draw[:,0],draw[:,1]
            def loss(pp):
                z=net.apply(pp,gather(ii),c[ii],e[jj])
                return jnp.mean(s[ii,jj]*jax.nn.softplus(-z)+f[ii,jj]*jax.nn.softplus(z))/exposure
            value,gradient=jax.value_and_grad(loss)(p)
            updates,o=optim.update(gradient,o,p)
            return (optax.apply_updates(p,updates),o),value
        return jax.lax.scan(step,(params,opt),draws)
    @jax.jit
    def predict(p,ii,jj):return net.apply(p,gather(ii),c[ii],e[jj])
    def infer(p,rows):
        ii=np.repeat(rows,16);jj=np.tile(np.arange(16),len(rows));result=[]
        for start in range(0,len(ii),256):
            result.append(np.asarray(predict(p,jnp.asarray(ii[start:start+256]),jnp.asarray(jj[start:start+256]))))
        return np.concatenate(result).reshape(len(rows),16)
    pools={}
    for scene in sorted(set(d['scene'][train])):
        for ctl in sorted(set(d['controller'][train])):
            inds=train[(d['scene'][train]==scene)&(d['controller'][train]==ctl)]
            pools[scene,ctl]=np.asarray([(i,j) for i in inds for j in range(16) if d['success'][i,j]+d['failure'][i,j]],np.int32)
    rng=np.random.default_rng(seed);history=[];best=None;best_bytes=None;t0=time.monotonic()
    for step in range(plan['validation_every'],plan['steps']+1,plan['validation_every']):
        chunks=[pool[rng.integers(len(pool),size=(plan['validation_every'],plan['batch_per_scene_controller']))] for pool in pools.values()]
        (params,opt),losses=block(params,opt,jnp.asarray(np.concatenate(chunks,1)))
        z=infer(params,val);score=metrics(z,d,val);row=dict(step=step,training_nll=float(np.mean(losses)),**score)
        history.append(row)
        key=(score['score'],score['selection_regret'],score['nll'],step)
        if best is None or key<best:
            best=key;best_bytes=serialization.to_bytes(params);best_row=row
            np.savez_compressed(dest/'best_validation.npz',row_indices=val,logits=z)
        if step%1000==0:
            write(dest/'history.json',history)
            print(dict(kind=kind,seed=seed,step=step,score=score['score'],best_step=best_row['step'],seconds=time.monotonic()-t0),flush=True)
    (dest/'checkpoint.msgpack').write_bytes(best_bytes)
    chosen=serialization.from_bytes(params,best_bytes)
    np.savez_compressed(dest/'source_predictions.npz',logits=infer(chosen,np.arange(len(d['scene']))))
    write(dest/'history.json',history)
    write(dest/'complete.json',dict(kind=kind,seed=seed,train_limit=limit,selected=best_row,
        dataset_sha256=sha(path),checkpoint_sha256=sha(dest/'checkpoint.msgpack'),
        training_code_sha256=sha(__file__),plan_sha256=sha(HERE/'training_plan.json'),
        final_labels_used=False,seconds=time.monotonic()-t0))


def freeze_models(limit=128):
    records=[]
    for kind in KINDS:
        for seed in SEEDS:
            folder=HERE/'fits'/f'source{limit}'/kind/f'seed{seed}'
            row=read(folder/'complete.json');assert row['checkpoint_sha256']==sha(folder/'checkpoint.msgpack')
            records.append(dict(path=str(folder),**row))
    choices={k:np.mean([r['selected']['score'] for r in records if r['kind']==k]) for k in ('state_context','state_field')}
    primary=min(choices,key=choices.get)
    freeze(HERE/'models_frozen.json',dict(records=records,primary=primary,selection_scores=choices,
        ensemble='Mean probabilities of all three source-selected seeds, fixed before target labels',
        train_limit=limit,final_labels_opened=False,training_plan_sha256=sha(HERE/'training_plan.json')))
    print(dict(primary=primary,selection_scores=choices),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','freeze']);p.add_argument('--kind');p.add_argument('--seed',type=int);p.add_argument('--limit',type=int,default=128)
    a=p.parse_args()
    if a.action=='fit':fit(a.kind,a.seed,a.limit)
    else:freeze_models(a.limit)
