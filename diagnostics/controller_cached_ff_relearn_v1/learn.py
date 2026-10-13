"""Fair source-only fits on paired cached FF labels; controller interaction ablation."""
from __future__ import annotations
import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
import optax
from diagnostics.controller_q_diversity_v1.train import Critic
from .study import HERE, read, write, freeze, sha


def load_inputs(kind):
    data = np.load(HERE / 'labels.npz')
    features = np.load(HERE / 'features.npz')
    # Row order state, controller; retain all sixteen candidate eta.
    x = {k[7:]:jnp.asarray(np.repeat(features[k],2,axis=0),jnp.float32)
         for k in features.files if k.startswith('entity_')}
    compact = features['branch_compact']
    base, alternate = compact[:,0], compact[:,1]
    future = np.stack((base, alternate),1)
    current = np.repeat(base[:,None],2,axis=1)
    if kind in ('eta_only','state_only'):
        controller = np.zeros(current.shape[:-1] + (14,), np.float32)
    elif kind == 'current_field':
        controller = current
    elif kind == 'branch_nominal':
        controller = np.concatenate((current[...,:14],future[...,:14]),-1)
    else:
        controller = np.concatenate((current,future),-1)
    controller = controller.reshape((-1,)+controller.shape[2:])
    train = np.repeat(data['split']=='train',2)
    mask = np.asarray(x['agent_mask'],bool)
    vals = controller[train][mask[train]]
    mean, scale = vals.mean(0), np.maximum(vals.std(0),.05)
    norm = dict(mean=mean.tolist(),scale=scale.tolist(),clip=10.,split='train')
    controller = np.clip((controller-mean)/scale,-10,10)*mask[...,None]
    c = jnp.asarray(controller,jnp.float32)
    e = jnp.asarray(data['eta'],jnp.float32)
    return data,x,c,e,norm


def metrics(z, success, failure):
    q = np.asarray(jax.nn.sigmoid(jnp.asarray(z)),float)
    count = success+failure; valid=count>0
    truth = success/np.maximum(count,1)
    mse = float(np.mean((q-truth)[valid]**2))
    nll = float(np.mean(((success*np.logaddexp(0,-z)+failure*np.logaddexp(0,z))/np.maximum(count,1))[valid]))
    paired = valid[:,0]&valid[:,1]
    delta_error = (q[:,1]-q[:,0])-(truth[:,1]-truth[:,0])
    dmse = float(np.mean(delta_error[paired]**2))
    centered = delta_error - (delta_error*paired).sum(-1,keepdims=True)/np.maximum(paired.sum(-1,keepdims=True),1)
    imse = float(np.mean(centered[paired]**2))
    selection=np.where(valid,q,-np.inf).argmax(-1)
    selected=np.take_along_axis(truth,selection[...,None],-1)[...,0]
    oracle=np.where(valid,truth,-np.inf).max(-1)
    good=valid.any(-1)
    regret=float(np.mean((oracle-selected)[good]))
    a,b=np.triu_indices(16,1)
    diff=truth[...,a]-truth[...,b]; pred=q[...,a]-q[...,b]
    strong=(np.abs(diff)>=.375)&valid[...,a]&valid[...,b]
    ranking=float(np.mean((pred[strong]*diff[strong]>0)+.5*(pred[strong]==0))) if strong.any() else None
    return dict(score=mse+dmse+imse,probability_mse=mse,controller_delta_mse=dmse,
        controller_eta_interaction_mse=imse,nll=nll,selection_regret_observed_eta=regret,pairwise_ranking_accuracy=ranking)


def fit(kind,seed):
    protocol=read(HERE/'protocol.json');assert kind in protocol['models'] and seed in protocol['fit_seeds']
    dest=HERE/'fits'/kind/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return
    data,x,c,e,norm=load_inputs(kind)
    assert sha(HERE/'labels.npz')==read(HERE/'cache_reuse_audit.json')['labels_sha256']
    train=np.flatnonzero(data['split']=='train');val=np.flatnonzero(data['split']=='validation')
    freeze(dest/'normalization.json',norm)
    model_kind=kind if kind in ('eta_only','state_only') else 'state_field'
    net=Critic(model_kind)
    gather=lambda idx:{k:v[idx] for k,v in x.items()}
    params=net.init(jax.random.PRNGKey(seed),gather(jnp.array([0])),c[:1],e[:1])
    optimizer=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(protocol['learning_rate'],weight_decay=protocol['weight_decay']))
    opt=optimizer.init(params)
    s=jnp.asarray(data['success'],jnp.float32);f=jnp.asarray(data['failure'],jnp.float32)
    @jax.jit
    def inference(p,ii,jj):return net.apply(p,gather(ii),c[ii],e[jj])
    def predict(p,states):
        rows=(states[:,None]*2+np.arange(2)).ravel()
        ii=np.repeat(rows,16);jj=np.tile(np.arange(16),len(rows));result=[]
        for start in range(0,len(ii),256):
            result.append(np.asarray(inference(p,jnp.asarray(ii[start:start+256]),jnp.asarray(jj[start:start+256]))))
        return np.concatenate(result).reshape(len(states),2,16)
    @jax.jit
    def block(params,opt,draws):
        def step(carry,states):
            params,opt=carry
            rows=(states[:,None]*2+jnp.arange(2)).reshape(-1)
            ii=jnp.repeat(rows,16);jj=jnp.tile(jnp.arange(16),rows.shape[0])
            success=s[states];failure=f[states];count=success+failure
            valid=count>0;truth=success/jnp.maximum(count,1)
            def loss(pp):
                z=net.apply(pp,gather(ii),c[ii],e[jj]).reshape(states.shape[0],2,16)
                terms=(success*jax.nn.softplus(-z)+failure*jax.nn.softplus(z))/jnp.maximum(count,1)
                value=jnp.sum(terms)/jnp.maximum(valid.sum(),1)
                if kind=='branch_field_contrast':
                    q=jax.nn.sigmoid(z);paired=valid[:,0]&valid[:,1]
                    error=(q[:,1]-q[:,0])-(truth[:,1]-truth[:,0])
                    centered=error-jnp.sum(error*paired,-1,keepdims=True)/jnp.maximum(paired.sum(-1,keepdims=True),1)
                    value+=.5*jnp.sum((error**2+centered**2)*paired)/jnp.maximum(paired.sum(),1)
                return value
            value,grad=jax.value_and_grad(loss)(params)
            updates,opt=optimizer.update(grad,opt,params)
            return (optax.apply_updates(params,updates),opt),value
        return jax.lax.scan(step,(params,opt),draws)
    rng=np.random.default_rng(seed);best=None;history=[];t0=time.monotonic()
    for step in range(protocol['validation_every'],protocol['steps']+1,protocol['validation_every']):
        draws=rng.choice(train,(protocol['validation_every'],protocol['batch_states']))
        (params,opt),losses=block(params,opt,jnp.asarray(draws))
        z=predict(params,val);m=metrics(z,data['success'][val],data['failure'][val])
        row=dict(step=step,training_objective=float(jnp.mean(losses)),**m);history.append(row)
        key=(m['score'],m['selection_regret_observed_eta'],m['nll'],step)
        if best is None or key<best:
            best=key;best_row=row;best_bytes=serialization.to_bytes(params)
            np.savez_compressed(dest/'best_validation.npz',state_indices=val,logits=z)
        if step%1000==0:
            write(dest/'history.json',history)
            print(dict(kind=kind,seed=seed,step=step,best_step=best_row['step'],score=m['score'],seconds=time.monotonic()-t0),flush=True)
    (dest/'checkpoint.msgpack').write_bytes(best_bytes)
    chosen=serialization.from_bytes(params,best_bytes)
    all_z=predict(chosen,np.arange(len(data['split'])))
    np.savez_compressed(dest/'predictions.npz',logits=all_z)
    write(dest/'history.json',history)
    write(dest/'complete.json',dict(kind=kind,seed=seed,selected=best_row,
        train=metrics(all_z[train],data['success'][train],data['failure'][train]),
        protocol_sha256=sha(HERE/'protocol.json'),labels_sha256=sha(HERE/'labels.npz'),
        feature_sha256=sha(HERE/'features.npz'),checkpoint_sha256=sha(dest/'checkpoint.msgpack'),
        fit_code_sha256=sha(__file__),shared_architecture_sha256=sha(__import__('diagnostics.controller_q_diversity_v1.train',fromlist=['']).__file__),
        development_labels_used_for_fitting_or_selection=False,seconds=time.monotonic()-t0))


def freeze_models():
    p=read(HERE/'protocol.json');rows=[]
    if (HERE/'root_cause_scope_amendment.json').exists():
        p['models']=read(HERE/'root_cause_scope_amendment.json')['models']
    for kind in p['models']:
        for seed in p['fit_seeds']:
            dest=HERE/'fits'/kind/f'seed{seed}';r=read(dest/'complete.json')
            assert r['checkpoint_sha256']==sha(dest/'checkpoint.msgpack')
            rows.append(r)
    scores={kind:float(np.mean([r['selected']['score'] for r in rows if r['kind']==kind])) for kind in p['models'] if kind.startswith('branch_')}
    primary=min(scores,key=scores.get)
    freeze(HERE/'models_frozen.json',dict(primary=primary,selection_scores=scores,records=rows,
        selection='Source validation only; all 3 seeds used as equal probability ensemble',
        scope_amendment=read(HERE/'root_cause_scope_amendment.json') if (HERE/'root_cause_scope_amendment.json').exists() else None,
        development_data='Previously inspected diagnostic, NOT a final confirmation set'))
    print(dict(primary=primary,selection_scores=scores),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fit','freeze']);p.add_argument('--kind');p.add_argument('--seed',type=int)
    a=p.parse_args()
    if a.action=='fit':fit(a.kind,a.seed)
    else:freeze_models()
