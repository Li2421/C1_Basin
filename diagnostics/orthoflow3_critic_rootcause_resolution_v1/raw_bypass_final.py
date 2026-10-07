"""Freeze source-selected models and exact source-only eta-prior control."""
import argparse,hashlib,os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from scipy.special import logit
from . import context_raw_bypass as skip
from . import state_breadth_train as data
from . import motion_factorial_train as base
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'raw_bypass_final_models'
RULE=ROOT/'raw_bypass_confirmation_protocol.json'
read,write,sha=base.read,base.write,base.sha


def train(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization,traverse_util
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert jax.default_backend()=='gpu'
    variant=('expanded_reference','raw_bypass')[index//3];seed=base.SEEDS[index%3]
    selection=read(skip.OUT/'selection_frozen.json')
    stop=selection['selected' if variant=='raw_bypass' else 'reference']['step']
    d,x,tr,va,norm=data.load(list(range(12)));si,e,c=d['state_index'],d['normalized_eta'],d['input_context']
    dest=OUT/variant/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'complete.json').exists()
    m=skip.model();old=base.model('rest_motion')
    args=(gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,120)))
    p=m.init(jax.random.PRNGKey(seed),*args)
    p['params']['core'].update(old.init(jax.random.PRNGKey(seed),*args)['params']['core'])
    initial=hashlib.sha256(serialization.to_bytes(p)).hexdigest()
    fk=traverse_util.flatten_dict(p)
    masks={mode:traverse_util.unflatten_dict({k:(k[-2]!='raw_context_skip' if mode=='warm' else k[-2] in ('trunk1','trunk2','out') or (variant=='raw_bypass' and k[-2]=='raw_context_skip')) for k in fk}) for mode in ('warm','fixed')}
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));o=opt.init(p)
    def make_step(mask):
        @jax.jit
        def step(pp,oo,xx,ee,cc,s,f):
            def loss(params):
                z=m.apply(params,xx,ee,cc);q=s/jnp.maximum(s+f,1)
                return jnp.mean(q*jax.nn.softplus(-z)+(1-q)*jax.nn.softplus(z))
            v,g=jax.value_and_grad(loss)(pp)
            g=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),g,mask)
            up,oo=opt.update(g,oo,pp);up=jax.tree.map(lambda a,b:a if b else jnp.zeros_like(a),up,mask)
            return optax.apply_updates(pp,up),oo,v
        return step
    warm,fixed=make_step(masks['warm']),make_step(masks['fixed'])
    rng=np.random.default_rng(seed);drawhash=hashlib.sha256()
    for it in range(1,401):
        u=rng.random(32);drawhash.update(u.tobytes())
        old_u=np.where(u<.4,u/.4,(u-.4)/.6)
        oi=np.minimum((old_u*128).astype(int),127);ni=np.minimum((u*320).astype(int),319)
        ii=np.concatenate([ni if ci<12 else oi for ci in range(16)]);ci=np.repeat(np.arange(16),32)
        p,o,v=(warm if it<=25 else fixed)(p,o,gather(x,si[ii]),jnp.asarray(e[ii]),jnp.asarray(c[ci,ii]),jnp.asarray(d['success'][ci,ii]),jnp.asarray(d['failure'][ci,ii]))
        if it in (25,stop):(dest/f'step{it}.msgpack').write_bytes(serialization.to_bytes(p))
    selected=serialization.msgpack_restore((dest/f'step{stop}.msgpack').read_bytes())
    predict=jax.jit(lambda pp,xx,ee,cc:m.apply(pp,xx,ee,cc))
    z=np.array([np.asarray(predict(selected,gather(x,si[va]),jnp.asarray(e[va]),jnp.asarray(c[ci,va]))) for ci in range(16)])
    np.savez_compressed(dest/'predictions.npz',**{f'step{stop}':z})
    write(dest/'normalization.json',norm)
    write(dest/'complete.json',dict(variant=variant,seed=seed,steps=stop,training_steps=400,
        initial_parameters_sha256=initial,draws_sha256=drawhash.hexdigest(),dataset_sha256=sha(data.OUT/'dataset.npz'),
        code_sha256=sha(__file__),protocol_sha256=sha(RULE),target_labels_used=False,new_rollouts=0))
    print(dict(variant=variant,seed=seed,frozen_step=stop),flush=True)


def freeze():
    from flax import serialization,traverse_util
    models=[];parity=[]
    for seed in base.SEEDS:
        docs=[];warm=[]
        for variant in ('expanded_reference','raw_bypass'):
            dest=OUT/variant/f'seed{seed}';doc=read(dest/'complete.json');docs.append(doc)
            assert doc['code_sha256']==sha(__file__) and doc['dataset_sha256']==sha(data.OUT/'dataset.npz')
            warm.append(traverse_util.flatten_dict(serialization.msgpack_restore((dest/'step25.msgpack').read_bytes())))
            ck=f'step{doc["steps"]}.msgpack'
            models.append(dict(variant=variant,seed=seed,path=str(dest),checkpoint=ck,steps=doc['steps'],
                checkpoint_sha256=sha(dest/ck),normalization_sha256=sha(dest/'normalization.json'),diagnostic_only=False))
        for key in ('initial_parameters_sha256','draws_sha256'):assert docs[0][key]==docs[1][key]
        error=max(float(np.max(abs(warm[0][k]-warm[1][k]))) for k in warm[0])
        assert error<2e-6;parity.append(dict(seed=seed,warm25_max_error=error))
    old=read(ROOT/'motion_final_source_models/models_frozen.json')
    for entry in old['models']:
        if entry['variant'] in ('eta_only','trunk_only'):
            models.append({**entry,'variant':'old_trunk' if entry['variant']=='trunk_only' else 'eta_only'})
    d,_,_,_,norm=data.load(list(range(12)))
    q=d['success']/np.maximum(d['success']+d['failure'],1)
    prior=np.mean([q[ci,:320 if ci<12 else 128].reshape(-1,2).mean(0) for ci in range(16)],0)
    etas=d['normalized_eta'][:2];direction=etas[1]-etas[0]
    zs=logit(np.clip(prior,1e-6,1-1e-6));coef=(zs[1]-zs[0])*direction/np.dot(direction,direction)
    intercept=zs[0]-etas[0]@coef
    ref=models[0]
    models.append({**ref,'variant':'eta_mle','seed':0,'steps':0,'analytic_prediction':True})
    frozen=dict(models=models,eta_prior=dict(probabilities=prior.tolist(),normalized_eta_coefficient=coef.tolist(),
        intercept=float(intercept),criterion='Exact controller-balanced W1 TRAIN likelihood for the two observed eta; continuous linear-logit extension; not unique outside this tested support'),
        source_selection_sha256=sha(skip.OUT/'selection_frozen.json'),source_data_sha256=sha(data.OUT/'dataset.npz'),
        protocol_sha256=sha(RULE),code_sha256=sha(__file__),warm_parity=parity,target_labels_used=False,
        generator_changed=False,notes='Target source/model/checkpoint freeze precedes new Flow construction and all target labels.')
    assert not (OUT/'models_frozen.json').exists();write(OUT/'models_frozen.json',frozen)
    print(dict(models=len(models),eta_prior=prior.tolist(),frozen=True),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('train','freeze'));p.add_argument('--index',type=int,default=0)
    a=p.parse_args();train(a.index) if a.action=='train' else freeze()
