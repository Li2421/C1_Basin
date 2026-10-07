"""Matched2 versus4 source-controller supervision, no architecture change."""
import argparse,json,os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from pathlib import Path
import numpy as np
from scipy.special import expit
from .controller_diversity import OUT,SOURCE,RULE,read,write
from . import support_train as old
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import sha,csvwrite
KINDS=('eta_only','entity_response_mean','entity_response_attached');ARMS=('two_controller','four_controller');SEEDS=(17,23,41)

def load():
    d=dict(np.load(OUT/'dataset.npz'));x=dict(np.load(OUT/'frozen_model_entities.npz'))
    assert d['valid'].all()
    _,_,_,_,norm=old.load('expanded206')
    d['normalized_eta']=(d['eta']-norm['eta_center'])/norm['eta_scale']
    cc=(d['context']-norm['context_center'])/norm['context_scale']
    ar=(d['agent_response']-norm['agent_center'])/norm['agent_scale']
    d['input_context']=np.concatenate([cc,ar.reshape(4,len(d['eta']),-1)],-1).astype(np.float32)
    return d,x,norm

def train(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert jax.default_backend()=='gpu'
    arm=ARMS[index//9];kind=KINDS[(index%9)//3];seed=SEEDS[index%3]
    dest=OUT/'models'/arm/kind/f'seed{seed}'
    if (dest/'complete.json').exists():return
    d,x,norm=load();si=d['state_index'];fit=np.flatnonzero(d['split']=='train');val=np.flatnonzero(d['split']=='validation')
    eta=d['normalized_eta'];ctx=d['input_context'];m=old.model(kind)
    p=m.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,88)))
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));o=opt.init(p)
    @jax.jit
    def step(p,o,xx,e,c,s,f):
        def loss(pp):
            z=m.apply(pp,xx,e,c);q=s/jnp.maximum(s+f,1)
            return jnp.mean(q*jax.nn.softplus(-z)+(1-q)*jax.nn.softplus(z))
        v,g=jax.value_and_grad(loss)(p);u,o=opt.update(g,o,p);return optax.apply_updates(p,u),o,v
    predict=jax.jit(lambda p,x,e,c:m.apply(p,x,e,c))
    def evaluate(p,indices):
        values=[]
        for c in range(4):
            values.append(np.asarray(predict(p,gather(x,si[indices]),jnp.asarray(eta[indices],jnp.float32),jnp.asarray(ctx[c,indices],jnp.float32))))
        z=np.array(values);s=d['success'][:,indices];f=d['failure'][:,indices];n=s+f
        nll=(s*np.logaddexp(0,-z)+f*np.logaddexp(0,z)).sum(1)/n.sum(1)
        return dict(NLL=float(nll.mean()),NLL_by_controller=nll.tolist(),MAE=float(abs(expit(z)-s/n).mean())),z
    rng=np.random.default_rng(seed);history=[];best=(np.inf,None,0)
    controller_indices=np.array([0,1,0,1] if arm=='two_controller' else [0,1,2,3])
    for it in range(1,1501):
        draw=rng.choice(fit,32);ii=np.tile(draw,4);ci=np.repeat(controller_indices,32)
        p,o,loss=step(p,o,gather(x,si[ii]),jnp.asarray(eta[ii],jnp.float32),jnp.asarray(ctx[ci,ii],jnp.float32),
            jnp.asarray(d['success'][ci,ii]),jnp.asarray(d['failure'][ci,ii]))
        if it not in old.STEPS:continue
        vm,_=evaluate(p,val);tm,_=evaluate(p,fit);history.append(dict(step=it,train=tm,validation=vm))
        if vm['NLL']<best[0]-1e-5:best=(vm['NLL'],serialization.to_bytes(p),it)
    dest.mkdir(parents=True,exist_ok=True);(dest/'best.msgpack').write_bytes(best[1]);(dest/'last.msgpack').write_bytes(serialization.to_bytes(p))
    p=serialization.from_bytes(p,best[1]);vm,z=evaluate(p,val)
    np.savez_compressed(dest/'validation_predictions.npz',indices=val,logits=z)
    write(dest/'normalization.json',norm);write(dest/'history.json',history)
    write(dest/'complete.json',dict(arm=arm,kind=kind,seed=seed,best_step=best[2],best_source_VAL_NLL=best[0],
        validation_controllers=4,TRAIN_controllers=2 if arm=='two_controller' else 4,
        control_note='2-controller TRAIN uses4-controller VAL only for matched checkpoint selection; target88127 absent everywhere',
        dataset_sha256=sha(OUT/'dataset.npz'),rule_sha256=sha(RULE),code_sha256=sha(__file__),
        secondary_protocol_sha256=sha(OUT.parent/'controller_diversity_secondary_protocol.json'),
        target_controller_labels_used=False,parameter_count=int(sum(v.size for v in jax.tree.leaves(p)))))
    print(json.dumps(dict(arm=arm,kind=kind,seed=seed,step=best[2],**vm)),flush=True)

def freeze():
    if (OUT/'models_frozen.json').exists():raise FileExistsError('Models already frozen')
    models=[]
    for arm in ARMS:
      for kind in KINDS:
       for seed in SEEDS:
        p=OUT/'models'/arm/kind/f'seed{seed}';done=read(p/'complete.json')
        models.append(dict(size=arm,kind=kind,seed=seed,path=str(p),checkpoint_sha256=sha(p/'best.msgpack'),
            normalization_sha256=sha(p/'normalization.json'),source_VAL_NLL=done['best_source_VAL_NLL']))
    write(OUT/'models_frozen.json',dict(models=models,rule_sha256=sha(RULE),dataset_sha256=sha(OUT/'dataset.npz'),
        target_controller_not_created=True,primary='four_controller entity_response_mean versus matched eta-only and two_controller control; all3seeds',
        secondary='Entity-attached response factorial, same parameter count, declared before training and target creation',
        secondary_protocol_sha256=sha(OUT.parent/'controller_diversity_secondary_protocol.json'),
        target_TEST_labels_used=False))
    print(dict(frozen_models=len(models),target_seed=88127))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('train','freeze'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='freeze':freeze()
    else:train(a.index)
