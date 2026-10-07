"""Four versus twelve functions; same family/pair draws and W1 objective."""
import argparse,os,json
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.special import expit
from .function_support import OUT,RULE,read,write
from . import support_train as old
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import sha,csvwrite
SEEDS=(17,23,41);KINDS=('eta_only','entity_response_mean');ARMS=('four_controller','twelve_controller','reweighted_four_controller')

def load():
    d=dict(np.load(OUT/'dataset.npz'));x=dict(np.load(OUT/'frozen_model_entities.npz'));fit=np.flatnonzero(d['split']=='train');val=np.flatnonzero(d['split']=='validation')
    ec=d['eta'][fit].mean(0);es=np.maximum(d['eta'][fit].std(0),.1)
    cc=d['context'][:4,fit].mean((0,1));cs=np.maximum(d['context'][:4,fit].std((0,1)),.05)
    ac=d['agent_response'][:4,fit].mean((0,1,2));ast=np.maximum(d['agent_response'][:4,fit].std((0,1,2)),.01)
    norm=dict(eta_center=ec.tolist(),eta_scale=es.tolist(),context_center=cc.tolist(),context_scale=cs.tolist(),agent_center=ac.tolist(),agent_scale=ast.tolist(),
        normalization_controllers=[0,1,2,3],source_TRAIN_only=True)
    d['normalized_eta']=(d['eta']-ec)/es
    d['input_context']=np.concatenate([(d['context']-cc)/cs,((d['agent_response']-ac)/ast).reshape(12,len(d['eta']),-1)],-1).astype(np.float32)
    return d,x,fit,val,norm

def train(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert jax.default_backend()=='gpu'
    arm=ARMS[index//6];kind=KINDS[(index%6)//3];seed=SEEDS[index%3]
    controllers=list(range(12)) if arm=='twelve_controller' else [0,1,2,3] if arm=='four_controller' else [0,1,2,2,2,2,2,3,3,3,3,3]
    dest=OUT/'models'/arm/kind/f'seed{seed}'
    if (dest/'complete.json').exists():return
    d,x,fit,val,norm=load();si=d['state_index'];eta=d['normalized_eta'];ctx=d['input_context'];m=old.model(kind)
    p=m.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,88)))
    opt=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));o=opt.init(p)
    @jax.jit
    def step(p,o,xx,e,c,s,f):
        def loss(pp):
            z=m.apply(pp,xx,e,c);q=s/jnp.maximum(s+f,1)
            return jnp.mean(q*jax.nn.softplus(-z)+(1-q)*jax.nn.softplus(z))
        v,g=jax.value_and_grad(loss)(p);u,o=opt.update(g,o,p);return optax.apply_updates(p,u),o,v
    pred=jax.jit(lambda p,x,e,c:m.apply(p,x,e,c))
    def evaluate(p,cs,ii,condition='correct'):
        shifted=np.roll(ii.reshape(-1,2),-1,axis=0).ravel()
        state=si[shifted] if condition in ('state_shuffle','joint_state_context_shuffle') else si[ii]
        cidx=shifted if condition in ('context_state_shuffle','joint_state_context_shuffle') else ii
        z=np.array([np.asarray(pred(p,gather(x,state),jnp.asarray(eta[ii],jnp.float32),jnp.asarray(ctx[(c+1)%12 if condition=='wrong_controller' else c,cidx],jnp.float32))) for c in cs])
        s=d['success'][cs][:,ii];f=d['failure'][cs][:,ii];n=s+f
        return dict(NLL=float(((s*np.logaddexp(0,-z)+f*np.logaddexp(0,z))/n).mean()),MAE=float(abs(expit(z)-s/n).mean())),z
    rng=np.random.default_rng(seed);best=(np.inf,None,0);history=[]
    cis=np.tile(np.arange(4),3) if arm=='four_controller' else np.arange(12) if arm=='twelve_controller' else np.array(controllers)
    for it in range(1,1501):
        draw=rng.choice(fit,32);ii=np.tile(draw,12);ci=np.repeat(cis,32)
        p,o,loss=step(p,o,gather(x,si[ii]),jnp.asarray(eta[ii],jnp.float32),jnp.asarray(ctx[ci,ii],jnp.float32),
            jnp.asarray(d['success'][ci,ii]),jnp.asarray(d['failure'][ci,ii]))
        if it not in old.STEPS:continue
        vm,_=evaluate(p,controllers,val);tm,_=evaluate(p,controllers,fit);history.append(dict(step=it,train=tm,validation=vm))
        if vm['NLL']<best[0]-1e-5:best=(vm['NLL'],serialization.to_bytes(p),it)
    dest.mkdir(parents=True,exist_ok=True);(dest/'best.msgpack').write_bytes(best[1]);write(dest/'normalization.json',norm);write(dest/'history.json',history)
    p=serialization.from_bytes(p,best[1]);predictions={}
    for condition in ('correct','wrong_controller','state_shuffle','context_state_shuffle','joint_state_context_shuffle'):
        _,z=evaluate(p,list(range(12)),val,condition);predictions[condition]=z
    np.savez_compressed(dest/'validation_predictions.npz',indices=val,**predictions)
    write(dest/'complete.json',dict(arm=arm,kind=kind,seed=seed,best_step=best[2],best_source_VAL_NLL=best[0],
        fit_controllers=controllers,selection_controllers=controllers,extra_controllers_are_heldout_for_four_controller_arm=True,
        rule_sha256=sha(RULE),dataset_sha256=sha(OUT/'dataset.npz'),code_sha256=sha(__file__),
        reweight_control_sha256=sha(OUT/'reweight_control_preregistration.json'),target_labels_used=False))
    print(dict(arm=arm,kind=kind,seed=seed,best_step=best[2],NLL=best[0]),flush=True)

def freeze():
    if (OUT/'models_frozen.json').exists():raise FileExistsError('Models frozen')
    local=OUT/'local_diagnostic';assert (local/'complete.json').exists()
    models=[]
    for arm in ARMS:
      for kind in KINDS:
       for seed in SEEDS:
        path=OUT/'models'/arm/kind/f'seed{seed}';done=read(path/'complete.json')
        models.append(dict(size=arm,kind=kind,seed=seed,path=str(path),checkpoint_sha256=sha(path/'best.msgpack'),
            normalization_sha256=sha(path/'normalization.json'),source_VAL_NLL=done['best_source_VAL_NLL']))
    write(OUT/'models_frozen.json',dict(models=models,rule_sha256=sha(RULE),dataset_sha256=sha(OUT/'dataset.npz'),
        primary='Twelve-controller mean-response versus matched four-controller and eta-only; all3seeds',
        secondary='Reweighted4-controller control matches10of12v10expertrecipe mass without addingfunctions',
        reweight_control_sha256=sha(OUT/'reweight_control_preregistration.json'),target_not_created=True,target_labels_used=False,
        local_diagnostic={name:sha(local/name) for name in ('knn.npz','normalization.json','complete.json')},
        independent_confirmation_protocol_sha256=sha(OUT/'independent_confirmation_protocol.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('train','freeze'));p.add_argument('--index',type=int,default=0);a=p.parse_args()
    if a.action=='train':train(a.index)
    else:freeze()
