"""Source-only controlled response-window intervention; no target labels."""
import argparse,json,os,time
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import numpy as np
from scipy.special import expit
from .controller_diversity import OUT as SOURCE,read,write
from . import support_train as old
from .agent_response import instrument
from diagnostics.orthoflow3_state_context_learning_audit_v1.experiment import sha,csvwrite
OUT=SOURCE.parent/'response_horizon';RULE=SOURCE.parent/'response_horizon_protocol.json'
NAMES=('alt','second','extra_a','extra_b');SEEDS=(17,23,41)

def profiles():
    return read(SOURCE.parent/'state_support/protocol.json')['profiles']+read(SOURCE/'protocol.json')['profiles']

def features(index):
    import jax
    from diagnostics.orthoflow3_controller_intervention_generalization_v1 import rich_context as rc
    assert jax.default_backend()=='gpu'
    name=NAMES[index];dest=OUT/f'inputs_H80_{name}.npz'
    if dest.exists():raise FileExistsError('Frozen features already exist')
    rc.PROTOCOL={**rc.PROTOCOL,'horizon_steps':80}
    profile=next(p for p in profiles() if p['name']==name)
    rt=rc.RichRuntime('ring_exchange',profile['path']);trace=instrument(rt)
    physical=read(SOURCE/'physical.json');pairs=read(SOURCE/'pairs.json')
    ctx=np.zeros((len(pairs),24),np.float32);agent=np.zeros((len(pairs),4,16),np.float32)
    valid=np.zeros(len(pairs),bool);errors=[];timings=[];lengths=[]
    for i,p in enumerate(pairs):
        state=physical[p['state_index']];assert state['state_uid']==p['state_uid']
        assert p['split'] in ('train','validation')
        trace['runs']=[];t=time.perf_counter()
        try:
            result=rt.features(state['physical'],p['eta'])
            assert len(trace['runs'])==4 and all(1<=len(r)<=80 for r in trace['runs'])
            means=np.array([np.array(r).mean(0) for r in trace['runs']])
            ctx[i]=result['mean'];agent[i]=np.concatenate([means[[0,2]].mean(0),means[[1,3]].mean(0)],axis=-1)
            assert np.isfinite(ctx[i]).all() and np.isfinite(agent[i]).all()
            valid[i]=True;lengths.append([len(r) for r in trace['runs']])
        except Exception as exc:errors.append(dict(pair_index=i,error=f'{type(exc).__name__}:{exc}'))
        timings.append(time.perf_counter()-t)
        if (i+1)%32==0:print(dict(controller=name,pairs=i+1,invalid=len(errors)),flush=True)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(dest,context=ctx,agent_response=agent,valid=valid)
    write(OUT/f'input_audit_{name}.json',dict(controller=name,pairs=len(pairs),invalid=errors,probe_lengths=lengths,
        warm_timing_p50_seconds=float(np.median(timings[1:])),warm_timing_p95_seconds=float(np.quantile(timings[1:],.95)),
        timing_scope='Real sequential2roots nominal+corrected physical probe, including device result materialization; not a vectorized deployment benchmark',
        source_controller_sha256=sha(profile['path']),rule_sha256=sha(RULE),new_task_continuations=0,labels_read=False))

def load(horizon,controllers):
    d=dict(np.load(SOURCE/'dataset.npz'));x=dict(np.load(SOURCE/'frozen_model_entities.npz'))
    if horizon==80:
        inputs=[np.load(OUT/f'inputs_H80_{name}.npz') for name in NAMES]
        for key in ('context','agent_response','valid'):d[key]=np.stack([v[key] for v in inputs])
    assert d['valid'].all(),'Do not silently exclude invalid contexts'
    fit=np.flatnonzero(d['split']=='train');val=np.flatnonzero(d['split']=='validation')
    ec=d['eta'][fit].mean(0);es=np.maximum(d['eta'][fit].std(0),.1)
    cc=d['context'][controllers][:,fit].mean((0,1));cs=np.maximum(d['context'][controllers][:,fit].std((0,1)),.05)
    ac=d['agent_response'][controllers][:,fit].mean((0,1,2));ast=np.maximum(d['agent_response'][controllers][:,fit].std((0,1,2)),.01)
    norm=dict(eta_center=ec.tolist(),eta_scale=es.tolist(),context_center=cc.tolist(),context_scale=cs.tolist(),
        agent_center=ac.tolist(),agent_scale=ast.tolist(),fit_controllers=list(controllers),horizon=horizon)
    d['normalized_eta']=(d['eta']-ec)/es
    d['input_context']=np.concatenate([(d['context']-cc)/cs,((d['agent_response']-ac)/ast).reshape(4,len(d['eta']),-1)],-1).astype(np.float32)
    return d,x,fit,val,norm

def train(index):
    import jax,jax.numpy as jnp,optax
    from flax import serialization
    from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
    assert jax.default_backend()=='gpu'
    held=index//9;variant=(index%9)//3;seed=SEEDS[index%3]
    kind='eta_only' if variant==0 else 'entity_response_mean';horizon=80 if variant==2 else 20
    controllers=[c for c in range(4) if c!=held]
    dest=OUT/'models'/f'held{held}'/f'{kind}_H{horizon}'/f'seed{seed}'
    if (dest/'complete.json').exists():return
    d,x,fit,val,norm=load(horizon,controllers);si=d['state_index'];eta=d['normalized_eta'];ctx=d['input_context'];m=old.model(kind)
    p=m.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,88)))
    optimizer=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(.0008,weight_decay=.0001));o=optimizer.init(p)
    @jax.jit
    def step(p,o,xx,e,c,s,f):
        def loss(pp):
            z=m.apply(pp,xx,e,c);q=s/jnp.maximum(s+f,1)
            return jnp.mean(q*jax.nn.softplus(-z)+(1-q)*jax.nn.softplus(z))
        v,g=jax.value_and_grad(loss)(p);u,o=optimizer.update(g,o,p);return optax.apply_updates(p,u),o,v
    pred=jax.jit(lambda p,x,e,c:m.apply(p,x,e,c))
    def evaluate(p,cs,ii):
        z=np.array([np.asarray(pred(p,gather(x,si[ii]),jnp.asarray(eta[ii],jnp.float32),jnp.asarray(ctx[c,ii],jnp.float32))) for c in cs])
        s=d['success'][cs][:,ii];f=d['failure'][cs][:,ii];n=s+f
        nll=(s*np.logaddexp(0,-z)+f*np.logaddexp(0,z)).sum(1)/n.sum(1)
        return dict(NLL=float(nll.mean()),NLL_by_controller=nll.tolist(),MAE=float(abs(expit(z)-s/n).mean())),z
    rng=np.random.default_rng(seed);history=[];best=(np.inf,None,0)
    for it in range(1,1501):
        draw=rng.choice(fit,32);ii=np.tile(draw,3);ci=np.repeat(controllers,32)
        p,o,loss=step(p,o,gather(x,si[ii]),jnp.asarray(eta[ii],jnp.float32),jnp.asarray(ctx[ci,ii],jnp.float32),
            jnp.asarray(d['success'][ci,ii]),jnp.asarray(d['failure'][ci,ii]))
        if it not in old.STEPS:continue
        vm,_=evaluate(p,controllers,val);tm,_=evaluate(p,controllers,fit);history.append(dict(step=it,train=tm,validation=vm))
        if vm['NLL']<best[0]-1e-5:best=(vm['NLL'],serialization.to_bytes(p),it)
    dest.mkdir(parents=True,exist_ok=True);(dest/'best.msgpack').write_bytes(best[1]);write(dest/'normalization.json',norm)
    write(dest/'history.json',history)
    # Held-source-controller labels are not read by the optimization/selection loop.
    p=serialization.from_bytes(p,best[1]);hm,z=evaluate(p,[held],val)
    states=sorted(set(si[val]));shift={s:states[(j+1)%len(states)] for j,s in enumerate(states)}
    lookup={(int(si[i]),int(d['eta_index'][i])):int(i) for i in val}
    shifted=np.array([lookup[(shift[int(si[i])],int(d['eta_index'][i]))] for i in val])
    controls={'logits':z}
    for other in controllers:
        controls[f'wrong_controller{other}']=np.asarray(pred(p,gather(x,si[val]),jnp.asarray(eta[val],jnp.float32),jnp.asarray(ctx[other,val],jnp.float32)))[None]
    controls['wrong_state_context']=np.asarray(pred(p,gather(x,si[val]),jnp.asarray(eta[val],jnp.float32),jnp.asarray(ctx[held,shifted],jnp.float32)))[None]
    controls['state_shuffle']=np.asarray(pred(p,gather(x,si[shifted]),jnp.asarray(eta[val],jnp.float32),jnp.asarray(ctx[held,val],jnp.float32)))[None]
    controls['joint_state_context_shuffle']=np.asarray(pred(p,gather(x,si[shifted]),jnp.asarray(eta[val],jnp.float32),jnp.asarray(ctx[held,shifted],jnp.float32)))[None]
    np.savez_compressed(dest/'held_predictions.npz',indices=val,**controls)
    write(dest/'complete.json',dict(held=NAMES[held],held_index=held,kind=kind,horizon=horizon,seed=seed,
        best_step=best[2],best_source_VAL_NLL=best[0],held_controller=hm,rule_sha256=sha(RULE),code_sha256=sha(__file__),
        dataset_sha256=sha(SOURCE/'dataset.npz'),target_confirmation_labels_used=False,
        normalization_controllers=controllers,parameter_count=int(sum(v.size for v in jax.tree.leaves(p)))))
    print(dict(held=NAMES[held],kind=kind,horizon=horizon,seed=seed,step=best[2],**hm),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('features','train'));p.add_argument('--index',type=int,required=True);a=p.parse_args()
    globals()[a.action](a.index)
