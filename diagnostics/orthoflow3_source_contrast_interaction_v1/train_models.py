"""Matched source-only pure-NLL models; independent families opened after freeze."""
from __future__ import annotations
import argparse,hashlib,json
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import serialization
from diagnostics.orthoflow3_controller_information_probe_v1.probe import Critic
from diagnostics.orthoflow3_controller_training_repair_v1.train import Additive,controller_balanced_nll
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
from .pipeline import OUT,read,write,sha

KINDS=('eta_only','additive','context_only','full','full_old_Q4','full_Q4_weight16')
SEEDS=(17,23,41)
CONFIG={'steps':1500,'batch_pairs':32,'eval_every':100,'learning_rate':.0008,'weight_decay':.0001,
        'checkpoint':'Q16 observed-count NLL on six source-TRAIN inner validation families',
        'independent_validation':'sixteen source families unopened until checkpoint frozen',
        'pure_NLL':True,'generator_modified':False}

def split():
    assert read(OUT/'working_state.json')['phase']=='postflight_complete'
    d=np.load(OUT/'dataset.npz');states=read(OUT/'states.json')
    train=sorted(int(i) for i in set(d['state_index'][d['split']=='train']))
    validation=sorted(int(i) for i in set(d['state_index'][d['split']=='validation']))
    assert len(train)>=40 and len(validation)>=12
    order=sorted(train,key=lambda i:hashlib.sha256(('source_contrast_inner_v1\0'+states[i]['source_group']).encode()).hexdigest())
    inner=sorted(order[:6]);fit=sorted(order[6:])
    assert not(set(inner)&set(fit) or set(train)&set(validation))
    write(OUT/'train_split.json',{'fit_state_indices':fit,'inner_state_indices':inner,
        'independent_validation_state_indices':validation,
        'fit_source_groups':[states[i]['source_group'] for i in fit],
        'inner_source_groups':[states[i]['source_group'] for i in inner],
        'independent_validation_source_groups':[states[i]['source_group'] for i in validation],
        'split_rule':'SHA256 source_group, no rollout outcomes; independent validation fixed before new outcomes',
        'architecture_and_hyperparameters':CONFIG})
    print({'fit_states':len(fit),'inner_states':len(inner),'independent_validation_states':len(validation)})

def model_for(kind):
    if kind=='eta_only':return Critic(False,False,False)
    if kind=='additive':return Additive()
    if kind=='context_only':return Critic(False,True,False)
    return Critic(True,True,False)

def load(kind):
    raw=dict(np.load(OUT/'dataset.npz'));raw['x']=dict(np.load(OUT/'entities.npz'))
    plan=read(OUT/'train_split.json')
    si=raw['state_index']
    fi=np.flatnonzero(np.isin(si,plan['fit_state_indices'])&(raw['split']=='train'))
    iv=np.flatnonzero(np.isin(si,plan['inner_state_indices'])&(raw['split']=='train'))
    ov=np.flatnonzero(np.isin(si,plan['independent_validation_state_indices'])&(raw['split']=='validation'))
    assert len(fi)==2*len(plan['fit_state_indices']) and len(iv)==2*len(plan['inner_state_indices'])
    assert len(ov)==2*len(plan['independent_validation_state_indices'])
    center=raw['eta'][fi].mean(0);scale=np.maximum(raw['eta'][fi].std(0),.1)
    ctx=raw['context'][:,fi,:][raw['valid'][:,fi]]
    ccenter=ctx.mean(0);cscale=np.maximum(ctx.std(0),.05)
    # Additive is truly additive: the eta-dependent final 14 channels are zero.
    if kind=='additive':
        raw['context']=raw['context'].copy();raw['context'][:,:,10:]=ccenter[None,None,10:]
        assert np.max(abs(raw['context'][:,::2,:10]-raw['context'][:,1::2,:10]))<1e-5
    raw['eta']=((raw['eta']-center)/scale).astype(np.float32)
    raw['context']=((raw['context']-ccenter)/cscale).astype(np.float32)
    assert np.isfinite(raw['context']).all() and np.isfinite(raw['eta']).all()
    raw['normalization']={'eta_center':center.tolist(),'eta_scale':scale.tolist(),
        'context_center':ccenter.tolist(),'context_scale':cscale.tolist(),
        'fit_source_only':True,'additive_candidate_response_channels_zeroed':kind=='additive'}
    return raw,fi,iv,ov

def train(kind,seed):
    assert kind in KINDS and seed in SEEDS
    dest=OUT/'models'/kind/f'seed{seed}'
    if (dest/'complete.json').exists():print({'already_complete':str(dest)});return
    d,fit,inner,outer=load(kind)
    x,si,eta,cx=d['x'],d['state_index'],d['eta'],d['context']
    model=model_for(kind)
    template=model.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,24)),jnp.zeros((1,3)))
    params=template
    optimizer=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay']))
    opt_state=optimizer.init(params)
    @jax.jit
    def step(params,opt_state,xx,ee,cc,ss,ff):
        def objective(p):
            logits=model.apply(p,xx,ee,cc,jnp.zeros((len(ee),3))).reshape(2,-1)
            return controller_balanced_nll(logits,ss,ff)
        loss,grads=jax.value_and_grad(objective)(params)
        update,opt_state=optimizer.update(grads,opt_state,params)
        return optax.apply_updates(params,update),opt_state,loss
    pred=jax.jit(lambda p,xx,ee,cc:model.apply(p,xx,ee,cc,jnp.zeros((len(ee),3))))
    def infer(p,indices,context_swap=False,state_shuffle=False):
        logits=np.zeros((2,len(indices)),np.float32)
        for controller in range(2):
            context_controller=1-controller if context_swap else controller
            for start in range(0,len(indices),128):
                ii=indices[start:start+128]
                state_indices=si[ii]
                if state_shuffle:
                    order=read(OUT/'train_split.json')['independent_validation_state_indices']
                    state_indices=np.asarray([order[(order.index(int(s))+1)%len(order)] for s in state_indices])
                logits[controller,start:start+len(ii)]=np.asarray(pred(p,gather(x,state_indices),
                    jnp.asarray(eta[ii]),jnp.asarray(cx[context_controller,ii])))
        return logits
    rng=np.random.default_rng(seed)
    best=(float('inf'),None,0);history=[]
    for iteration in range(1,CONFIG['steps']+1):
        draw=rng.choice(fit,CONFIG['batch_pairs'])
        ii=np.tile(draw,2);controller=np.repeat(np.arange(2),len(draw))
        ss=(d['original_Q4_success'] if kind in ('full_old_Q4','full_Q4_weight16') else d['success'])[:,draw]
        ff=(d['original_Q4_failure'] if kind in ('full_old_Q4','full_Q4_weight16') else d['failure'])[:,draw]
        if kind=='full_Q4_weight16':ss=ss*4;ff=ff*4
        params,opt_state,loss=step(params,opt_state,gather(x,si[ii]),jnp.asarray(eta[ii]),
            jnp.asarray(cx[controller,ii]),jnp.asarray(ss),jnp.asarray(ff))
        if iteration%CONFIG['eval_every']:continue
        inner_logits=infer(params,inner)
        value=float(controller_balanced_nll(jnp.asarray(inner_logits),jnp.asarray(d['success'][:,inner]),
            jnp.asarray(d['failure'][:,inner])))
        history.append({'step':iteration,'train_minibatch_objective':float(loss),'inner_Q16_NLL':value})
        if value<best[0]-1e-5:best=(value,serialization.to_bytes(params),iteration)
    assert best[1] is not None
    params=serialization.from_bytes(template,best[1])
    correct=infer(params,outer)
    wrong=infer(params,outer,context_swap=True)
    shuffled=infer(params,outer,state_shuffle=True)
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'checkpoint.msgpack').write_bytes(best[1])
    write(dest/'normalization.json',d['normalization'])
    write(dest/'history.json',history)
    np.savez_compressed(dest/'frozen_validation_predictions.npz',pair_indices=outer,
        state_indices=si[outer],eta_indices=d['eta_index'][outer],
        correct_logits=correct,wrong_controller_logits=wrong,state_shuffled_logits=shuffled)
    write(dest/'complete.json',{'kind':kind,'seed':seed,'best_step':best[2],
        'inner_source_TRAIN_NLL':best[0],
        'checkpoint_sha256':sha(dest/'checkpoint.msgpack'),
        'dataset_sha256':sha(OUT/'dataset.npz'),
        'independent_validation_labels_used_for_checkpoint':False,
        'held_controller_or_LOSO_target_labels_used':False,
        'generator_modified':False})
    print({'kind':kind,'seed':seed,'best_step':best[2],'inner_NLL':best[0],
           'independent_source_states':len(set(si[outer]))},flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('split','train'))
    p.add_argument('--kind',choices=KINDS,default='full')
    p.add_argument('--seed',type=int,choices=SEEDS,default=17)
    p.add_argument('--index',type=int,default=None)
    a=p.parse_args()
    if a.action=='split':split();return
    if a.index is not None:
        assert 0<=a.index<len(KINDS)*len(SEEDS)
        ki,si=divmod(a.index,len(SEEDS));train(KINDS[ki],SEEDS[si])
    else:train(a.kind,a.seed)
if __name__=='__main__':main()
