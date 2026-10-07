"""Matched pure-NLL family crossfit for Q4 vs Q16 information and weight.

Outer source families are held out from normalization, training, early stopping
and all model choices. The seven old source VAL families are not used here.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import serialization
from diagnostics.orthoflow3_controller_information_probe_v1.probe import Critic
from diagnostics.orthoflow3_cross_scene_zero_shot_v1.train import gather
from diagnostics.orthoflow3_controller_training_repair_v1.train import controller_balanced_nll
from .pipeline import OUT,SRC,read,write,sha

ARMS=('original_Q4','Q16_rate_weight4','Q4_weight16')
SEEDS=(17,23,41)
FOLDS=(0,1,2)
CONFIG={'steps':1500,'batch_pairs':32,'learning_rate':0.0008,'weight_decay':0.0001,
        'eval_cadence':100,'inner_validation_families':6,'source_folds':3,
        'checkpoint_criterion':'common observed-count NLL on inner validation families, including Q16 contrast evidence',
        'pair_draw':'identical numpy RNG sequence for matched seed and fold across all arms',
        'supervision_difference_only':'original_Q4 vs Q16 rate with effective count4 vs original Q4 with count16'}

def fold_plan():
    assert read(OUT/'working_state.json')['phase']=='postflight_complete'
    d=dict(np.load(SRC/'variants/large_20/dataset.npz'))
    states=read(SRC/'states.json')
    tr=sorted(int(i) for i in set(d['state_index'][d['split']=='train']))
    assert len(tr)==46
    assert len({states[i]['source_group'] for i in tr})==46
    def order(salt,indices):return sorted(indices,key=lambda i:hashlib.sha256((salt+'\0'+states[i]['source_group']).encode()).hexdigest())
    ranked=order('q16_source_family_crossfit_v1',tr)
    plan=[]
    for fold in FOLDS:
        outer=sorted(ranked[fold::3])
        rest=order('q16_inner_validation_v1',sorted(set(tr)-set(outer)))
        inner=sorted(rest[:6]);fit=sorted(rest[6:])
        assert not(set(outer)&set(inner) or set(outer)&set(fit) or set(inner)&set(fit))
        plan.append({'fold':fold,'outer_state_indices':outer,'inner_state_indices':inner,
                     'fit_state_indices':fit,'outer_source_groups':[states[i]['source_group'] for i in outer],
                     'inner_source_groups':[states[i]['source_group'] for i in inner],
                     'fit_source_groups':[states[i]['source_group'] for i in fit]})
    assert sorted(i for r in plan for i in r['outer_state_indices'])==tr
    assert all(not(set(r['outer_source_groups'])&set(r['inner_source_groups'])) for r in plan)
    write(OUT/'crossfit_splits.json',{'protocol':CONFIG,'folds':plan,
        'no_original_source_VAL_or_target_family_used':True,
        'split_keys':'SHA256 only of frozen source_group identifiers, no rollout outcomes'})
    return plan

def _indices(d,state_indices):
    return np.flatnonzero((d['split']=='train')&np.isin(d['state_index'],state_indices))

def load(fold,arm):
    assert arm in ARMS
    src=SRC/'variants/large_20'
    d=dict(np.load(src/'dataset.npz'));x=dict(np.load(src/'entities.npz'))
    labels=dict(np.load(OUT/f'labels_{arm}.npz'))
    d.update(x=x,success=labels['success'],failure=labels['failure'])
    r=read(OUT/'crossfit_splits.json')['folds'][fold]
    fit=_indices(d,r['fit_state_indices']);inner=_indices(d,r['inner_state_indices']);outer=_indices(d,r['outer_state_indices'])
    assert len(inner)==6*16 and len(outer)==len(r['outer_state_indices'])*16
    assert len(fit)==len(r['fit_state_indices'])*16
    eta=d['eta'][fit];context=d['context'][:,fit][d['valid'][:,fit]]
    norm={'eta_center':eta.mean(0),'eta_scale':np.maximum(eta.std(0),.1),
          'context_center':context.mean(0),'context_scale':np.maximum(context.std(0),.05)}
    d['eta']=((d['eta']-norm['eta_center'])/norm['eta_scale']).astype(np.float32)
    d['context']=np.nan_to_num((d['context']-norm['context_center'])/norm['context_scale']).astype(np.float32)
    for key in ('success','failure'):d[key]=np.where(d['valid'],d[key],0.).astype(np.float32)
    d['norm']={k:v.tolist() for k,v in norm.items()}
    full=dict(np.load(OUT/'full_Q16_evidence.npz'))
    return d,fit,inner,outer,full

def train(fold,arm,seed):
    assert fold in FOLDS and arm in ARMS and seed in SEEDS
    target=OUT/'crossfit_models'/f'fold{fold}'/arm/f'seed{seed}'
    if (target/'complete.json').exists():
        print({'fold':fold,'arm':arm,'seed':seed,'already_complete':True},flush=True);return
    d,fit,inner,outer,full=load(fold,arm)
    x,si,eta,cx=d['x'],d['state_index'],d['eta'],d['context']
    model=Critic(True,True,False)
    template=model.init(jax.random.PRNGKey(seed),gather(x,si[:1]),jnp.zeros((1,3)),jnp.zeros((1,24)),jnp.zeros((1,3)))
    pars=template
    optimizer=optax.chain(optax.clip_by_global_norm(5.),optax.adamw(CONFIG['learning_rate'],weight_decay=CONFIG['weight_decay']))
    state=optimizer.init(pars)
    @jax.jit
    def step(pars,state,xx,ee,cc,ss,ff):
        def objective(p):
            z=model.apply(p,xx,ee,cc,jnp.zeros((len(ee),3))).reshape(3,-1)
            return controller_balanced_nll(z,ss,ff)
        loss,grads=jax.value_and_grad(objective)(pars)
        update,state=optimizer.update(grads,state,pars)
        return optax.apply_updates(pars,update),state,loss
    predictor=jax.jit(lambda p,xx,ee,cc:model.apply(p,xx,ee,cc,jnp.zeros((len(ee),3))))
    def infer(params,indices):
        logits=np.zeros((3,len(indices)),np.float32)
        for c in range(3):
            for start in range(0,len(indices),128):
                ii=indices[start:start+128]
                logits[c,start:start+len(ii)]=np.asarray(predictor(params,gather(x,si[ii]),jnp.asarray(eta[ii]),jnp.asarray(cx[c,ii])))
        return logits
    rng=np.random.default_rng(seed)
    best=(float('inf'),None,0);history=[]
    for iteration in range(1,CONFIG['steps']+1):
        draw=rng.choice(fit,CONFIG['batch_pairs'])
        ii=np.tile(draw,3);cc=np.repeat(np.arange(3),len(draw))
        pars,state,loss=step(pars,state,gather(x,si[ii]),jnp.asarray(eta[ii]),
             jnp.asarray(cx[cc,ii]),jnp.asarray(d['success'][:,draw]),jnp.asarray(d['failure'][:,draw]))
        if iteration%CONFIG['eval_cadence']:continue
        z=infer(pars,inner)
        value=float(controller_balanced_nll(jnp.asarray(z),jnp.asarray(full['success'][:,inner]),
                                             jnp.asarray(full['failure'][:,inner])))
        history.append({'step':iteration,'TRAIN_minibatch_objective':float(loss),'inner_validation_common_NLL':value})
        if value<best[0]-1e-5:best=(value,serialization.to_bytes(pars),iteration)
    assert best[1] is not None
    pars=serialization.from_bytes(template,best[1])
    outer_predictions=infer(pars,outer)
    target.mkdir(parents=True,exist_ok=True)
    (target/'checkpoint.msgpack').write_bytes(best[1])
    write(target/'normalization.json',d['norm'])
    write(target/'history.json',history)
    np.savez_compressed(target/'outer_predictions.npz',indices=outer,state_indices=si[outer],logits=outer_predictions)
    write(target/'complete.json',{'fold':fold,'arm':arm,'seed':seed,'outer_state_indices':read(OUT/'crossfit_splits.json')['folds'][fold]['outer_state_indices'],
        'inner_validation_best_NLL':best[0],'best_step':best[2],
        'checkpoint_sha256':sha(target/'checkpoint.msgpack'),
        'source_dataset_sha256':sha(SRC/'variants/large_20/dataset.npz'),
        'training_label_sha256':sha(OUT/f'labels_{arm}.npz'),
        'outer_labels_used_in_training_or_checkpoint':False,
        'old_source_VAL_or_target_scene_labels_used':False,'architecture':'frozen Critic(True,True,False)',
        'optimizer_and_pair_draw_identical_across_arms':True})
    print({'fold':fold,'arm':arm,'seed':seed,'inner_best_NLL':best[0],
           'best_step':best[2],'outer_families':len(set(si[outer]))},flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('split','train'))
    p.add_argument('--fold',type=int,choices=FOLDS,default=0)
    p.add_argument('--arm',choices=ARMS,default=ARMS[0])
    p.add_argument('--seed',type=int,choices=SEEDS,default=17)
    p.add_argument('--index',type=int,default=None,
                   help='Array index in [0,26], ordered fold / arm / seed')
    a=p.parse_args()
    if a.action=='split':fold_plan()
    elif a.index is not None:
        assert 0<=a.index<len(FOLDS)*len(ARMS)*len(SEEDS)
        fold,rem=divmod(a.index,len(ARMS)*len(SEEDS))
        arm,seed_index=divmod(rem,len(SEEDS))
        train(FOLDS[fold],ARMS[arm],SEEDS[seed_index])
    else:train(a.fold,a.arm,a.seed)
if __name__=='__main__':main()
