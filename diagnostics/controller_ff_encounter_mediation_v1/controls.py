"""Falsify whether the encounter gain is geometry-only or uses correct C."""
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from diagnostics.controller_q_diversity_v1.train import Critic
from diagnostics.controller_operator_repair_v1.review import gain_bounds,corr,center
from .study import HERE,read,write,freeze
from .learn import FOLDS

FOLD_KEYS=('held_toy','held_db','held_four','held_ring')

def prepare():
    freeze(HERE/'controls_protocol.json',dict(
        purpose='Do not attribute Ring improvement to controller mechanism before ruling out added geometry alone.',
        geometry_control='Same83-channel network,labels,initialization,optimizer and source-only checkpoint rule. All encounter action mean/variance channels zero. Keep map/agent probe displacement and physical validity, not controller-dependent solver validity.',
        frozen_model_control='Swap only encounter response(mean2,var2,solved fraction1) across all4 other controllers at the SAME physical state and eta. Keep correct physical state, original controller field, current action and probe geometry. Average donor losses, not loss of mean prediction.',
        unchanged='All fitted main checkpoints and original outcomes remain immutable.',
        seeds=[17,23,41],bootstrap_unit='physical state',bootstrap_replicates=20000,
        status='DEVELOPMENT follow-up; not final confirmation and not an exclusion of all alternative causes.'))


def infer(index):
    fold=FOLD_KEYS[index];sc=FOLDS[fold];d=np.load(HERE/'dataset.npz')
    ix=np.flatnonzero((d['scene']==sc)&(d['split']=='validation'));ix=np.asarray(sorted(ix,key=lambda i:(d['state_uid'][i],d['controller'][i])))
    assert len(ix)==40;kk=np.array([0,5,10,15]);mask=d['entity_agent_mask'][ix]
    x={k[7:]:jnp.asarray(d[k][ix],jnp.float32) for k in d.files if k.startswith('entity_')}
    raw=np.concatenate((np.broadcast_to(d['feature'][ix,None],d['action_feature'][ix].shape[:-1]+(64,)),d['action_feature'][ix]),-1)
    variants=[raw]
    for shift in range(1,5):
        z=raw.copy();donor=np.roll(raw.reshape(8,5,16,4,83),shift,axis=1).reshape(raw.shape)
        z[...,69:74]=donor[...,69:74];z[...,76:81]=donor[...,76:81];variants.append(z)
    e=jnp.asarray(d['eta'][ix],jnp.float32);net=Critic('state_field_compact');allpred=[]
    for seed in (17,23,41):
        out=HERE/'fits'/fold/'current_and_encounter'/f'seed{seed}';norm=read(out/'normalization.json')
        mean=np.asarray(norm['mean']);scale=np.asarray(norm['scale'])
        init=net.init(jax.random.PRNGKey(seed),{k:v[:1] for k,v in x.items()},jnp.zeros((1,4,83)),e[:1,0])
        params=serialization.from_bytes(init,(out/'checkpoint.msgpack').read_bytes())
        @jax.jit
        def apply(cc):
            return jax.vmap(lambda j:net.apply(params,x,cc[:,j],e[:,j]),out_axes=1)(jnp.asarray(kk))
        pred=[]
        for v in variants:
            cc=jnp.asarray(np.clip((v-mean)/scale,-10,10)*mask[:,None,:,None],jnp.float32)
            logits=np.asarray(apply(cc),np.float64);pred.append(1/(1+np.exp(-np.clip(logits,-100,100))))
        stored=np.load(out/'predictions.npz')['probability'][ix][:,kk]
        np.testing.assert_allclose(pred[0],stored,rtol=2e-5,atol=2e-6)
        allpred.append(pred)
    probability=np.mean(allpred,0).reshape(5,8,5,4)
    np.savez_compressed(HERE/f'contact_swap_{fold}.npz',probability=probability,row_indices=ix,
        success=d['success'][ix][:,kk].reshape(8,5,4),failure=d['failure'][ix][:,kk].reshape(8,5,4),unknown=d['unknown'][ix][:,kk].reshape(8,5,4))
    print(fold,'all4 wrong-controller response donors evaluated',flush=True)


def review():
    d=np.load(HERE/'dataset.npz');rows=[];saved={}
    for fold in FOLD_KEYS:
        sc=FOLDS[fold];z=np.load(HERE/f'contact_swap_{fold}.npz');ix=z['row_indices'];kk=[0,5,10,15]
        s,f,u=(z[k] for k in ('success','failure','unknown'));lo=s/4;hi=(s+u)/4;y=s/np.maximum(s+f,1);valid=s+f>0
        qs=z['probability'];q=qs[0]
        geo=np.mean([np.load(HERE/'fits'/fold/'encounter_geometry_only'/f'seed{seed}'/'predictions.npz')['probability'][ix][:,kk] for seed in (17,23,41)],0).reshape(8,5,4)
        boot=np.random.default_rng(2026101206).integers(8,size=(20000,8))
        for name,baselines in [('geometry_only',[geo]),('wrong_contact_controller',qs[1:])]:
            diff=np.mean([(((b-y)**2-(q-y)**2)*valid).sum((1,2))/np.maximum(valid.sum((1,2)),1) for b in baselines],0)
            bounds=[gain_bounds(b,q,lo,hi) for b in baselines];lower=np.mean([r[0] for r in bounds],0);upper=np.mean([r[1] for r in bounds],0)
            bb=diff[boot].mean(1);bl=lower[boot].mean(1);bh=upper[boot].mean(1);saved[fold+'_'+name]=bb
            rows.append(dict(fold=fold,scene=sc,comparison='correct_encounter_vs_'+name,N=8,Q_MSE_improvement=float(diff.mean()),
                CI95=np.quantile(bb,[.025,.975]).tolist(),censor_robust_CI95=[float(np.quantile(bl,.025)),float(np.quantile(bh,.975))],
                correct_MSE=float(np.mean(((q-y)**2)[valid])),baseline_MSE=float(np.mean([np.mean(((b-y)**2)[valid]) for b in baselines]))))
        aa,bb=np.triu_indices(5,1);good=valid[:,aa]&valid[:,bb]
        rows.append(dict(fold=fold,scene=sc,comparison='geometry_only_diagnostics',N=8,Q_MSE=float(np.mean(((geo-y)**2)[valid])),
            controller_contrast_corr=corr((geo[:,aa]-geo[:,bb])[good],(y[:,aa]-y[:,bb])[good]),state_effect_corr=corr(center(geo,[0])[valid],center(y,[0])[valid])))
    np.savez_compressed(HERE/'controls_bootstrap.npz',**saved);write(HERE/'controls_results.json',dict(rows=rows,new_rollouts=0,status='DEVELOPMENT'))
    for r in rows:print(r)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','infer','review']);p.add_argument('--index',type=int);a=p.parse_args()
    if a.stage=='infer':infer(a.index)
    else:globals()[a.stage]()
