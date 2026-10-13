"""Source-affine-support intervention; zero new labels or optimization.

Project physical token inputs onto the source TRAIN affine span, independently
of outcomes. This leaves source inputs unchanged and removes only directions
never varied during fitting. An improvement diagnoses unsupported extrapolation;
it is not, by itself, a repaired cross-scene critic.
"""
import argparse
from pathlib import Path
import json
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from diagnostics.controller_q_diversity_v1 import train

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'controller_q_diversity_v1'

def masks(d):
    a=d['entity_agent_mask'].astype(bool);o=d['entity_obstacle_mask'].astype(bool)
    return dict(agents=a,pairs=a[:,:,None]&a[:,None,:]&~np.eye(a.shape[1],dtype=bool)[None],
                obstacles=a[:,:,None]&o[:,None,:],globals=np.ones(len(a),bool))

def projector(v):
    mean=v.mean(0);_,s,vt=np.linalg.svd(v-mean,full_matrices=False)
    rank=int(np.sum(s>max(float(s.max(initial=0))*1e-6,1e-7)))
    p=vt[:rank].T@vt[:rank]
    return mean,p,rank

def run(fold):
    variant='factorial_'+fold
    fitroot=OLD/'checks/factorial_breadth8'
    source=np.load(fitroot/'data'/variant/'dataset.npz')
    final=np.load(OLD/'data/final/dataset.npz')
    datasets={'development':source,'opened_confirmation':final}
    ti=source['split']=='train';mm=masks(source);projections={};audit={}
    for k,m in mm.items():
        v=source['entity_'+k][ti][m[ti]].astype(float)
        mean,p,rank=projector(v);projections[k]=(mean,p)
        audit[k]=dict(dimension=len(mean),source_rank=rank,
                      source_max_residual=float(np.max(np.abs((v-mean)@(np.eye(len(mean))-p)))))
    output={};rows=[]
    for name,d in datasets.items():
        ds_mask=masks(d);variants={a:[] for a in ('baseline','physical_affine','controller_unclipped','both')}
        for seed in train.SEEDS:
            folder=fitroot/'fits'/variant/'state_contact_bilinear'/f'seed{seed}'
            norm=json.loads((folder/'normalization.json').read_text())
            x,c,e=train.inputs(d,norm)
            xp={k:np.asarray(v).copy() for k,v in x.items()}
            for k,(mean,p) in projections.items():
                xp[k]=((xp[k]-mean)@p+mean).astype(np.float32)
            xp={k:jnp.asarray(v) for k,v in xp.items()}
            # Also remove the source-constant count direction before the scene
            # layer, by a mathematically exact bias/kernel reparameterization.
            net=train.Critic('state_contact_bilinear')
            par=net.init(jax.random.PRNGKey(seed),{k:v[:1] for k,v in x.items()},c[:1],e[:1])
            par=serialization.from_bytes(par,(folder/'checkpoint.msgpack').read_bytes())
            projected=jax.tree_util.tree_map(lambda a:a.copy(),par)
            layer=projected['params']['physical_encoder']['scene']
            counts=np.stack([np.log1p(source['entity_agent_mask'][ti].sum(-1)),
                             np.log1p(source['entity_obstacle_mask'][ti].sum(-1))],-1)
            assert np.max(np.ptp(counts,axis=0))==0
            layer['bias']=layer['bias']+jnp.asarray(counts[0])@layer['kernel'][-2:]
            layer['kernel']=jnp.asarray(layer['kernel']).at[-2:].set(0)
            cn=dict(norm);cn['clip']=1e9
            _,unc,e=train.inputs(d,cn)
            # This control removes pre-encoder clipping only; the frozen
            # bilinear head's +/-3 pooling clamp is retained and audited.
            @jax.jit
            def pred(pp,xx,cc,ee):return jax.nn.sigmoid(net.apply(pp,xx,cc,ee))
            for arm,xx,cc,pp in [('baseline',x,c,par),('physical_affine',xp,c,projected),
                                 ('controller_unclipped',x,unc,par),('both',xp,unc,projected)]:
                yy=[];ii=np.repeat(np.arange(len(c)),16);jj=np.tile(np.arange(16),len(c))
                for st in range(0,len(ii),256):
                    sel=ii[st:st+256];eta=jj[st:st+256]
                    yy.extend(np.asarray(pred(pp,{k:v[sel] for k,v in xx.items()},cc[sel],e[eta])))
                variants[arm].append(np.asarray(yy).reshape(len(c),16))
            if seed==17:
                from diagnostics.orthoflow3_unified_representation_v1.representation import pool
                for scene in sorted(set(d['scene'])):
                    ids=d['scene']==scene;real=d['entity_agent_mask'][ids].astype(bool)
                    cc=np.asarray(unc)[ids][real];pooled=np.asarray(pool(unc[ids],x['agent_mask'][ids],1))
                    residual={}
                    for k,(mean,p) in projections.items():
                        raw=d['entity_'+k][ids][ds_mask[k][ids]]
                        r=(raw-mean)@(np.eye(len(mean))-p)
                        residual[k]=dict(rms=float(np.sqrt(np.mean(r*r))),max=float(np.max(abs(r))))
                    audit[name+'_'+str(scene)]=dict(controller_fraction_preclip=float(np.mean(abs(cc)>10)),
                        controller_fraction_headclip=float(np.mean(abs(pooled)>3)),physical_residual=residual)
        predictions={k:np.mean(v,axis=0) for k,v in variants.items()}
        output.update({name+'_'+k:v for k,v in predictions.items()})
        if name=='development':
            for arm in ('physical_affine','both'):
                audit[arm+'_source_probability_maxchange']=float(np.max(abs(predictions[arm][ti]-predictions['baseline'][ti])))
        for scene in sorted(set(d['scene'])):
            ids=np.flatnonzero((d['scene']==scene)&((d['original_split']=='validation') if name=='development' else True))
            known=d['success'][ids]+d['failure'][ids];y=d['success'][ids]/np.maximum(known,1)
            uids=d['state_uid'][ids];uids_unique=np.unique(uids)
            errors={k:np.asarray([np.sum(((q[ids][uids==sid]-y[uids==sid])**2)*(known[uids==sid]>0))/max(1,np.sum(known[uids==sid]>0)) for sid in uids_unique]) for k,q in predictions.items()}
            rng=np.random.default_rng(2026101101);boot=rng.integers(len(uids_unique),size=(20000,len(uids_unique)))
            for arm,err in errors.items():
                diff=errors['baseline']-err
                rows.append(dict(fold=fold,panel=name,scene=str(scene),arm=arm,N=len(uids_unique),
                                 mse=float(err.mean()),improvement=float(diff.mean()),
                                 CI95=np.quantile(diff[boot].mean(1),[.025,.975]).tolist()))
    np.savez_compressed(ROOT/f'support_{fold}_predictions.npz',**output)
    (ROOT/f'support_{fold}.json').write_text(json.dumps(dict(audit=audit,rows=rows,new_rollouts=0,
        limitation='Diagnostic only: target-specific information removed. No claim of deployable repair. Both final pools now DEVELOPMENT.'),indent=2)+'\n')
    print(json.dumps({'fold':fold,'rows':[r for r in rows if r['panel']=='opened_confirmation']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('fold',choices=['toy','ring']);run(p.parse_args().fold)
