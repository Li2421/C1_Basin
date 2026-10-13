"""Zero-rollout development controls; never changes frozen confirmation models."""
import argparse
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from . import train
from .experiment import HERE,read,write,sha
from .evaluate import masked_mean,interval

OUT=HERE/'checks/factorial_breadth8'
KINDS=('eta_only','state_only','state_contact_bilinear')
SCENES={'factorial_toy':'toy_give_way','factorial_ring':'ring_exchange'}


def prepare():
    parent=HERE/'data/source_diverse64/dataset.npz';d=np.load(parent);keep=d['controller']!='flow'
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'training_plan.json').write_bytes((HERE/'training_plan.json').read_bytes())
    for variant,source in SCENES.items():
        data={k:d[k][keep] if d[k].ndim and len(d[k])==len(keep) else d[k] for k in d.files}
        data['original_split']=data['split'].copy();data['split']=data['split'].astype('<U32')
        data['split'][data['scene']!=source]='development_target'
        folder=OUT/'data'/variant;folder.mkdir(exist_ok=True,parents=True)
        path=folder/'dataset.npz';assert not path.exists();np.savez_compressed(path,**data)
        write(folder/'manifest.json',dict(dataset_sha256=sha(path),parent_sha256=sha(parent),
            final_labels_used=False,source_scene=source,labels='8analytic x64states; same label budget as4analytic x128states; no Flow',
            held_scene_usage='Development evaluation only; source-only gradients, normalization and checkpoint choice'))
    write(OUT/'protocol.json',dict(models=list(KINDS),seeds=list(train.SEEDS),new_rollouts=0,
        main_confirmation_models_unchanged=True,geometry_nullspace_cutoff=1e-8,
        question='Does controller breadth reduce the scene/pipeline shift gap at equal analytic label budget?'))


def fit(index):
    variant=list(SCENES)[index//9];kind=KINDS[(index%9)//3];seed=train.SEEDS[index%3]
    train.HERE=OUT
    train.fit(kind,seed,variant=variant)


def source_projection_control():
    rows=[];audits=[]
    for variant,source in SCENES.items():
        d=np.load(HERE/'data'/variant/'dataset.npz');training=d['split']=='train'
        mask=d['entity_agent_mask'][:,:,None]*d['entity_obstacle_mask'][:,None,:]
        x=d['entity_obstacles'];values=x[training][mask[training].astype(bool)].astype(float)
        mean=values.mean(0);_,singular,vectors=np.linalg.svd(values-mean,full_matrices=False)
        rank=int(np.sum(singular>singular.max()*1e-8));projector=vectors[:rank].T@vectors[:rank]
        projected=(x-mean)@projector+mean
        train_residual=float(np.max(np.abs(projected[training][mask[training].astype(bool)]-values)))
        before=[];after=[]
        for seed in train.SEEDS:
            folder=HERE/'fits'/variant/'state_contact_bilinear'/f'seed{seed}'
            physical,controller,eta=train.inputs(d,read(folder/'normalization.json'));net=train.Critic('state_contact_bilinear')
            init=net.init(jax.random.PRNGKey(seed),{k:v[:1] for k,v in physical.items()},controller[:1],eta[:1])
            parameters=serialization.from_bytes(init,(folder/'checkpoint.msgpack').read_bytes())
            def prediction(obstacles):
                data=dict(physical);data['obstacles']=jnp.asarray(obstacles,jnp.float32)
                @jax.jit
                def batch(ii,jj):return net.apply(parameters,{k:v[ii] for k,v in data.items()},controller[ii],eta[jj])
                ii=np.repeat(np.arange(len(x)),16);jj=np.tile(np.arange(16),len(x));zs=[]
                for start in range(0,len(ii),256):zs.append(np.asarray(batch(jnp.asarray(ii[start:start+256]),jnp.asarray(jj[start:start+256]))))
                return 1/(1+np.exp(-np.clip(np.concatenate(zs).reshape(len(x),16),-80,80)))
            before.append(prediction(x));after.append(prediction(projected))
        before=np.mean(before,0);after=np.mean(after,0)
        audits.append(dict(source_scene=source,feature_dim=values.shape[1],source_affine_rank=rank,
            source_null_dimensions=values.shape[1]-rank,singular_values=singular.tolist(),
            train_feature_residual_max=train_residual,train_prediction_change_max=float(np.max(np.abs(before[training]-after[training]))),
            note='Source-only affine projection of physical obstacle channels; removes unsupported target directions. Diagnostic control, not a claim of correct target geometry encoding.'))
        for scene in set(d['scene']):
            for seen in (True,False):
                names=[f'analytic{k}' for k in (range(4) if seen else (4,5))]
                ids=np.flatnonzero((d['scene']==scene)&(d['original_split']=='validation')&np.isin(d['controller'],names));c=len(names);n=len(ids)//c
                known=d['success'][ids]+d['failure'][ids]>0;y=d['success'][ids]/np.maximum(d['success'][ids]+d['failure'][ids],1)
                old=masked_mean(((before[ids]-y)**2).reshape(n,c,16),known.reshape(n,c,16),axis=(1,2))
                new=masked_mean(((after[ids]-y)**2).reshape(n,c,16),known.reshape(n,c,16),axis=(1,2))
                bootstrap=np.random.default_rng(2026100951).integers(n,size=(20000,n))
                rows.append(dict(source_scene=source,evaluation_scene=str(scene),seen_analytic_parameters=seen,physical_states=n,
                    original_Q_MSE=float(old.mean()),projected_Q_MSE=float(new.mean()),improvement=float(np.mean(old-new)),CI=interval((old-new)[bootstrap].mean(1))))
    write(OUT/'source_geometry_projection.json',dict(audits=audits,rows=rows,new_rollouts=0,confirmation_inputs_used=False))


def review():
    rows=[]
    for variant,source in SCENES.items():
        d=np.load(OUT/'data'/variant/'dataset.npz');old=np.load(HERE/'data'/variant/'dataset.npz')
        for kind in KINDS:
            current=[];previous=[]
            for seed in train.SEEDS:
                for root,collector in [(OUT,current),(HERE,previous)]:
                    z=np.load(root/'fits'/variant/kind/f'seed{seed}'/'source_predictions.npz')['logits']
                    collector.append(1/(1+np.exp(-np.clip(z,-80,80))))
            q=np.mean(current,0);prior=np.mean(previous,0)
            lookup={(str(s),str(c)):i for i,(s,c) in enumerate(zip(old['state_uid'],old['controller']))}
            for scene in sorted(set(d['scene'])):
                for seen in (True,False):
                    names=[f'analytic{k}' for k in (range(4) if seen else (4,5))]
                    ids=np.flatnonzero((d['scene']==scene)&(d['original_split']=='validation')&np.isin(d['controller'],names));c=len(names);n=len(ids)//c
                    old_ids=np.array([lookup[str(d['state_uid'][i]),str(d['controller'][i])] for i in ids])
                    for key in ('success','failure','unknown'):np.testing.assert_array_equal(d[key][ids],old[key][old_ids])
                    known=d['success'][ids]+d['failure'][ids]>0;y=d['success'][ids]/np.maximum(d['success'][ids]+d['failure'][ids],1)
                    errors=[]
                    for pred in [prior[old_ids],q[ids]]:errors.append(masked_mean(((pred-y)**2).reshape(n,c,16),known.reshape(n,c,16),axis=(1,2)))
                    gain=errors[0]-errors[1];bootstrap=np.random.default_rng(2026100950).integers(n,size=(20000,n))
                    rows.append(dict(source_scene=source,evaluation_scene=str(scene),seen_analytic_parameters=seen,model=kind,
                        physical_states=n,Q_MSE_4controllers=float(errors[0].mean()),Q_MSE_8controllers=float(errors[1].mean()),
                        improvement=float(gain.mean()),CI=interval(gain[bootstrap].mean(1))))
    write(OUT/'breadth_comparison.json',dict(rows=rows,new_rollouts=0,final_confirmation_used=False,
        interpretation='Same source-selected architecture and optimizer;4x128 versus8x64analytic labels; same Q4 validation physical states/controllers0..5. Native scene pipelines remain unchanged.'))
    source_projection_control()
    print('Controller breadth and unsupported geometry controls complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fit','review']);p.add_argument('--index',type=int);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='fit':fit(a.index)
    else:review()
