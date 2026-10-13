"""Cache-only DEVELOPMENT separation of scene and controller shifts.

For each fold, gradients, normalization and checkpoint choice use one scene.
The second scene is evaluated only after its source-selected fits finish.
Representations were already investigated using pooled development data, so
this is mechanistic diagnosis, not an untouched LOSO confirmation claim.
"""
import argparse
import numpy as np
from .experiment import HERE,read,write,freeze,sha
from .train import KINDS,SEEDS
from .evaluate import masked_mean,center_known,corr,interval,square_gain_bounds

SCENES={'factorial_toy':'toy_give_way','factorial_ring':'ring_exchange',
        'factorial_toy_constants':'toy_give_way','factorial_ring_constants':'ring_exchange'}


def execution_constants(scene):
    from diagnostics.orthoflow3_unified_representation_v1.representation import SPEED
    cfg=read(HERE/'protocol.json')['environments'][scene]['config']['config']
    seconds=cfg['max_steps']*cfg['dt']
    return np.asarray([1.,seconds/60.,seconds/60.,cfg['dt']/.05,cfg['max_speed']/SPEED,
        0.,cfg.get('wall_collision_margin',cfg.get('collision_margin',0.))/.01,
        cfg.get('agent_collision_margin',0.)/.01,bool(cfg.get('terminate_on_deadlock',False)),
        cfg.get('progress_window_seconds',0.)/5.,cfg.get('deadlock_hold_seconds',0.)/5.,
        cfg.get('progress_epsilon',0.)/.01,cfg.get('speed_epsilon_fraction',0.),0.,1.],np.float32)


def prepare():
    original=HERE/'data/source128/dataset.npz';d=np.load(original)
    mask=d['controller']!='flow'
    for variant,source in SCENES.items():
        out={k:d[k][mask] if d[k].ndim and len(d[k])==len(mask) else d[k] for k in d.files}
        out['original_split']=out['split'].copy()
        out['split']=out['split'].astype('<U32')
        out['split'][out['scene']!=source]='development_target'
        if variant.endswith('_constants'):
            for scene in set(out['scene']):out['entity_globals'][out['scene']==scene]=execution_constants(str(scene))
        folder=HERE/'data'/variant;folder.mkdir(exist_ok=True,parents=True)
        path=folder/'dataset.npz'
        if not path.exists():np.savez_compressed(path,**out)
        freeze(folder/'manifest.json',dict(dataset_sha256=sha(path),parent_sha256=sha(original),
            source_scene=source,final_labels_used=False,added_rollouts=0,
            training_controllers='analytic0..3',checkpoint_validation_controllers='analytic4/5, source scene only',
            environment_constants_restored=variant.endswith('_constants'),
            scope='Development factorial; both scene datasets previously informed generic representation investigation. All rows retain identical cached labels. Restored global physical/task constants use deployment configuration, not label-fitted values or scene-ID.',
            controller_identity='Same analytic function and six parameters across scenes; native scene-specific FF initialization, physical configuration and safety remain unchanged. Not an assertion of identical effective execution kernels.'))
    print('Factorial cache datasets prepared; no new rollouts',flush=True)


def review():
    rows=[];comparisons=[];selection={};boots={}
    environment_pairs={}
    for variant,source in SCENES.items():
        d=np.load(HERE/'data'/variant/'dataset.npz');pred={};scores={}
        for kind in KINDS:
            values=[];score=[]
            for seed in SEEDS:
                folder=HERE/'fits'/variant/kind/f'seed{seed}'
                record=read(folder/'complete.json');assert record['dataset_sha256']==sha(HERE/'data'/variant/'dataset.npz')
                z=np.load(folder/'source_predictions.npz')['logits'];values.append(1/(1+np.exp(-np.clip(z,-80,80))))
                score.append(record['selected']['score'])
            pred[kind]=np.mean(values,0);scores[kind]=float(np.mean(score))
        environment_pairs[variant]=(d,pred)
        primary=min((k for k in KINDS if k.startswith('state_') and k!='state_only'),key=scores.get)
        selection[variant]=dict(primary=primary,source_only_scores=scores)
        for scene in sorted(set(d['scene'])):
            for seen in (True,False):
                names=[f'analytic{k}' for k in (range(4) if seen else (4,5))]
                ids=np.flatnonzero((d['scene']==scene)&(d['original_split']=='validation')&np.isin(d['controller'],names))
                c=len(names);n=len(ids)//c
                assert np.all(d['state_uid'][ids].reshape(n,c)==d['state_uid'][ids[::c]][:,None])
                assert np.all(d['controller'][ids].reshape(n,c)==np.asarray(names)[None,:])
                s=d['success'][ids].reshape(n,c,16);f=d['failure'][ids].reshape(n,c,16);u=d['unknown'][ids].reshape(n,c,16)
                known=s+f>0;y=s/np.maximum(s+f,1);lo=s/(s+f+u);hi=(s+u)/(s+f+u)
                panel=('same_scene' if scene==source else 'cross_scene')+('_seen_controller' if seen else '_unseen_controller')
                boot=np.random.default_rng(2026100947).integers(n,size=(20000,n));errors={}
                for kind,allq in pred.items():
                    q=allq[ids].reshape(n,c,16);error=masked_mean((q-y)**2,known,axis=(1,2));errors[kind]=error
                    sel=q.argmax(-1);selected=lo[np.arange(n)[:,None],np.arange(c)[None,:],sel]
                    rows.append(dict(variant=variant,source_scene=source,evaluation_scene=scene,panel=panel,model=kind,
                        source_selected_primary=kind==primary,physical_states=n,controllers=names,
                        Q_MSE=float(error.mean()),Q_MSE_CI=interval(error[boot].mean(1)),
                        state_effect_corr=corr(center_known(q,known),center_known(y,known),known),
                        selected_verified_success=float(selected.mean()),oracle_success=float(lo.max(-1).mean()),
                        numerical_unknown=int(u.sum())))
                q=pred[primary][ids].reshape(n,c,16)
                for baseline in ('eta_only','state_only'):
                    gain=errors[baseline]-errors[primary]
                    b=pred[baseline][ids].reshape(n,c,16);lower,_=square_gain_bounds(b,q,lo,hi)
                    for suffix,value in [('',gain),('_censor_worst',lower.mean((1,2)))]:
                        samples=value[boot].mean(1);label=baseline+'_minus_primary_Q_MSE'+suffix
                        comparisons.append(dict(variant=variant,source_scene=source,panel=panel,comparison=label,primary=primary,
                            physical_states=n,mean=float(value.mean()),CI=interval(samples)))
                        boots[variant+'_'+panel+'_'+label]=samples
    environment_comparisons=[]
    for restored in ('factorial_toy_constants','factorial_ring_constants'):
        original=restored.removesuffix('_constants');da,pa=environment_pairs[original];db,pb=environment_pairs[restored]
        for key in ('state_uid','controller','success','failure','unknown','split','eta'):
            np.testing.assert_array_equal(da[key],db[key])
        for kind in ('eta_only','context_only','field_only'):
            np.testing.assert_array_equal(pa[kind],pb[kind])
        for scene in sorted(set(da['scene'])):
            for seen in (True,False):
                names=[f'analytic{k}' for k in (range(4) if seen else (4,5))]
                ids=np.flatnonzero((da['scene']==scene)&(da['original_split']=='validation')&np.isin(da['controller'],names))
                c=len(names);n=len(ids)//c;known=da['success'][ids]+da['failure'][ids]>0
                truth=da['success'][ids]/np.maximum(da['success'][ids]+da['failure'][ids],1)
                boot=np.random.default_rng(2026100948).integers(n,size=(20000,n))
                for kind in KINDS:
                    change=(pa[kind][ids]-truth)**2-(pb[kind][ids]-truth)**2
                    value=masked_mean(change.reshape(n,c,16),known.reshape(n,c,16),axis=(1,2))
                    environment_comparisons.append(dict(source_scene=SCENES[restored],evaluation_scene=scene,
                        seen_controller=seen,model=kind,physical_states=n,zero_globals_minus_restored_Q_MSE=float(value.mean()),
                        CI=interval(value[boot].mean(1))))
    write(HERE/'factorial_development_results.json',dict(rows=rows,comparisons=comparisons,selection=selection,
        environment_constants_comparisons=environment_comparisons,
        environment_negative_control='eta_only/context_only/field_only predictions must be bitwise unchanged when only physical globals are restored',
        environment_interpretation='Each source fold contains only one environment-constant setting. Adding constants supplies information but does not provide supervised variation with which to identify their causal effect.',
        zero_new_rollouts=True,final_confirmation_used=False,
        status='DEVELOPMENT diagnosis; target labels not used by these fits, but previous pooled method investigation prevents untouched cross-scene confirmation interpretation'))
    np.savez_compressed(HERE/'factorial_development_bootstrap.npz',**boots)
    print('Scene/controller-shift factorial complete',selection,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','review']);a=p.parse_args()
    prepare() if a.action=='prepare' else review()
