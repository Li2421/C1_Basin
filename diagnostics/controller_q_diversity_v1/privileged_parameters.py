"""Cache-only diagnostic: explicit analytic parameters versus query descriptors.

The exact same bilinear architecture receives six standardized known family
parameters, padded to the existing32channels. No hidden nativeFlow weights.
This privileged family-specific control is not a general-controller solution.
"""
import argparse
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from . import train
from .experiment import HERE, read, write, sha
from .evaluate import masked_mean, interval

OUT=HERE/'checks/privileged_parameters'
VARIANTS=('factorial_toy','factorial_ring','joint')
KINDS=('eta_only','state_only','state_contact_bilinear')


def parameters(d):
    profiles=read(HERE/'protocol.json')['controllers']
    for scene,extra in read(HERE/'extra_controller_profiles.json').items():profiles[scene].update(extra)
    return np.asarray([profiles[str(sc)][str(c)]['payload']['controller_parameters']
                       for sc,c in zip(d['scene'],d['controller'])],np.float32)


def normalization(d,kind=None):
    p=parameters(d);v=p[d['split']=='train']
    return dict(mean=v.mean(0).tolist(),scale=np.maximum(v.std(0),.05).tolist(),
        clip=10.,representation='known_analytic_six_parameters',fit_split='train')


def inputs(d,norm):
    if norm.get('representation')!='known_analytic_six_parameters':return ORIGINAL_INPUTS(d,norm)
    p=np.clip((parameters(d)-np.asarray(norm['mean']))/np.asarray(norm['scale']),-10,10)
    c=np.zeros((len(p),4,32),np.float32);c[:,:,:6]=p[:,None,:]
    c*=d['entity_agent_mask'][...,None]
    x={k[7:]:jnp.asarray(d[k],jnp.float32) for k in d.files if k.startswith('entity_')}
    return x,jnp.asarray(c),jnp.asarray(d['eta'],jnp.float32)


ORIGINAL_INPUTS=train.inputs
ORIGINAL_NORMALIZATION=train.normalization


def prepare_fit_roots():
    for condition in ('queries','parameters'):
        root=OUT/condition;root.mkdir(exist_ok=True)
        (root/'training_plan.json').write_bytes((HERE/'training_plan.json').read_bytes())
        for variant in VARIANTS:
            alias='source_diverse64' if variant=='joint' else variant
            destination=root/'data'/alias;destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.is_symlink():destination.symlink_to(OUT/'data'/variant,target_is_directory=True)


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'training_plan.json').write_bytes((HERE/'training_plan.json').read_bytes())
    for variant in VARIANTS[:2]:
        target=HERE/'checks/factorial_breadth8/data'/variant
        destination=OUT/'data'/variant;destination.parent.mkdir(parents=True,exist_ok=True)
        assert not destination.exists();destination.symlink_to(target,target_is_directory=True)
    d=np.load(HERE/'data/source_diverse64/dataset.npz');keep=d['controller']!='flow'
    data={k:d[k][keep] if d[k].ndim and len(d[k])==len(keep) else d[k] for k in d.files}
    data['original_split']=data['split'].copy()
    folder=OUT/'data/joint';folder.mkdir(parents=True)
    np.savez_compressed(folder/'dataset.npz',**data)
    write(folder/'manifest.json',dict(parent_sha256=sha(HERE/'data/source_diverse64/dataset.npz'),
        dataset_sha256=sha(folder/'dataset.npz'),final_labels_used=False,
        labels='Exactly all8analyticx64TRAIN and6analyticx16VAL per scene;nativeFlow excluded for both compared representations'))
    jobs=[]
    for variant in VARIANTS[:2]:
        jobs.extend(dict(variant=variant,kind='state_contact_bilinear',seed=s,input='parameters') for s in train.SEEDS)
    for kind in KINDS:
        jobs.extend(dict(variant='joint',kind=kind,seed=s,input='queries') for s in train.SEEDS)
    jobs.extend(dict(variant='joint',kind='state_contact_bilinear',seed=s,input='parameters') for s in train.SEEDS)
    write(OUT/'jobs.json',jobs)
    write(OUT/'protocol.json',dict(new_rollouts=0,final_labels_used=False,main_model_unchanged=True,
        hypothesis='Query descriptors entangle controller identity with scene/state; canonical family parameters may transfer differently',
        caveat='Six parameters specify the analytic field, not the full scene-specific FF initialization/safety kernel. A negative result does not establish complete input sufficiency.',
        models='Same bilinear model and32controller-channel shape;known6parameters+26zeros versus nominal14+contact18',
        normalization='Source TRAIN only;no scene ID;exact same TRAIN labels and source checkpoint selection',
        code_sha256=sha(__file__)))
    prepare_fit_roots()


def fit(index):
    job=read(OUT/'jobs.json')[index];variant=job['variant'];kind=job['kind'];seed=job['seed']
    train.HERE=OUT/job['input']
    variant='source_diverse64' if variant=='joint' else variant
    if job['input']=='parameters':
        train.inputs=inputs;train.normalization=normalization
    train.fit(kind,seed,variant=variant)
    record=train.HERE/'fits'/variant/kind/f'seed{seed}'/'complete.json'
    r=read(record);r.update(controller_input=job['input'],diagnostic_code_sha256=sha(__file__))
    write(record,r)


def review():
    rows=[]
    for variant in VARIANTS:
        d=np.load(OUT/'data'/variant/'dataset.npz')
        reference_root=OUT/'queries' if variant=='joint' else HERE/'checks/factorial_breadth8'
        alias='source_diverse64' if variant=='joint' else variant
        for kind in ('state_contact_bilinear','state_contact'):
            pred=[]
            query_root=reference_root if kind=='state_contact_bilinear' else OUT/'queries'
            for root,v in [(query_root,alias),(OUT/'parameters',alias)]:
                z=[np.load(root/'fits'/v/kind/f'seed{s}'/'source_predictions.npz')['logits'] for s in train.SEEDS]
                pred.append(np.mean([1/(1+np.exp(-np.clip(v,-80,80))) for v in z],0))
            for scene in sorted(set(d['scene'])):
                for seen in (True,False):
                    names=[f'analytic{i}' for i in (range(4) if seen else (4,5))]
                    ids=np.flatnonzero((d['scene']==scene)&(d['original_split']=='validation')&np.isin(d['controller'],names));n=len(ids)//len(names)
                    known=d['success'][ids]+d['failure'][ids]>0
                    y=d['success'][ids]/np.maximum(d['success'][ids]+d['failure'][ids],1)
                    errors=[masked_mean(((p[ids]-y)**2).reshape(n,-1,16),known.reshape(n,-1,16),axis=(1,2)) for p in pred]
                    gain=errors[0]-errors[1];boot=np.random.default_rng(2026100953).integers(n,size=(20000,n))
                    rows.append(dict(training_variant=variant,architecture=kind,evaluation_scene=str(scene),seen_analytic_parameters=seen,
                        physical_states=n,query_Q_MSE=float(errors[0].mean()),parameter_Q_MSE=float(errors[1].mean()),
                        improvement=float(gain.mean()),CI95=interval(gain[boot].mean(1))))
    write(OUT/'comparison.json',dict(rows=rows,new_rollouts=0,final_labels_used=False,
        interpretation='DEVELOPMENT diagnostic only;samebilinear architecture,labels,budget;family-specific privileged parameter input is not general arbitrary-controller identification'))
    print('Privileged-parameter comparison complete')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fit','review']);p.add_argument('--index',type=int);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='fit':fit(a.index)
    else:review()
