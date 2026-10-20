"""Source-nested probability shrinkage; lambda=1 contains the original critic."""
from common import *
from select_models import inner_selection,predictions
from critic_model import choices,metrics
from train_critic import load_data


def combine_model(full,no_controller,weight):
    assert full['source_types']==no_controller['source_types']
    return dict(kind='probability_shrinkage',mode='full',source_types=full['source_types'],
        full_weight=float(weight),full=full,no_controller=no_controller,
        ensemble_pairing='Corresponding seeds17/23/41; average probabilities, then mean argmax')


def select(source,configuration,data,initialization=None):
    source=sorted(source)
    full=inner_selection(source,'full',configuration,initialization)
    no_controller=inner_selection(source,'no_controller',configuration,initialization)
    rows=[];cache={}
    for held in source:
        inside=[t for t in source if t!=held]
        pf,order,_=predictions(inside,'full',full)
        pn,order2,_=predictions(inside,'no_controller',no_controller)
        assert np.array_equal(order,data['names']) and np.array_equal(order2,order)
        cache[held]=(pf,pn)
    for weight in read(HERE/'ITERATION_V2_DESIGN.json')['lambda_grid']:
        folds={}
        for held,(pf,pn) in cache.items():
            p=weight*pf+(1-weight)*pn
            folds[held]=metrics(p,data['labels'],data['types'],[held])[held]
        rows.append(dict(full_weight=weight,source_mean_nll=float(np.mean([m['nll'] for m in folds.values()])),source_only_folds=folds))
    picked=min(rows,key=lambda r:(round(r['source_mean_nll'],12),-r['full_weight']))
    pf,order,mf=predictions(source,'full',full);pn,order2,mn=predictions(source,'no_controller',no_controller)
    weight=picked['full_weight'];p=weight*pf+(1-weight)*pn
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    record=dict(source_types=source,initialization_constraint=initialization,full_selection=full,
        no_controller_selection=no_controller,lambda_curve=rows,selected=picked,outer_labels_used=False)
    return p,combine_model(mf,mn,weight),record


def main():
    assert not (HERE/'DEVELOPMENT_CHOICES_V2.receipt.json').exists()
    for path,digest in read(HERE/'ITERATION_V2_DESIGN.receipt.json')['files'].items():assert sha(path)==digest
    configuration=read(HERE/'CRITIC_TRAINING_PLAN.json');data=load_data(configuration['bundle_path'])
    old=read(HERE/'SELECTED_MODELS.json');models={};folds=[];ns=48;nc=62;types=data['types']
    with np.load(HERE/'DEVELOPMENT_CHOICES.npz',allow_pickle=False) as z:
        output={k:z[k].copy() for k in z.files if k not in ('names','types')}
    probability=np.zeros((3,nc,ns,13),np.float32)
    for held in sorted(set(types)):
        source=sorted(set(types)-{held});mask=types==held;models[str(held)]=dict(old['outer'][str(held)]);records={}
        for init in (None,'scratch','warm'):
            p,model,record=select(source,configuration,data,init)
            method='pipeline' if init is None else 'pipeline_'+init
            output[method][mask]=choices(p)['mean'][mask]
            key='full' if init is None else 'full_'+init;models[str(held)][key]=model;records[key]=record
            if init is None:
                probability[:,mask]=p[:,mask];output['pipeline_conservative'][mask]=choices(p)['conservative'][mask]
        folds.append(dict(held_type=str(held),source_only_selections=records))
    source=sorted(set(types));p,model,record=select(source,configuration,data)
    final=dict(old['all_types']);final['full']=dict(model=model,source_only_selection=record)
    write(HERE/'SELECTED_MODELS_V2.json',dict(utc=utc(),outer=models,all_types=final,
        rule='Positive controller probability shrinkage, source-only nested branch selection and lambda'))
    write(HERE/'DEVELOPMENT_OUTER_FOLDS_V2.json',folds)
    np.savez_compressed(HERE/'DEVELOPMENT_CHOICES_V2.npz',names=data['names'],types=types,**output)
    np.savez_compressed(HERE/'DEVELOPMENT_PROBABILITIES_V2.npz',names=data['names'],types=types,full=probability)
    seal('DEVELOPMENT_CHOICES_V2.receipt.json',[HERE/'SELECTED_MODELS_V2.json',HERE/'DEVELOPMENT_OUTER_FOLDS_V2.json',
        HERE/'DEVELOPMENT_CHOICES_V2.npz',HERE/'DEVELOPMENT_PROBABILITIES_V2.npz',HERE/'select_models_v2.py',
        HERE/'ITERATION_V2_DESIGN.receipt.json',HERE/'DEVELOPMENT_CHOICES.receipt.json'],
        new_rollouts=0,new_training=0,outer_labels_used_for_selection=False)
    print('v2 source-nested shrinkage choices sealed',flush=True)


if __name__=='__main__':main()
