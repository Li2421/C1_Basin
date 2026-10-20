"""Nested source-only model selection; all four branches share fit caches."""
from common import *
from select_models import inner_selection,predictions
from critic_model import choices,metrics
from train_critic import load_data
from baselines import source_fixed,few_label,LEARNED,NONLEARNED


def combine(first,second,weight,mode):
    assert first['source_types']==second['source_types']
    return dict(kind='probability_shrinkage',mode=mode,source_types=first['source_types'],full_weight=float(weight),full=first,no_controller=second,ensemble_pairing='Corresponding seeds17/23/41')


def select(source,configuration,data,mode='full',initialization=None):
    first_mode,second_mode=('full','no_controller') if mode=='full' else ('no_state','action_only')
    source=sorted(source)
    if len(source)==1:
        from single_source_selection import choose,held_predictions
        first=choose(source,first_mode,configuration,data,initialization)
        second=choose(source,second_mode,configuration,data,initialization)
        a=held_predictions(source,first_mode,first,data);b=held_predictions(source,second_mode,second,data)
        curve=[]
        for weight in (.1,.25,.5,.75,1.):
            metric=metrics(weight*a+(1-weight)*b,data['labels'],data['types'],source)[source[0]]
            curve.append(dict(full_weight=weight,source_mean_nll=metric['nll'],source_controller_inner=True))
        chosen=min(curve,key=lambda r:(round(r['source_mean_nll'],12),-r['full_weight']))
        a,_,ma=predictions(source,first_mode,first);b,_,mb=predictions(source,second_mode,second);w=chosen['full_weight']
        return w*a+(1-w)*b,combine(ma,mb,w,mode),dict(source_types=source,first=first,second=second,lambda_curve=curve,selected=chosen,source_controller_inner=True,outer_labels_used=False)
    first=inner_selection(source,first_mode,configuration,initialization);second=inner_selection(source,second_mode,configuration,initialization)
    cache={}
    for held in source:
        inside=[t for t in source if t!=held]
        a,n,_=predictions(inside,first_mode,first);b,n2,_=predictions(inside,second_mode,second)
        assert np.array_equal(n,n2);cache[held]=(a,b)
    curve=[]
    for weight in (.1,.25,.5,.75,1.):
        folds={held:metrics(weight*a+(1-weight)*b,data['labels'],data['types'],[held])[held] for held,(a,b) in cache.items()}
        curve.append(dict(full_weight=weight,source_mean_nll=float(np.mean([m['nll'] for m in folds.values()])),source_only_folds=folds))
    chosen=min(curve,key=lambda r:(round(r['source_mean_nll'],12),-r['full_weight']))
    a,n,ma=predictions(source,first_mode,first);b,n2,mb=predictions(source,second_mode,second);w=chosen['full_weight']
    return w*a+(1-w)*b,combine(ma,mb,w,mode),dict(source_types=source,first=first,second=second,lambda_curve=curve,selected=chosen,outer_labels_used=False)


def baselines(data,source_by_type):
    from legacy_inference import choices as old_choices
    nc=len(data['names']);ns=len(data['h']);out={}
    for method,index in [('all_safety_baseline',10),('all_S500',11),('all_Sprog',12),('all_RHR',13),('all_RHR_b',14),('all_RHR_c',15)]:out[method]=np.full((nc,ns),index,np.int32)
    out['source_fixed']=np.empty((nc,ns),np.int32);out['source_mean_fixed']=np.empty((nc,ns),np.int32)
    records={}
    for held,source in source_by_type.items():
        a,b,record=source_fixed(data,source);mask=data['types']==held
        out['source_fixed'][mask]=a;out['source_mean_fixed'][mask]=b;records[held]=record
    for k in (2,4,8):out[f'few_{k}']=np.broadcast_to(few_label(data,k)[:,None],(nc,ns)).copy()
    candidates=read(HERE/'CANDIDATES.json')['states'][:ns]
    out['original_pipeline']=np.broadcast_to(np.asarray([c['original_pipeline_choice'] for c in candidates])[None],(nc,ns)).copy()
    out['generator_only']=np.ones((nc,ns),np.int32);out['old_pipeline']=old_choices(data)
    return out,records


def main():
    import jax
    assert jax.default_backend()=='gpu'
    assert not (HERE/'DEVELOPMENT_CHOICES.receipt.json').exists()
    configuration=read(HERE/'CRITIC_TRAINING_PLAN.json');data=load_data(configuration['bundle_path']);types=data['types'];alltypes=sorted(set(types));nc=len(types)
    source_by={str(t):[x for x in alltypes if x!=t] for t in alltypes}
    output,baseline_records=baselines(data,source_by);models={};folds=[];probs={m:np.zeros((3,nc,48,16),np.float32) for m in ('full','no_controller','no_state')}
    for method in ('pipeline','pipeline_conservative','pipeline_scratch','pipeline_warm','pipeline_single'):output[method]=np.empty((nc,48),np.int32)
    for method in ('no_controller','no_state'):output[method]=np.empty((nc,48),np.int32)
    for held,source in source_by.items():
        mask=types==held;group={};record={}
        for mode in ('full','no_state'):
            p,model,selection=select(source,configuration,data,mode);group[mode]=model;record[mode]=selection;probs[mode][:,mask]=p[:,mask]
            method='pipeline' if mode=='full' else mode;output[method][mask]=choices(p)['mean'][mask]
            if mode=='full':
                output['pipeline_conservative'][mask]=choices(p)['conservative'][mask]
                for init in ('scratch','warm'):
                    pp,mm,ss=select(source,configuration,data,'full',init);group['full_'+init]=mm;record['full_'+init]=ss;output['pipeline_'+init][mask]=choices(pp)['mean'][mask]
                single=inner_selection(source,'full',configuration);pp,_,mm=predictions(source,'full',single)
                output['pipeline_single'][mask]=choices(pp)['mean'][mask];group['full_single']=mm;record['single']=single
        selection=inner_selection(source,'no_controller',configuration);p,_,model=predictions(source,'no_controller',selection)
        group['no_controller']=model;record['no_controller']=selection;probs['no_controller'][:,mask]=p[:,mask];output['no_controller'][mask]=choices(p)['mean'][mask]
        models[held]=group;folds.append(dict(held_type=held,source_types=source,source_only_selections=record))
    final={}
    for mode in ('full','no_state'):
        _,model,selection=select(alltypes,configuration,data,mode);final[mode]=dict(model=model,source_only_selection=selection)
    selection=inner_selection(alltypes,'no_controller',configuration);_,_,model=predictions(alltypes,'no_controller',selection);final['no_controller']=dict(model=model,source_only_selection=selection)
    all_initializations={}
    for init in ('scratch','warm'):
        _,model,selection=select(alltypes,configuration,data,'full',init)
        all_initializations[init]=dict(model=model,source_only_selection=selection)
    assert all(np.all((v>=0)&(v<16)) for v in output.values())
    write(HERE/'SELECTED_MODELS.json',dict(utc=utc(),outer=models,all_types=final,all_initializations=all_initializations,source_fixed_baselines=baseline_records,outer_labels_used_for_selection=False))
    write(HERE/'DEVELOPMENT_OUTER_FOLDS.json',folds)
    np.savez_compressed(HERE/'DEVELOPMENT_CHOICES.npz',names=data['names'],types=types,**output)
    np.savez_compressed(HERE/'DEVELOPMENT_PROBABILITIES.npz',names=data['names'],types=types,**probs)
    seal('DEVELOPMENT_CHOICES.receipt.json',[HERE/n for n in ('SELECTED_MODELS.json','DEVELOPMENT_OUTER_FOLDS.json','DEVELOPMENT_CHOICES.npz','DEVELOPMENT_PROBABILITIES.npz','select_mixed.py','CRITIC_TRAINING_PLAN.receipt.json')],source_only_nested_selection=True)
    print(len(alltypes),'outer held-type models and all-development models selected')


if __name__=='__main__':
    import jax
    with jax.default_matmul_precision('highest'):main()
