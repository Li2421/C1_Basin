"""Complete the source-nested shrinkage critic's structurally matched ablation."""
from common import *
from select_models import inner_selection,predictions
from select_models_v2 import combine_model
from critic_model import metrics,choices
from train_critic import key_for,load_data


def action_folder(source,init):return HERE/'critic_action_only_fits'/('action_only_'+init)/key_for(source)


def inner_action(source,configuration):
    source=sorted(source);rows=[]
    for init in ('scratch','warm'):
        histories={}
        for held in source:
            path=action_folder([t for t in source if t!=held],init)
            assert (path/'COMPLETE.receipt.json').exists()
            assert set(read(path/'TASK.json')['source_types'])==set(source)-{held}
            histories[held]={r['step']:r for r in read(path/'HISTORY.json')}
        for step in configuration['steps']:
            folds={held:histories[held][step]['validation'][held] for held in source}
            ranks=[v['ranking_accuracy'] for v in folds.values() if v['ranking_accuracy'] is not None]
            rows.append(dict(initialization=init,step=step,mean_nll=float(np.mean([v['nll'] for v in folds.values()])),
                mean_ranking_accuracy=float(np.mean(ranks)) if ranks else None,source_only_folds=folds))
    pick=min(rows,key=lambda r:(round(r['mean_nll'],12),r['step'],-(r['mean_ranking_accuracy'] or 0),r['initialization']!='scratch'))
    return dict(selected=pick,history=rows,source_types=source,outer_labels_used=False)


def action_predictions(source,selection):
    selected=selection['selected'];folder=action_folder(source,selected['initialization']);receipt=read(folder/'COMPLETE.receipt.json')
    assert set(read(folder/'TASK.json')['source_types'])==set(source)
    checkpoint=folder/f"step{selected['step']}.msgpack";packet=folder/f"step{selected['step']}_predictions.npz"
    assert sha(checkpoint)==receipt['files'][str(checkpoint)] and sha(packet)==receipt['files'][str(packet)]
    with np.load(packet,allow_pickle=False) as z:p=z['probability'];names=z['names']
    model=dict(source_types=sorted(source),mode='action_only',initialization=selected['initialization'],step=selected['step'])
    for key,path in (('checkpoint',checkpoint),('transform',folder/'TRANSFORM.json'),('fit_receipt',folder/'COMPLETE.receipt.json')):
        model[key+'_path']=str(path);model[key+'_sha256']=sha(path)
    return p,names,model


def select_no_state(source,configuration,data):
    source=sorted(source);first=inner_selection(source,'no_state',configuration);second=inner_action(source,configuration);cache={};rows=[]
    for held in source:
        inside=[t for t in source if t!=held]
        p,names,_=predictions(inside,'no_state',first);q,names2,_=action_predictions(inside,second)
        assert np.array_equal(names,names2) and np.array_equal(names,data['names']);cache[held]=(p,q)
    for weight in read(HERE/'ITERATION_V2_DESIGN.json')['lambda_grid']:
        folds={held:metrics(weight*p+(1-weight)*q,data['labels'],data['types'],[held])[held] for held,(p,q) in cache.items()}
        rows.append(dict(full_weight=weight,source_mean_nll=float(np.mean([v['nll'] for v in folds.values()])),source_only_folds=folds))
    picked=min(rows,key=lambda r:(round(r['source_mean_nll'],12),-r['full_weight']))
    p,names,m=predictions(source,'no_state',first);q,names2,n=action_predictions(source,second)
    weight=picked['full_weight'];model=combine_model(m,n,weight);model['mode']='no_state'
    return weight*p+(1-weight)*q,model,dict(source_types=source,first=first,second=second,
        lambda_curve=rows,selected=picked,outer_labels_used=False)


def main():
    assert not (HERE/'DEVELOPMENT_CHOICES_V2B.receipt.json').exists()
    configuration=read(HERE/'CRITIC_TRAINING_PLAN.json');data=load_data(configuration['bundle_path'])
    model=read(HERE/'SELECTED_MODELS_V2.json');folds=read(HERE/'DEVELOPMENT_OUTER_FOLDS_V2.json');types=data['types']
    with np.load(HERE/'DEVELOPMENT_CHOICES_V2.npz') as z:output={k:z[k].copy() for k in z.files if k not in ('names','types')}
    probabilities=np.zeros((3,62,48,13),np.float32)
    for held in sorted(set(types)):
        source=sorted(set(types)-{held});p,m,selection=select_no_state(source,configuration,data);mask=types==held
        output['no_state'][mask]=choices(p)['mean'][mask];probabilities[:,mask]=p[:,mask];model['outer'][str(held)]['no_state']=m
        [row for row in folds if row['held_type']==held][0]['source_only_selections']['no_state']=selection
    p,m,selection=select_no_state(sorted(set(types)),configuration,data)
    model['all_types']['no_state']=dict(model=m,source_only_selection=selection)
    model['utc']=utc();model['matched_ablations']=True
    write(HERE/'SELECTED_MODELS_V2B.json',model);write(HERE/'DEVELOPMENT_OUTER_FOLDS_V2B.json',folds)
    np.savez_compressed(HERE/'DEVELOPMENT_CHOICES_V2B.npz',names=data['names'],types=types,**output)
    np.savez_compressed(HERE/'MATCHED_NO_STATE_PROBABILITIES_V2B.npz',probability=probabilities,names=data['names'],types=types)
    seal('DEVELOPMENT_CHOICES_V2B.receipt.json',[HERE/'SELECTED_MODELS_V2B.json',HERE/'DEVELOPMENT_OUTER_FOLDS_V2B.json',
        HERE/'DEVELOPMENT_CHOICES_V2B.npz',HERE/'MATCHED_NO_STATE_PROBABILITIES_V2B.npz',HERE/'select_matched_v2b.py',
        HERE/'DEVELOPMENT_CHOICES_V2.receipt.json',HERE/'ACTION_ONLY_TRAINING_PLAN.receipt.json',HERE/'ITERATION_V2_MATCHED_ABLATIONS_DESIGN.receipt.json'],
        matched_no_state_ablation=True,outer_labels_used_for_selection=False,new_rollouts=0)
    print('Matched v2b no-state ablation and final choices sealed',flush=True)


if __name__=='__main__':main()
