"""Nested source-type-only checkpoint/init selection, never outer-label tuning."""
from common import *
from train_critic import key_for,folder_for,load_data
from critic_model import candidate_choice,choices,MODES


def path_for(source,mode,initialization):
    return HERE/'critic_fits'/f'{mode}_{initialization}'/key_for(source)


def inner_selection(source,mode,configuration,initialization=None):
    source=sorted(source);initializations=(initialization,) if initialization else ('scratch','warm')
    rows=[];used=[]
    for init in initializations:
        histories={}
        for held in source:
            fit_source=[t for t in source if t!=held];folder=path_for(fit_source,mode,init)
            task=read(folder/'TASK.json');assert set(task['source_types'])==set(fit_source)
            assert (folder/'COMPLETE.receipt.json').exists()
            histories[held]={r['step']:r for r in read(folder/'HISTORY.json')}
            used.append(str(folder/'COMPLETE.receipt.json'))
        for step in configuration['steps']:
            # Access only the inner held SOURCE type, even when the cached fit
            # also evaluated the other omitted (outer held) type for reporting.
            folds={held:histories[held][step]['validation'][held] for held in source}
            nll=float(np.mean([r['nll'] for r in folds.values()]))
            rank=[r['ranking_accuracy'] for r in folds.values() if r['ranking_accuracy'] is not None]
            rows.append(dict(initialization=init,step=step,mean_nll=nll,
                mean_ranking_accuracy=float(np.mean(rank)) if rank else None,source_only_folds=folds))
    pick=min(rows,key=lambda r:(round(r['mean_nll'],12),r['step'],
        -(r['mean_ranking_accuracy'] if r['mean_ranking_accuracy'] is not None else -1.),r['initialization']!='scratch'))
    return dict(selected=pick,history=rows,source_types=source,mode=mode,fit_receipts=sorted(set(used)),outer_labels_used=False)


def predictions(source,mode,selection):
    chosen=selection['selected'];folder=path_for(source,mode,chosen['initialization'])
    task=read(folder/'TASK.json');assert set(task['source_types'])==set(source)
    checkpoint=folder/f"step{chosen['step']}.msgpack";packet=folder/f"step{chosen['step']}_predictions.npz"
    receipt=read(folder/'COMPLETE.receipt.json')
    assert sha(checkpoint)==receipt['files'][str(checkpoint)] and sha(packet)==receipt['files'][str(packet)]
    with np.load(packet,allow_pickle=False) as z:probability=z['probability'];names=z['names']
    return probability,names,dict(source_types=sorted(source),mode=mode,initialization=chosen['initialization'],step=chosen['step'],
        checkpoint_path=str(checkpoint),checkpoint_sha256=sha(checkpoint),transform_path=str(folder/'TRANSFORM.json'),
        transform_sha256=sha(folder/'TRANSFORM.json'),fit_receipt_path=str(folder/'COMPLETE.receipt.json'),
        fit_receipt_sha256=sha(folder/'COMPLETE.receipt.json'))


def main():
    assert not (HERE/'DEVELOPMENT_CHOICES.receipt.json').exists()
    configuration=read(HERE/'CRITIC_TRAINING_PLAN.json');data=load_data(configuration['bundle_path'])
    types=data['types'];all_types=sorted(set(types.tolist()));names=data['names'];nc=len(names)
    output={k:np.full((nc,48),-1,np.int32) for k in ('pipeline','pipeline_conservative','pipeline_scratch','pipeline_warm','no_controller','no_state')}
    folds=[];models={};final={};predictions_all={mode:np.zeros((3,nc,48,13),np.float32) for mode in MODES}
    for held in all_types:
        source=[t for t in all_types if t!=held];mask=types==held;records={}
        for mode in MODES:
            selection=inner_selection(source,mode,configuration);p,order,model=predictions(source,mode,selection)
            assert np.array_equal(order,names)
            selected=choices(p);method='pipeline' if mode=='full' else mode
            output[method][mask]=selected['mean'][mask];predictions_all[mode][:,mask]=p[:,mask]
            records[mode]=selection;models.setdefault(str(held),{})[mode]=model
            if mode=='full':
                output['pipeline_conservative'][mask]=selected['conservative'][mask]
                for init in ('scratch','warm'):
                    own=inner_selection(source,mode,configuration,initialization=init)
                    p2,order,other=predictions(source,mode,own)
                    output[f'pipeline_{init}'][mask]=choices(p2)['mean'][mask]
                    records[f'full_{init}']=own;models[str(held)][f'full_{init}']=other
        folds.append(dict(held_type=str(held),source_types=source,source_only_selections=records))
    for mode in MODES:
        selection=inner_selection(all_types,mode,configuration);p,order,model=predictions(all_types,mode,selection)
        final[mode]=dict(model=model,source_only_selection=selection)
    frozen=read(D/'crosstype1_learned_zoo_selector_v1/FROZEN_PIPELINE.json')
    from selector import Regression
    old=Regression.from_dict(frozen['model']).choose(data['features'])
    mapping={0:10,5:11,6:12};assert all(int(i) in mapping for i in old)
    output['frozen_controller_selector']=np.broadcast_to(np.asarray([mapping[int(i)] for i in old])[:,None],(nc,48))
    candidate=read(HERE/'CANDIDATES.json')['states'][:48]
    original=np.asarray([r['original_pipeline_choice'] for r in candidate]);generator=np.ones(48,np.int32)
    output['original_pipeline']=np.broadcast_to(original[None],(nc,48));output['generator_only']=np.broadcast_to(generator[None],(nc,48))
    for name,a in (('safety_baseline',10),('S500',11),('Sprog',12)):output['all_'+name]=np.full((nc,48),a,np.int32)
    assert all(np.all((v>=0)&(v<13)) for v in output.values())
    write(HERE/'DEVELOPMENT_OUTER_FOLDS.json',folds);write(HERE/'SELECTED_MODELS.json',dict(utc=utc(),outer=models,all_types=final,
        selection_rule='Source-only nested ensemble NLL; primary uses ensemble mean argmax; conservative separately reported'))
    np.savez_compressed(HERE/'DEVELOPMENT_CHOICES.npz',names=names,types=types,**output)
    np.savez_compressed(HERE/'DEVELOPMENT_PROBABILITIES.npz',names=names,types=types,**predictions_all)
    seal('DEVELOPMENT_CHOICES.receipt.json',[HERE/'DEVELOPMENT_OUTER_FOLDS.json',HERE/'SELECTED_MODELS.json',
        HERE/'DEVELOPMENT_CHOICES.npz',HERE/'DEVELOPMENT_PROBABILITIES.npz',HERE/'select_models.py',HERE/'CRITIC_TRAINING_PLAN.receipt.json'],
        all_preprocessing_and_checkpoint_selection_source_types_only=True,new_evaluation_outcomes_read=False)
    print('Nine outer source-only model selections and all-type references sealed',flush=True)


if __name__=='__main__':main()
