"""Source-only nested selection; all outer curves retained for reporting."""
from common import *
from train_critic import key_for,folder_for,load_data
from critic_model import metrics
def path_for(source,mode):return HERE/'critic_fits'/mode/key_for(source)
def summarize_inner(source,histories,steps):
    rows=[]
    for step in steps:
        folds={held:histories[held][step]['validation'][held] for held in source};ranks=[r['ranking_accuracy'] for r in folds.values() if r['ranking_accuracy'] is not None]
        rows.append(dict(step=step,mean_nll=float(np.mean([r['nll'] for r in folds.values()])),mean_ranking_accuracy=float(np.mean(ranks)) if ranks else None,source_only_folds=folds))
    selected=min(rows,key=lambda r:(round(r['mean_nll'],12),r['step'],-(r['mean_ranking_accuracy'] if r['mean_ranking_accuracy'] is not None else -1)))
    return dict(source_types=sorted(source),selected=selected,history=rows,outer_labels_used=False)
def inner_selection(source,mode,configuration):
    histories={}
    for held in source:
        folder=path_for([t for t in source if t!=held],mode);assert (folder/'COMPLETE.receipt.json').exists();assert set(read(folder/'TASK.json')['source_types'])==set(source)-{held};histories[held]={r['step']:r for r in read(folder/'HISTORY.json')}
    return summarize_inner(source,histories,configuration['steps'])
def predictions(source,mode,selection):
    folder=path_for(source,mode);assert (folder/'COMPLETE.receipt.json').exists();assert set(read(folder/'TASK.json')['source_types'])==set(source);step=selection['selected']['step'];checkpoint=folder/f'step{step}.msgpack';packet=folder/f'step{step}_predictions.npz';receipt=read(folder/'COMPLETE.receipt.json')
    assert sha(checkpoint)==receipt['files'][str(checkpoint)] and sha(packet)==receipt['files'][str(packet)]
    with np.load(packet,allow_pickle=False) as z:p=z['probability'];names=z['names']
    model=dict(mode=mode,source_types=sorted(source),initialization='scratch',step=step)
    for key,path in [('checkpoint',checkpoint),('transform',folder/'TRANSFORM.json'),('fit_receipt',folder/'COMPLETE.receipt.json')]:model[key+'_path']=str(path);model[key+'_sha256']=sha(path)
    return p,names,model
def select(source,configuration,data):
    selections={mode:inner_selection(source,mode,configuration) for mode in ('full','no_controller')};cache={}
    for held in source:
        inside=[t for t in source if t!=held];pf,names,_=predictions(inside,'full',selections['full']);pn,names2,_=predictions(inside,'no_controller',selections['no_controller']);assert np.array_equal(names,names2) and np.array_equal(names,data['names']);mask=data['types']==held;cache[held]=(pf[:,mask],pn[:,mask],data['labels'][mask],data['types'][mask],data['candidate_indices'][mask])
    curve=[]
    for weight in configuration['lambda_grid']:
        folds={}
        for held,(pf,pn,labels,types,indices) in cache.items():folds[held]=metrics(weight*pf+(1-weight)*pn,labels,types,[held],indices)[held]
        curve.append(dict(full_weight=weight,source_mean_nll=float(np.mean([r['nll'] for r in folds.values()])),source_only_folds=folds))
    picked=min(curve,key=lambda r:(round(r['source_mean_nll'],12),-r['full_weight']));pf,names,mf=predictions(source,'full',selections['full']);pn,names2,mn=predictions(source,'no_controller',selections['no_controller']);weight=picked['full_weight']
    model=dict(kind='probability_shrinkage',source_types=sorted(source),full_weight=weight,full=mf,no_controller=mn)
    return weight*pf+(1-weight)*pn,pn,pf,model,dict(source_types=sorted(source),branches=selections,lambda_curve=curve,selected=picked,outer_labels_used=False)
def choose_sampled(p,indices):
    score=p.mean(0);sd=p.std(0);diagonal=indices//54==indices//6%9;fixed=diagonal|(indices%6==3)
    def pick(values,mask):
        allowed=np.where(mask,values,-np.inf);maximum=allowed.max(-1,keepdims=True);best_id=np.where(mask&(values==maximum),indices,np.iinfo(np.int32).max).min(-1);return np.argmax(indices==best_id[...,None],-1)
    zero=(indices//54==8)|(indices//6%9==8);progress=indices%6==5
    return dict(self_schedule=pick(score,np.ones_like(diagonal)),fixed500=pick(score,fixed),no_schedule=pick(score,diagonal),conservative=pick(score-sd,np.ones_like(diagonal)),no_progress=pick(score,~progress),no_zero=pick(score,~zero),legacy_fixed=pick(score,~zero&~progress))
def main():
    config=read(HERE/'CRITIC_TRAINING_PLAN.json');data=load_data();types=data['types'];all_types=sorted(set(types.tolist()));models={};folds=[];output={k:np.full((122,48),-1,np.int32) for k in ('self_schedule','fixed500','no_schedule','conservative','no_progress','no_zero','legacy_fixed','no_controller','single_branch')};probability=np.zeros((3,122,48,34),np.float32);noC=np.zeros_like(probability)
    for held in all_types:
        source=[t for t in all_types if t!=held];p,pn,pf,model,record=select(source,config,data);mask=types==held;selected=choose_sampled(p,data['candidate_indices']);models[str(held)]=dict(full=model,no_controller=model['no_controller']);folds.append(dict(held_type=held,source_types=source,source_only_selection=record))
        for key in selected:output[key][mask]=selected[key][mask]
        output['no_controller'][mask]=choose_sampled(pn,data['candidate_indices'])['self_schedule'][mask];output['single_branch'][mask]=choose_sampled(pf,data['candidate_indices'])['self_schedule'][mask];probability[:,mask]=p[:,mask];noC[:,mask]=pn[:,mask]
    p,pn,pf,model,record=select(all_types,config,data);final=dict(full=model,no_controller=model['no_controller'],source_only_selection=record)
    np.savez_compressed(HERE/'DEVELOPMENT_CHOICES.npz',names=data['names'],types=types,**output);np.savez_compressed(HERE/'DEVELOPMENT_PROBABILITIES.npz',names=data['names'],types=types,full=probability,no_controller=noC)
    write(HERE/'SELECTED_MODELS.json',dict(utc=utc(),outer=models,all_types=final));write(HERE/'DEVELOPMENT_OUTER_FOLDS.json',folds)
    seal('DEVELOPMENT_CHOICES.receipt.json',[HERE/'DEVELOPMENT_CHOICES.npz',HERE/'DEVELOPMENT_PROBABILITIES.npz',HERE/'SELECTED_MODELS.json',HERE/'DEVELOPMENT_OUTER_FOLDS.json',HERE/'select_models.py',HERE/'CRITIC_TRAINING_PLAN.receipt.json'],all_selection_source_only=True,new_evaluation_labels_read=False)
if __name__=='__main__':main()
