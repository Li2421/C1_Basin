"""Source-only checkpoint/probability mixing and global trigger selection."""
from common import *
from source_select_models import select,predictions
from train_critic import load_data
from menu_c6_v2 import METHODS,choices,TRIGGERS

def actions_for(data,ci,si):
    rows=[]
    for ai,v in enumerate(data['action'][ci,si]):
        rows.append(dict(eta_a=v[:3].astype(float).tolist(),eta_b=v[3:6].astype(float).tolist(),switch_mode='progress' if v[12] else 'fixed',switch_step=round(float(v[9])*700),late_source='requery' if v[13] else 'given',progress_threshold=float(data['thresholds'][ci,si,ai]),trigger=list(TRIGGERS)[int(data['trigger_indices'][ci,si,ai])]))
    return rows

def choose_slots(data,pn,pf,mask,global_trigger):
    indices=np.full((len(data['names']),48,6),-1,np.int32)
    for ci in np.flatnonzero(mask):
        for si in range(48):
            selected=choices(actions_for(data,ci,si),pn[:,ci,si],pf[:,ci,si],global_trigger)
            for mi,method in enumerate((*METHODS[:5],METHODS[7])):indices[ci,si,mi]=selected[method]
    return indices

def outcome_summary(data,selected,mask):
    result={};observed=data['labels'][...,0]
    for t in sorted(set(data['types'][mask])):
        values=[];unscored=[]
        for ci in np.flatnonzero(mask&(data['types']==t)):
            selected_labels=observed[ci,np.arange(48),selected[ci]];values.append(np.mean(selected_labels==1));unscored.append(np.mean(selected_labels<0))
        result[str(t)]=dict(success_lower=float(np.mean(values)),unscored_fraction=float(np.mean(unscored)),controllers=len(values))
    return result

def global_setting(source,configuration,data,record):
    alltypes=data['types'];out={t:[] for t in list(TRIGGERS)[1:]};noC=np.zeros((3,122,48,53),np.float32)
    for held in source:
        inside=[t for t in source if t!=held];pn,_,_=predictions(inside,'no_controller',record['branches']['no_controller']);mask=alltypes==held;noC[:,mask]=pn[:,mask]
    for setting in out:
        slots=choose_slots(data,noC,noC,np.isin(alltypes,source),setting);folds=outcome_summary(data,slots[:,:,4],np.isin(alltypes,source));out[setting]=dict(folds=folds,success_lower_macro=float(np.mean([r['success_lower'] for r in folds.values()])),unscored_fraction_macro=float(np.mean([r['unscored_fraction'] for r in folds.values()])))
    chosen=min(out,key=lambda k:(-out[k]['success_lower_macro'],out[k]['unscored_fraction_macro'],list(TRIGGERS).index(k)))
    return dict(setting=chosen,source_types=source,curve=out,outer_labels_used=False,selection='type-equal q0 actual chosen success lower; unknown0, report unscored; ties lower unscored then sensitive/medium/conservative',support='53 labeled physical training slots only; no hypothetical full-menu oracle')

def main():
    config=read(HERE/'CRITIC_TRAINING_PLAN.json');data=load_data();types=sorted(set(data['types'].tolist()));outer={};folds=[];noC=np.zeros((3,122,48,53),np.float32);full=np.zeros_like(noC);allchoices=np.full((122,48,6),-1,np.int32)
    for held in types:
        source=[t for t in types if t!=held];p,pn,pf,model,record=select(source,config,data);global_trigger=global_setting(source,config,data,record);mask=data['types']==held;noC[:,mask]=pn[:,mask];full[:,mask]=p[:,mask];slots=choose_slots(data,pn,p,mask,global_trigger['setting']);allchoices[mask]=slots[mask];outer[held]=dict(full=model,no_controller=model['no_controller'],global_trigger=global_trigger);folds.append(dict(held_type=held,source_types=source,source_only_selection=record,global_trigger=global_trigger,success_and_unscored={m:outcome_summary(data,slots[:,:,mi],mask)[held] for mi,m in enumerate((*METHODS[:5],METHODS[7]))},ranking=__import__('critic_model').metrics(p,data['labels'],data['types'],[held],data['candidate_indices'])[held]))
    p,pn,pf,model,record=select(types,config,data);global_trigger=global_setting(types,config,data,record);final=dict(full=model,no_controller=model['no_controller'],global_trigger=global_trigger,source_only_selection=record)
    np.savez_compressed(HERE/'DEVELOPMENT_PROBABILITIES.npz',names=data['names'],types=data['types'],full=full,no_controller=noC,chosen_slots=allchoices)
    write(HERE/'DEVELOPMENT_OUTER_FOLDS.json',folds);write(HERE/'SELECTED_MODELS.json',dict(utc=utc(),outer=outer,all_types=final));seal('SELECTED_MODELS.receipt.json',[HERE/n for n in ('DEVELOPMENT_PROBABILITIES.npz','DEVELOPMENT_OUTER_FOLDS.json','SELECTED_MODELS.json','select_c6_v2.py','source_select_models.py','CRITIC_TRAINING_PLAN.receipt.json')],all_selection_source_only=True,new_evaluation_labels_read=False)
    print('19 LOTO folds and final model selected; source-only global triggers sealed',flush=True)

if __name__=='__main__':main()
