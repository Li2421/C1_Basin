from common import *
from train_critic import load_data
from select_mixed import select
from critic_model import choices
from baselines import LEARNED,source_fixed
from statistics_pipeline import evaluate_group,equivalence
import jax


def main():
    configuration=read(HERE/'CRITIC_TRAINING_PLAN.json');data=load_data(configuration['bundle_path']);alltypes=set(data['types']);results={};models={}
    for family,kinds in [('learned',LEARNED),('optimization',{'distributed_shooting','centralized_sampling','cross_entropy'}),('search',{'prioritized_astar'})]:
        mask=np.isin(data['types'],list(kinds));source=sorted(alltypes-set(kinds))
        if not mask.any() or not source:continue
        p,model,record=select(source,configuration,data);models[family]=dict(model=model,source_selection=record)
        labels={k:data[k][mask,...,:1] for k in ('labels','uids','collision','timeout','unknown')};labels.update(names=data['names'][mask],types=data['types'][mask]);labels['uid_class']=equivalence(labels['uids'])
        fixed,_,_=source_fixed(data,source);n=int(mask.sum())
        methods={'pipeline':choices(p)['mean'][mask],'all_Sprog':np.full((n,48),12),'all_RHR':np.full((n,48),13),'all_safety_baseline':np.full((n,48),10),'source_fixed':np.full((n,48),fixed)}
        results[family]=evaluate_group(labels,methods,True,5000)
    write(HERE/'LEAVE_ONE_FAMILY.json',dict(utc=utc(),evidence='DEVELOPMENT descriptive',results=results,models=models,source_only=True))
    seal('LEAVE_ONE_FAMILY.receipt.json',[HERE/'LEAVE_ONE_FAMILY.json',HERE/'family_validation.py'])
    print('Descriptive leave-one-family validation complete')


if __name__=='__main__':
    with jax.default_matmul_precision('highest'):main()
