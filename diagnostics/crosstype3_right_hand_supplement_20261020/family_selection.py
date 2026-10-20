"""Descriptive leave-one-family fits use the same source-only nested rules."""
from common import *
from select_models import select,choose_sampled
from train_critic import load_data
from critic_model import metrics
from families import family_groups
def main():
    data=load_data();types=data['types'];alltypes=sorted(set(types));config=read(HERE/'CRITIC_TRAINING_PLAN.json');rows=[];arrays={}
    for name,held in family_groups(alltypes).items():
        source=[t for t in alltypes if t not in held];p,pn,pf,model,record=select(source,config,data);mask=np.isin(types,held);selected=choose_sampled(p,data['candidate_indices']);ranks=metrics(p,data['labels'],types,list(held),data['candidate_indices'])
        arrays[name+'_probability']=p[:,mask];arrays[name+'_names']=data['names'][mask];arrays[name+'_types']=types[mask]
        for key,v in selected.items():arrays[name+'_'+key]=v[mask]
        rows.append(dict(held_family=name,held_types=list(held),source_types=source,model=model,source_only_selection=record,ranking=ranks,evidence='DEVELOPMENT_DESCRIPTIVE'))
    write(HERE/'LEAVE_ONE_FAMILY_MODELS.json',rows);np.savez_compressed(HERE/'LEAVE_ONE_FAMILY_RESULTS.npz',**arrays)
    seal('LEAVE_ONE_FAMILY_RESULTS.receipt.json',[HERE/'LEAVE_ONE_FAMILY_MODELS.json',HERE/'LEAVE_ONE_FAMILY_RESULTS.npz',HERE/'family_selection.py',HERE/'CRITIC_TRAINING_PLAN.receipt.json'],outer_labels_never_used_for_selection=True)
if __name__=='__main__':main()
