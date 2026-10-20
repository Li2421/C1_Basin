from common import *
from statistics_pipeline import evaluate_group,equivalence
from baselines import LEARNED,NONLEARNED


def main():
    with np.load(HERE/'TRAINING_BUNDLE.npz',allow_pickle=False) as z:data={k:z[k] for k in ('names','types','labels','uids','collision','timeout','unknown')}
    for key in ('labels','uids','collision','timeout','unknown'):data[key]=data[key][...,:1]
    data['uid_class']=equivalence(data['uids'])
    with np.load(HERE/'DEVELOPMENT_CHOICES.npz',allow_pickle=False) as z:methods={k:z[k] for k in z.files if k not in ('names','types')}
    reports={};nc=len(data['types'])
    groups={str(t):data['types']==t for t in sorted(set(data['types']))}
    groups.update(development=np.ones(nc,bool),learned=np.isin(data['types'],list(LEARNED)),nonlearned=np.isin(data['types'],list(NONLEARNED)))
    for group,mask in groups.items():
        subset={k:v[mask] for k,v in data.items()};selected={k:v[mask] for k,v in methods.items()}
        reports[group]={'type_equal':evaluate_group(subset,selected,True,5000)}
        if group in ('development','learned','nonlearned'):reports[group]['controller_equal']=evaluate_group(subset,selected,False,5000)
    all_=reports['development']['type_equal'];learn=reports['learned']['type_equal'];non=reports['nonlearned']['type_equal']
    contrast=lambda group,key:group['paired_comparisons']['pipeline__minus__'+key]
    targets=dict(nonlearned_vs_source_fixed=contrast(non,'source_fixed')['CI95_pp'][0]>0,nonlearned_vs_progress=contrast(non,'all_Sprog')['CI95_pp'][0]>0,type_equal_vs_progress=contrast(all_,'all_Sprog')['CI95_pp'][0]>0,learned_not_worse_1pp=contrast(learn,'all_Sprog')['gain_pp']>=-1,harm_zero=all_['methods']['pipeline']['harm_possible']==0,nonlearned_controller_information_point_positive=contrast(non,'no_controller')['gain_pp']>0)
    write(HERE/'DEVELOPMENT_RESULTS.json',dict(utc=utc(),evidence='DEVELOPMENT',reports=reports,targets=targets,all_targets_met=all(targets.values()),training_oracle='Common seed0 in-sample; no training crossfit claim'))
    seal('DEVELOPMENT_RESULTS.receipt.json',[HERE/'DEVELOPMENT_RESULTS.json',HERE/'development_statistics.py',HERE/'DEVELOPMENT_CHOICES.receipt.json',HERE/'STATISTICS_TESTS.json'])
    write(HERE/'DEVELOPMENT_REVIEW.json',dict(utc=utc(),freeze_current_version=True,targets=targets,all_targets_met=all(targets.values()),rule='Source-only NLL selects branch initialization/checkpoint and positive controller weight; not external held-type winner picking. Report single-branch and shrinkage versions. Freeze current best within fixed candidate and training allocation even if targets fail.',additional_label_budget_used=0))
    seal('ARCHITECTURE_FIXED.receipt.json',[HERE/'DEVELOPMENT_REVIEW.json',HERE/'critic_model.py',HERE/'select_mixed.py',HERE/'SELECTED_MODELS.json'],before_scaling_curve=True,before_confirmation=True)
    print(json.dumps(dict(targets=targets,all_targets_met=all(targets.values()),development_main=all_['methods']['pipeline']['success_lower_pct'])))


if __name__=='__main__':main()
