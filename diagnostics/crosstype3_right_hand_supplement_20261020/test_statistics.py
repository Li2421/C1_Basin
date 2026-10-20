"""Sparse-oracle, paired missingness and strongest-human-baseline analytics."""
from common import *
from statistics_pipeline import *
def main():
    labels=np.asarray([[[[1,0],[0,1],[0,0],[1,1],[-2,-2]]]],np.int8);uids=np.asarray([[[['a0','a1'],['b0','b1'],['s0','s1'],['n0','n1'],['','']]]]);mask=np.asarray([[[True,True,True,False,False]]]);data=dict(labels=labels,uids=uids,uid_class=equivalence(uids),types=np.asarray(['type']),names=np.asarray(['C']),oracle_mask=mask,non_diagonal=np.asarray([[[True,True,False,False,False]]]),collision=np.zeros_like(labels),timeout=(labels==0).astype(np.int8),unknown=(labels==-1).astype(np.int8));methods={k:np.asarray([[v]],np.int32) for k,v in dict(pipeline=0,self_schedule=0,fixed500=1,no_schedule=0,all_safety_baseline=2,all_S500=0,all_Sprog=1,nominal_only=3).items()}
    o=oracle_choices(labels[None],mask[None]);assert np.array_equal(o['evaluated_state_crossfit_oracle'],np.asarray([[[[1,0]]]]));report=evaluate_group(data,methods,True,32);assert report['methods']['evaluated_state_in_sample_oracle']['success_lower_pct']==50 and report['methods']['evaluated_state_crossfit_oracle']['success_lower_pct']==0;assert report['methods']['pipeline']['evaluated_action_crossfit_regret_pp']==-50;assert report['paired_comparisons']['pipeline__minus__best_human_schedule']['CI95_pp']==[0.,0.]
    unknown=np.asarray([[[[-1,-1],[-1,-1],[0,0]]]],np.int8);uids2=np.asarray([[[['x0','x1'],['x0','x1'],['z0','z1']]]]);d=dict(data,labels=unknown,uids=uids2,uid_class=equivalence(uids2),oracle_mask=np.ones((1,1,3),bool),non_diagonal=np.zeros((1,1,3),bool),collision=np.zeros_like(unknown),timeout=(unknown==0).astype(np.int8),unknown=(unknown==-1).astype(np.int8));m={k:np.asarray([[v]],np.int32) for k,v in dict(pipeline=0,self_schedule=0,fixed500=1,all_safety_baseline=1,all_S500=1,all_Sprog=2).items()};r=evaluate_group(d,m,True,32);assert r['paired_comparisons']['pipeline__minus__all_safety_baseline']['unknown_bounds_pp']==[0.,0.];assert r['paired_comparisons']['pipeline__minus__best_human_schedule']['unknown_bounds_pp']==[0.,0.]
    types=np.asarray(['a','a','b','b','b','c'])
    cc,ss,w=sample_indices(types,4,np.random.default_rng(7),500,True,True)
    for t in np.unique(types):assert np.allclose(np.sum(w*(types[cc]==t),1),1/3)
    assert ss.shape==(*cc.shape,4)
    write(HERE/'STATISTICS_TESTS.json',dict(utc=utc(),passed=True,sparse_padding_excluded=True,nominal_excluded_from_oracle=True,opposite_seed_oracle_analytic=True,negative_crossfit_regret_retained=True,paired_same_uid_unknown_cancellation=True,strongest_human_schedule_reselected=True,two_seeds_clustered=True))
    seal('STATISTICS_TESTS.receipt.json',[HERE/'STATISTICS_TESTS.json',HERE/'test_statistics.py',HERE/'statistics_pipeline.py'])
    print('Sparse-menu statistics analytic tests passed',flush=True)
if __name__=='__main__':main()
