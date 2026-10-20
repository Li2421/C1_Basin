"""Small analytical checks of paired resampling and privileged oracles."""
from core import *
from statistics_pipeline import *


def main():
    # The winning seed0 action fails on seed1: crossfit must score zero.
    y=np.zeros((1,2,3,14,2),int)
    y[...,0,0]=1;y[...,1,1]=1;y[...,13,:]=1
    o=oracle_choices(y)
    assert np.all(o['state_crossfit_oracle'][...,0]==1)
    assert np.all(o['state_crossfit_oracle'][...,1]==0)
    assert gather(y,o['state_crossfit_oracle']).mean()==0
    assert gather(y,o['state_in_sample_oracle']).mean()==.5
    assert np.all(o['state_in_sample_oracle']<13),'Nominal is not a candidate'
    assert 'state_crossfit_oracle' not in oracle_choices(y[...,:1])
    # On tied score, progress is preferred; controller oracle reselects after states change.
    assert np.all(best(np.zeros((2,13)))==12)
    a=np.zeros((1,1,2,13,2));a[:,:,0,0]=1;a[:,:,1,1]=1
    assert np.all(oracle_choices(a[:,:,1:])['controller_oracle']==1)
    types=np.asarray(['A','B','B']);cc,ss,w=point_indices(types,2,True)
    assert np.allclose(w,[[.5,.25,.25]])
    cc,ss,w=sample_indices(types,2,np.random.default_rng(3),200,True)
    assert np.allclose(w.sum(1),1) and cc.shape==w.shape
    assert ss.shape==(*cc.shape,2) and np.any(w==0)
    # Unknown labels for two actions sharing the same rollout cancel exactly.
    labels=np.zeros((3,2,14,2),int);labels[:,:,0]=-1;labels[:,:,12]=-1
    uids=np.broadcast_to(np.arange(14)[None,None,:,None],labels.shape).copy()
    uids[:,:,0]=12
    data=dict(labels=labels,types=types,names=np.asarray(['a','b','c']),uids=uids,
        collision=np.zeros_like(labels),timeout=np.zeros_like(labels),unknown=labels<0)
    fixed=dict(all_Sprog=np.full((3,2),12),all_safety_baseline=np.full((3,2),10),pipeline=np.zeros((3,2),int))
    report=evaluate(data,fixed,replicates=64)['combined']['type_equal']
    paired=report['paired_comparisons']['pipeline__minus__all_Sprog']
    assert paired['unknown_bounds_pp']==[0.,0.] and paired['CI95_pp']==[0.,0.]
    # Single common seed explicitly leaves crossfit regret unavailable.
    single={k:(v[...,:1] if k in ('labels','uids','collision','timeout','unknown') else v) for k,v in data.items()}
    r=evaluate(single,fixed,replicates=32)['combined']['type_equal']
    assert not r['state_crossfit_oracle_available']
    assert r['methods']['pipeline']['state_crossfit_regret_pp'] is None
    labels[0,0,0,0]=-2
    try:evaluate(data,fixed,replicates=2)
    except AssertionError:pass
    else:raise AssertionError('Missing labels must not become unknown')
    write(HERE/'STATISTICS_TESTS.json',dict(utc=utc(),passed=True,checks=[
        'opposite-seed selection and scoring','nominal excluded from oracle','progress tie rule',
        'controller oracle reselected on sampled states','type-equal weights and padding',
        'same-UID unknown cancellation','single-seed crossfit unavailable','missing label rejection']))
    seal('STATISTICS_TESTS.receipt.json',[HERE/'STATISTICS_TESTS.json',HERE/'test_statistics_pipeline.py',HERE/'statistics_pipeline.py'])
    print('Paired-statistics analytical checks passed',flush=True)


if __name__=='__main__':main()
