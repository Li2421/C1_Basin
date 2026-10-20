"""Statistics invariants: paired identity, original parity, hierarchical weighting."""
import json,datetime
from pathlib import Path
import numpy as np
from analysis_statistics import evaluate,evaluate_group,oracle_choices,sample_indices
from selector import BASELINE

def data(labels,types):
    n,s,_,q=labels.shape
    return dict(labels=labels,types=np.array(types),nominal=np.zeros((n,s,q),int),
        **{e:np.zeros_like(labels) for e in ('collision','timeout','unknown')},
        **{'nominal_'+e:np.zeros((n,s,q),int) for e in ('collision','timeout','unknown')})

def main():
    labels=np.zeros((3,8,8,2),int);labels[:2,:,BASELINE]=1
    d=data(labels,['a','a','b']);choices={'pipeline':np.full(3,BASELINE)}
    te=evaluate_group(d,choices,True,200);ce=evaluate_group(d,choices,False,200)
    assert te['methods']['pipeline']['success_lower_pct']==50
    assert abs(ce['methods']['pipeline']['success_lower_pct']-100*2/3)<1e-10
    assert te['methods']['pipeline']['gain_CI95_pp']==[0.,0.]
    d['labels'][:]=-1
    for equal in (True,False):
        report=evaluate_group(d,choices,equal,200)['methods']
        assert report['pipeline']['gain_CI95_pp']==[0.,0.]
        assert report['crossfit_oracle']['crossfit_regret_CI95_pp']==[0.,0.]
        assert report['in_sample_oracle']['in_sample_regret_CI95_pp']==[0.,0.]
        assert report['pipeline']['success_CI95_pct']==[0.,100.]
    lower=np.zeros((1,1,6,8,2));ids=np.array([[[0,0,2,1,3,3]]])
    lower[0,0,:,1,:]=(ids[0,0]%2==0)[:,None]
    lower[0,0,:,2,:]=(ids[0,0]%2==1)[:,None]
    oracle=oracle_choices(lower,ids,np.array([['a']]),np.ones((1,1)))['crossfit_oracle']
    assert np.array_equal(oracle,np.where(ids%2==0,2,1))
    # Padding slots have zero statistical weight and cannot change type oracle.
    lower=np.zeros((1,3,2,8,2));lower[0,0,:,1]=1;lower[0,1:,:,2]=1
    chosen=oracle_choices(lower,np.zeros((1,3,2),int),np.array([['a','a','a']]),np.array([[1.,0.,0.]]))
    assert np.all(chosen['type_best_fixed_oracle']==1)
    from scale_complete import interval,paired_gain
    all_unknown=data(np.full((3,8,8,2),-1,int),['a','a','b'])
    shared=interval(all_unknown,np.full((3,20),BASELINE),replicates=200)
    distinct=interval(all_unknown,np.full((3,20),1),replicates=200)
    assert shared['CI95_pp']==[0.,0.] and shared['regret_pp']==0
    assert np.allclose(distinct['CI95_pp'],[-100.,100.],rtol=0,atol=1e-10)
    a=np.tile([1,BASELINE,-1],(3,1));b=np.tile([BASELINE,-1,1],(3,1))
    mixed=paired_gain(all_unknown,a,b,replicates=200)
    assert mixed['CI95_pp']==[0.,0.] and mixed['gain_pp']==0
    ragged=interval(all_unknown,np.tile([BASELINE,-1,BASELINE],(3,1)),replicates=200)
    assert ragged['CI95_pp']==[0.,0.]
    available=evaluate(data(labels,['a','a','b']),{'partial':np.array([BASELINE,BASELINE,-1])},replicates=200)
    assert 'partial' in available['a']['type_equal']['methods']
    assert 'partial' not in available['b']['type_equal']['methods']
    assert 'partial' in available['combined']['type_equal']['unavailable_methods']
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,
        synthetic_only=True,hierarchical_type_weighting=True,unknown_identity_cancels=True,
        original_state_parity=True,padding_cannot_affect_type_oracle=True,oracle_bootstrap_reselection=True,
        scale_mixture_shared_unknown_cancellation=True,ragged_matched_controls=True,
        identical_arm_mixtures_cancel_unknown=True,unsupported_arms_never_substituted=True)
    (Path(__file__).parent/'STATISTICS_UNIT_TESTS.json').write_text(json.dumps(out,indent=2)+'\n')
    print(out)

if __name__=='__main__':main()
