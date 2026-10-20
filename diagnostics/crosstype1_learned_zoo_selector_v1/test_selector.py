"""Leakage and paired-inference invariants, using synthetic data only."""
from pathlib import Path
import datetime
import json
import numpy as np
from selector import leave_one_type, Regression, crossfit_oracle, paired_interval, paired_bounds, BASELINE
from features import aggregate,NAMES


def main():
    rng=np.random.default_rng(2026101801)
    types=np.repeat(['a','b','c','d'],4)
    x=rng.normal(size=(16,6));y=rng.uniform(0,1,(16,8))
    x[0,0]=np.nan
    choices,folds=leave_one_type(x,y,types)
    altered=y.copy();altered[types=='a']=1-y[types=='a']
    _,other=leave_one_type(x,altered,types)
    # If outer-held labels leak into inner selection or fitting, this changes.
    assert folds[0]['model']==other[0]['model']
    assert folds[0]['inner']==other[0]['inner']
    restored=Regression.from_dict(folds[0]['model'])
    assert np.array_equal(restored.choose(x[types=='a']),choices[types=='a'])
    successes=np.zeros((2,4,8,2))
    successes[:,::2,1]=1;successes[:,1::2,2]=1
    value,oracle=crossfit_oracle(successes)
    assert not value.any() # Opposite parity selects the other arm.
    assert np.all(oracle[:,::2]==2) and np.all(oracle[:,1::2]==1)
    a=rng.normal(size=(16,8,2));ci=paired_interval(a,a,types,replicates=100)
    assert ci['CI95_pp']==[0.,0.] and ci['point_pp']==0.
    unknown_low=np.zeros((16,8,2));unknown_high=np.ones_like(unknown_low)
    ci=paired_bounds(unknown_low,unknown_high,unknown_low,unknown_high,types,
                     same_cases=np.ones((16,8,1),bool),replicates=100)
    assert ci['CI95_pp']==[0.,0.],'The same unknown rollout must cancel in a paired self-comparison'
    ci=paired_bounds(unknown_low,unknown_high,unknown_low,unknown_high,types,replicates=100)
    assert ci['CI95_pp']==[-100.,100.],'Distinct unknown rollouts retain uncertainty'
    p=np.broadcast_to(np.array([[2.,0.],[0.,2.],[-2.,0.],[0.,-2.]])[None],(301,4,2)).copy()
    trace=dict(positions=p,velocities=np.zeros_like(p),nominal_actions=np.zeros((300,4,2)),
               safe_actions=np.zeros((300,4,2)),projection_active=np.zeros(300),swept_clearance=np.ones(300),
               goals=-p[0])
    features=aggregate({a:[trace]*16 for a in ('safety_baseline','F13','F10')},
                       np.zeros((16,8,4,2)),.05,.52,.09)
    assert len(features)==len(NAMES) and features[0]==1 and features[1]==0
    assert features[-1]==0 and not features[2:].any()
    try:
        aggregate({a:[trace]*15 for a in ('safety_baseline','F13','F10')},
                  np.zeros((16,8,4,2)),.05,.52,.09)
    except ValueError:
        pass
    else:
        raise AssertionError('Incomplete probes accepted')
    result=dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                synthetic_only=True,outer_label_leakage_test=True,
                serialization_roundtrip=True,oracle_state_parity=True,
                paired_identical_zero=True,paired_unknown_identity_cancels=True,
                distinct_unknown_bounds_preserved=True,strict_freeze_feature=True,incomplete_probe_rejected=True)
    (Path(__file__).resolve().parent/'SELECTOR_UNIT_TESTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
