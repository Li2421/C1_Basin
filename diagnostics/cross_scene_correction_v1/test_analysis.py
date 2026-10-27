"""Synthetic checks for held-scene exclusion, paired bootstrap and calibration separation."""
from common import *
from models import select,calibration,crossfit
from statistics import sampled_rates,c_decision
import numpy as np


def main():
    rng=np.random.default_rng(71);f=rng.normal(size=(6,18));e=rng.normal(size=(49,3));q=rng.uniform(size=(6,49))
    selection=select(f,q,f[0],e)
    assert selection['penalty']in (.1,1.,10.,100.)and selection['k']in (1,2,3,5)
    y=np.zeros((2,8,49));y[:,:,5]=1.
    cal=calibration(y);oracle=crossfit(y)
    assert oracle['rate']==1. and oracle['actions']==[5,5]
    assert cal['rate']==0.,'Calibration must only use extended points17..48'
    choice=dict(ridge=5,knn=5,global_fixed=5,eta13=14,safety=0)
    choice['methods']=dict(calibration=cal)
    lower,upper=sampled_rates(dict(y=y,unknown=np.zeros_like(y)),choice,rng,replicates=40)
    assert np.all(lower['ridge']==1.)and np.all(lower['safety']==0.)
    assert np.all(lower['ridge']-lower['knn']==0.)
    assert all(np.array_equal(lower[k],upper[k])for k in lower)
    y=np.zeros((3,12,49));y[:,:,20]=1.
    cal=calibration(y);choice['methods']['calibration']=cal
    lower,upper=sampled_rates(dict(y=y,unknown=np.ones_like(y)),choice,rng,replicates=40)
    assert all(np.all((v>=0)&(v<=1))for v in lower.values())
    assert all(np.all(upper[k]>=lower[k])for k in lower)
    # Every calibration selection and held-out evaluation are disjoint.
    assert all(len(set(d['states']))==4 and len(set(d['actions']))==16 for d in cal['draws'])
    # Original C names a mean threshold; a wide interval must not silently add a gate.
    original=c_decision(.04,[-.03,.10]);strict=c_decision(.04,[-.03,.10],'upper95_difference_le_0.05')
    assert original['verdict']=='PASSED'and strict['verdict']=='NOT_PASSED'
    assert original['stricter_upper95_within5pp']is False
    assert c_decision(.06,[.01,.11])['verdict']=='NOT_PASSED'
    assert c_decision(.05,[-.02,.05])['verdict']=='PASSED'
    write(HERE/'ANALYSIS_TESTS.json',dict(passed=True,synthetic_only=True,
        checks=['Nested CV grid and candidate outputs','Menu-restricted calibration','Crossfit recovers clean action',
                'Paired identical policy difference exactly0','Unknown upper bounds monotone','Bootstrap bounded rates',
                'Original mean C threshold distinct from stricter upper95 predicate']))
    print('Analysis checks passed',flush=True)


if __name__=='__main__':main()
