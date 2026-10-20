import numpy as np
from analysis_statistics import oracle,values
lo=np.zeros((2,4,8,2));hi=lo.copy();lo[0,:2,2]=1;lo[0,2:,6]=1;hi=lo.copy();a,b,p,choice=oracle(lo,hi);assert np.array_equal(a,b);assert choice.shape==(2,4);assert np.all(choice[1]==2)
# All-unknown menu yields honest [0,1], rather than a known failure.
lo=np.zeros((1,4,8,2));hi=np.ones_like(lo);a,b,_,_=oracle(lo,hi);assert not a.any() and b.all()
print('cross-fit oracle and unknown envelope tests PASS')
