from statistics_core import *
# Held-out oracle cannot use the test fold: opposite best actions by parity.
lo=np.zeros((1,4,4,2));lo[:,::2,2]=1;lo[:,1::2,3]=1
l,h,p,s=oracle_bounds(lo,lo,[2,3]);assert l.sum()==0 and h.sum()==0;assert s.tolist()==[[3,2,3,2]]
# Every unknown selection remains an envelope containing both possible outcomes.
hi=lo.copy();hi[:,:,2:]=1;l,h,_,_=oracle_bounds(lo,hi,[2,3]);assert (l<=h).all() and h.min()==1
# Bootstrap with only one original parity uses all actions conservatively.
l,h,_,_=oracle_bounds(lo,hi,[2,3],np.zeros((1,4),int));assert (l<=h).all()
# Frozen controller choices index both seeds identically.
metrics,_=evaluate(lo,hi,{'nominal only':0,'safety baseline':1,'F10':2},[2,3],[2,3]);assert metrics['F10'][0].shape==(1,4,2)
print('PASS: held-out parity, unknown bounds, empty training fold, paired choice indexing')
