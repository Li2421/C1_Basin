from statistics_core import *
from fewshot import beta_gt
import ast
assert abs(beta_gt(0,0,0,0)-.5)<1e-12 and abs(beta_gt(1,1,0,0)-2/3)<1e-12
for path in HERE.glob('*.py'):ast.parse(path.read_text())
lo=np.zeros((1,4,5,2));lo[:,::2,2]=1;lo[:,1::2,3]=1;l,h,_,_=oracle_bounds(lo,lo,[2,3]);assert l.sum()==h.sum()==0
# Native deadlock is retained as a separate transition, never inferred from timeout.
events={k:np.zeros(lo.shape) for k in ['collision','timeout','unknown','deadlock']};events['deadlock'][:,:,0]=1;events['timeout'][:,:,1]=1
r=summarize('synthetic',lo,lo,events,{'nominal only':0,'safety baseline':1,'F10':2},[2,3,4],[2,3,4],1);assert r['nominal_to_safety']['deadlock']['counts']['timeout']==8;assert r['nominal_to_safety']['timeout']['denominator']==0
assert np.array_equal(lo[:,:,2]-lo[:,:,3],(lo[:,:,4]-lo[:,:,3])-(lo[:,:,4]-lo[:,:,2]))
print('PASS parity/unknown/paired identity/Beta/native deadlock taxonomy')
