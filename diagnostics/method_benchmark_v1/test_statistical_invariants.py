from core import *
from summarize import boot
# Identical policies must cancel even when unknown outcomes are present.
lo=np.array([[[0.,1.],[1.,0.]],[[1.,0.],[0.,1.]]]);hi=np.ones_like(lo);weights=np.broadcast_to(np.array([1.,0.]),(2,2,2)).copy();av=np.ones((2,2),bool)
r=boot((lo,hi,weights,av,av,np.array([2,2])),123,[('A','B')],['A','B']);bounds=r[-1]['A__B'];assert np.array_equal(bounds,np.zeros((5000,2)));assert np.array_equal(r[0][:,0],r[0][:,1]);write(HERE/'STATISTICAL_INVARIANTS.json',dict(utc=utc(),status='PASS',identical_paired_policy_unknown_cancellation=True,bootstrap=5000,strict_upper_CI_criterion=True));print('Paired unknown cancellation PASS')
