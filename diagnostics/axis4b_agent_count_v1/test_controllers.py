from core import *
from controllers import potential,nominal_reference,corrected_action,P2_GAINS
from types import SimpleNamespace
# No native physical steps or learned-policy calls in these unit tests.
for n in [2,10,20,50,100]:
 p=np.column_stack([np.linspace(-2,2,n),np.sin(np.arange(n))]);g=-p;walls=np.array([[[-5.,-4.],[5.,-4.]],[[-5.,4.],[5.,4.]]]);env=SimpleNamespace(positions=p,goals=g,walls=walls,instance=SimpleNamespace(geometry=SimpleNamespace(margin=.17166666666666666)),config=SimpleNamespace(max_speed=.5,goal_tolerance=.08));a=potential(env);assert a.shape==(n,2) and np.isfinite(a).all() and np.max(np.linalg.norm(a,axis=-1))<=.5+1e-14
 perm=np.arange(n)[::-1];other=SimpleNamespace(**env.__dict__);other.positions=p[perm];other.goals=g[perm];np.testing.assert_allclose(potential(other),a[perm],rtol=1e-13,atol=1e-13)
 shifted=SimpleNamespace(**env.__dict__);shifted.positions=p+np.array([3.,5.]);shifted.goals=g+np.array([3.,5.]);shifted.walls=walls+np.array([3.,5.]);np.testing.assert_allclose(potential(shifted),a,rtol=1e-12,atol=1e-12)
 assert np.array_equal(nominal_reference(a,SimpleNamespace(positions=p,goals=p,config=env.config)),np.zeros_like(a))
# Verify TT order and that eta enters the basis rather than a post-plant edit.
calls=[];env=SimpleNamespace(positions=np.zeros((2,2)),goals=np.ones((2,2)),config=SimpleNamespace(max_speed=.5),snapshot=lambda:dict(original_native_snapshot=True));ref=np.ones((2,2))*.1
class Projector:
 def __call__(self,snapshot,target):calls.append(np.asarray(target).copy());return SimpleNamespace(velocity=np.asarray(target)*.5)
class Basis:
 def compute(self,p,g,safe,speed):assert np.array_equal(safe,ref*.5);return SimpleNamespace(correction=lambda eta:np.ones_like(safe)*eta[0])
a,safe,delta=corrected_action(env,ref,[.02,0,0],Projector(),Basis());assert len(calls)==2 and np.array_equal(calls[0],ref) and np.array_equal(calls[1],safe+delta) and np.array_equal(a,(safe+delta)*.5);assert P2_GAINS==(1.,.015,.01)
write(HERE/'UNIT_TESTS.json',dict(utc=utc(),passed=True,counts=[2,10,20,50,100],finite_bounded=True,permutation_equivariant=True,translation_equivariant=True,goal_stop=True,TT_two_projection_order=True,native_simulation_steps=0));print('Generic P2 and TT composition tests PASS')
