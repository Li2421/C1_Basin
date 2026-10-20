from core import *
from episode_extended import CountedRuntime
from actions import nominal_only
from data import teacher
import jax
jax.config.update('jax_enable_x64',False)
rt=teacher(0);env=rt.make_env(read(HERE/'EVAL_STATES.json')[0]);env.reset(np.array([[2.,0.],[0.,2.],[-2.,0.],[0.,-2.]]),velocities=np.zeros((4,2)),goals=np.array([[-2.,0.],[0.,-2.],[2.,0.],[0.,2.]]))
class Zero:
    def nominal(self,env,key,traced=False):return None,np.zeros((4,2)),None
z=CountedRuntime(Zero());z.additional_mode='rhr_local';key=jax.random.PRNGKey(0)
for step in range(41):
    env.step_count=step;a,_,_=z.action(env,key,np.zeros(3))
    if step<40:assert not a.any()
expected=.15*np.stack((env.positions[:,1],-env.positions[:,0]),axis=1)/np.linalg.norm(env.positions,axis=1,keepdims=True);assert a.tobytes()==expected.tobytes();assert (z.slow_counts==40).all();again,_,_=z.action(env,key,np.zeros(3));assert again.tobytes()==a.tobytes() and (z.slow_counts==40).all()
env.step_count=0;reset,_,_=z.action(env,key,np.zeros(3));assert not reset.any()
z.additional_mode='nominal';a,_,n=z.action(env,key,np.zeros(3));assert a.tobytes()==Zero().nominal(env,key)[1].tobytes() and n==0
write(HERE/'WRAPPER_TESTS.json',dict(utc=utc(),status='PASS',clock_counter_equals_frozen_loop=True,step39_inactive_step40_active=True,no_increment_on_repeated_same_clock=True,episode_clock_reset=True,nominal_no_projection=True,simulation_steps=0,note='clock values assigned in isolated constructed fixture; no simulator rollout or task labels'))
