from core import *
from student import Student
from actions import right_hand_local
from tuned import right_hand_tuned,TunedRuntime
from episode_extended import CountedRuntime
from audit_utils import source_inventory
import jax
jax.config.update('jax_enable_x64',False)
def main():
    rt=Student(1);env=rt.make_env(read(HERE/'EVAL_STATES.json')[0]);key=jax.random.PRNGKey(12)
    for counts in [np.full(4,0),np.full(4,39),np.full(4,40),np.full(4,80)]:
        a,d=right_hand_local(rt,env,key,np.zeros(3),counts);b,e=right_hand_tuned(rt,env,key,np.zeros(3),counts,.15,40);assert a.tobytes()==b.tobytes();assert d['rhr_active'].tobytes()==e['rhr_active'].tobytes()
    a=CountedRuntime(rt);b=TunedRuntime(rt,.15,40);a.additional_mode='rhr_local';b.additional_mode='rhr_local'
    for step in range(85):
        env.step_count=step;env.velocities[:]=0;u,_,_=a.action(env,key,np.zeros(3));v,_,_=b.action(env,key,np.zeros(3));assert u.tobytes()==v.tobytes();assert a.slow_counts.tobytes()==b.slow_counts.tobytes()
    env.reset(np.array([[2.,0.],[0.,2.],[-2.,0.],[0.,-2.]]),velocities=np.zeros((4,2)),goals=np.array([[-2.,0.],[0.,-2.],[2.,0.],[0.,2.]]))
    class Zero:
        def nominal(self,env,key):return None,np.zeros((4,2)),None
    for speed in [.10,.15,.25,.35]:
        for trigger in [20,40,80]:
            u,d=right_hand_tuned(Zero(),env,key,np.zeros(3),np.full(4,trigger-1),speed,trigger);assert not u.any() and not d['rhr_active'].any();u,d=right_hand_tuned(Zero(),env,key,np.zeros(3),np.full(4,trigger),speed,trigger);expected=speed*np.stack((env.positions[:,1],-env.positions[:,0]),1)/np.linalg.norm(env.positions,axis=-1,keepdims=True);np.testing.assert_array_equal(u,expected);assert d['rhr_active'].all()
    write(HERE/'COMPONENT_TESTS.json',dict(status='PASS',r28_grid_point_actions_bitwise=True,r28_counter_sequence_85_steps_bitwise=True,all12_trigger_amplitudes_exact=True,simulation_steps=0,jax_enable_x64=bool(jax.config.x64_enabled)));seal('IMPLEMENTATION_FROZEN.receipt.json',[HERE/f for f in ['student.py','pipeline.py','episode_frozen.py','episode_extended.py','tuned.py','actions.py','run.py','planning.py','tests.py','COMPONENT_TESTS.json']],before_new_closed_loop=True);source_inventory('imports_tests.json')
if __name__=='__main__':main()
