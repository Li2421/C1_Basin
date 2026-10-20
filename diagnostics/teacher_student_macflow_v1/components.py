from core import *
from audit_utils import source_inventory
from data import teacher
from actions import *
from diagnostics.orthoflow3_tt_ff_matched_learnability_v1.execution import key_for
import jax
jax.config.update('jax_enable_x64',False)
def main():
    rt=teacher(0);s=read(HERE/'EVAL_STATES.json')[0];env=rt.make_env(s);key=key_for(s,0,0);eta=np.zeros(3);counts=np.zeros(4,int)
    a,d=right_hand_local(rt,env,key,eta,counts);b,_=decentralized_action(rt,env,key,eta,'local',counts);assert a.tobytes()==b.tobytes();assert not d['rhr_active'].any()
    # Safe, widely separated low-speed positions make the target perturbation directly observable.
    env.reset(np.array([[2.,0.],[0.,2.],[-2.,0.],[0.,-2.]]),velocities=np.zeros((4,2)),goals=np.array([[-2.,0.],[0.,-2.],[2.,0.],[0.,2.]]))
    class Zero:
        def nominal(self,env,key):return None,np.zeros((4,2)),None
    z=Zero();inactive,_=right_hand_local(z,env,key,eta,np.full(4,39));assert not inactive.any()
    active,d=right_hand_local(z,env,key,eta,np.full(4,40));expected=.15*np.stack((env.positions[:,1],-env.positions[:,0]),axis=1)/np.linalg.norm(env.positions,axis=1,keepdims=True);np.testing.assert_array_equal(active,expected);assert d['rhr_active'].all()
    env.goals[0]=env.positions[0];a,d=right_hand_local(z,env,key,eta,np.full(4,40));assert not d['rhr_active'][0] and np.all(a[0]==0)
    n,_=nominal_only(rt,env,key,eta,counts);assert n.tobytes()==rt.nominal(env,key)[1].tobytes()
    write(HERE/'COMPONENT_TESTS.json',dict(utc=utc(),status='PASS',rhr_inactive_equals_safety_bitwise=True,rhr_trigger_39_off_40_on=True,rhr_amplitude_direction_exact=True,at_goal_no_trigger=True,nominal_only_equals_nominal_bitwise=True,student_nominal_specific_test='pending checkpoint tests',simulation_steps=0,jax_enable_x64=bool(jax.config.x64_enabled)));source_inventory('imports_components.json')
if __name__=='__main__':main()
