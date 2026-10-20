from core import *
from controls import Control
from fair_rhr import Trigger,FairRuntime,grid
from poc.field_controller import Runtime
from pipeline import bounded_rows
from types import SimpleNamespace
from audit_utils import source_inventory
import jax
jax.config.update('jax_enable_x64',False)
def main():
    class FakeControl(Control):
        def observation(self,env):return None
    checks={};cfg=SimpleNamespace(max_speed=.52,goal_tolerance=.1);p=np.array([[2.,0],[0,2],[-2,0],[0,-2]]);env=SimpleNamespace(positions=p.copy(),goals=-p.copy(),velocities=np.zeros((4,2)),config=cfg,step_count=0)
    for kind in ['Z','G']:
        # Exercise actual nominal method without allocating irrelevant teacher weights.
        rt=FakeControl.__new__(FakeControl);rt.kind=kind
        a=rt.nominal(env,jax.random.PRNGKey(0))[1];b=rt.nominal(env,jax.random.PRNGKey(1))[1];assert np.array_equal(a,b);assert np.linalg.norm(a,axis=1).max()<=.52+1e-15
        expected=np.zeros_like(p) if kind=='Z' else bounded_rows(env.goals-env.positions,.52);np.testing.assert_array_equal(a,expected)
        changed=SimpleNamespace(**env.__dict__);changed.goals=env.goals.copy();changed.goals[1]+=[4,2];np.testing.assert_array_equal(rt.nominal(changed,None)[1][[0,2,3]],a[[0,2,3]])
        checks[kind]='shape,speed,seed determinism,own-goal isolation PASS'
    assert Control.action is Runtime.action and Control.prepare is Runtime.prepare;checks['central_TT']='inherited original Runtime.action and prepare; nominal only override'
    assert len(grid())==20
    for mode in ['always','slow','progress']:
        for w in ([0] if mode=='always' else [20,40]):
            tr=Trigger(dict(mode=mode,steps=w,speed=.1));env.positions=p.copy();env.goals=-p;env.velocities=np.zeros((4,2));env.step_count=0
            for t in range(max(w,1)+1):
                env.step_count=t;active=tr.update(env);assert bool(active.all())==(mode=='always' or t>=w);np.testing.assert_array_equal(tr.update(env),active)
            if mode=='slow':env.step_count+=1;env.velocities[:]=[.05,0];assert not tr.update(env).any()
            if mode=='progress':env.step_count+=1;env.positions=p*.95;assert not tr.update(env).any()
            env.step_count=0;env.positions=p.copy();assert tr.update(env).all()==(mode=='always')
    checks['triggers']='20/40 warmup boundary,strict speed recovery,rolling progress recovery,repeated-step idempotence,t0 reset PASS'
    tr=Trigger(dict(mode='progress',steps=20,speed=.1));env.positions=p.copy();env.velocities[:]=0
    for t in range(22):env.step_count=t;env.positions=p*(1-.001*t);a=tr.update(env)
    assert a.all();checks['progress']='continual motion with insufficient progress triggers, unlike old .01-speed gate'
    # Full local safety stack at real physical state, with all speeds and trigger modes.
    control=Control('Z');real=control.make_env(read(HERE/'EVAL_STATES.json')[0]);key=jax.random.PRNGKey(0)
    for v in [.1,.15,.25,.35]:
        rt=FairRuntime(control,dict(mode='always',steps=0,speed=v));a,cert,calls=rt.action(real,key,np.zeros(3));assert a.shape==(4,2) and np.isfinite(a).all();assert np.linalg.norm(a,axis=1).max()<=real.config.max_speed+1e-7;assert cert['min_linear_residual']>=-control.cbf.feasibility_tol;assert calls>=8
    q=np.array([2.,0]);tangent=np.array([q[1],-q[0]])/np.linalg.norm(q);np.testing.assert_array_equal(tangent,[0.,-1.]);checks['fair_action']='actual unchanged local CBF twice,all4speeds,clockwise tangent,no environment stepping PASS'
    C=D/'teacher_student_macflow_v1c';assert all(sha(HERE/f)==sha(C/f) for f in ['pipeline.py','episode_frozen.py','student.py','core.py']);checks['reuse']='core,student,pipeline,episode byte-identical round30c; checkpoints identical; prior replay gate reused';checks['precision']=bool(jax.config.x64_enabled);assert not jax.config.x64_enabled
    write(HERE/'COMPONENT_TESTS.json',dict(utc=utc(),status='PASS',checks=checks,simulation_steps=0));source_inventory('imports_tests.json')
if __name__=='__main__':main()
