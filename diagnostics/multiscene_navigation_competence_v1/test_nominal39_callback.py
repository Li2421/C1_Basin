"""Unit inference only: new nominal callback, unchanged native correction-chain entry."""
from common import *
from geometry import GeneratedGeometry,GeneratedEnv
from bottleneck_family.scenario import Config
from policies39 import Policy,load_policy as load_native
from nominal39_callback import load_policy,observed39
from observation39 import observe
from generated_runtime import action,clockwise_goal
from controllers import nominal_reference,bound
from flax import serialization
import jax,jax.numpy as jnp,tempfile

def main():
    geo=GeneratedGeometry(Config(num_agents=4),[(-.3,-4.,.3,-.7),(-.3,.7,.3,4.)])
    positions=np.array([[-6.,-1.],[6.,-1.],[-6.,1.],[6.,1.]])
    goals=positions*np.array([-1.,1.]);assert geo.valid_points(positions).all() and geo.valid_points(goals).all()
    assert all(geo.path(p,g) is not None for p,g in zip(positions,goals))
    case=dict(state_uid='UNIT_ONLY_no_rollout_identity',seed=0,arm='nominal_only')
    root=jax.random.fold_in(jax.random.PRNGKey(2026101739),int(hashlib.sha256(case['state_uid'].encode()).hexdigest()[:8],16));key=jax.random.fold_in(jax.random.fold_in(root,0),0)
    checks=[]
    with tempfile.TemporaryDirectory(dir=HERE/'tmp') as folder:
        for kind in ['deterministic','mixture']:
            p=Path(folder)/kind;p.mkdir();net=Policy(kind);params=net.init(jax.random.PRNGKey(17),jnp.zeros((1,39),jnp.float32));(p/'best.msgpack').write_bytes(serialization.to_bytes(params));write(p/'CONFIG.json',dict(family=kind,observation_dimensions=39));write(p/'NORMALIZATION.json',dict(obs_mean=[0]*39,obs_scale=[1]*39,act_mean=[0]*2,act_scale=[1]*2))
            env=GeneratedEnv(geo,positions,goals);policy=load_policy(p/'best.msgpack');direct=load_native(p/'best.msgpack')
            callback=policy(env,key);expected=direct(env,env.round40_navigation_fields,key)
            expected_observed=observe(env,env.round40_navigation_fields)
            assert expected_observed.tobytes()==observed39(env).tobytes() and np.any(np.abs(expected_observed[:,20:35])>0)
            assert callback.shape==(4,2) and callback.tobytes()==expected.tobytes()
            actual=action(env,policy,case,0);native=nominal_reference(bound(expected,env.config.max_speed),env)
            assert actual.tobytes()==native.tobytes() and np.max(np.linalg.norm(actual,axis=1))<=.5+1e-10
            fields=env.round40_navigation_fields;old=observe(env,fields)[0,-4:].copy();new_positions=positions.copy();new_positions[2,1]+=.2;env.reset(new_positions,goals);policy(env,key)
            assert env.round40_navigation_fields is fields and observe(env,fields)[0,-4:].tobytes()==old.tobytes()
            assert observe(env,fields).tobytes()==observed39(env).tobytes()
            near=goals.copy();near[0]=positions[0]+[.04,0.];near[1]=[-6.,0.];assert geo.valid_points(near).all();assert all(geo.path(p,g) is not None for p,g in zip(positions,near))
            env=GeneratedEnv(geo,positions,near);stopped=policy(env,key);assert np.array_equal(stopped[0],np.zeros(2))
            checks.append(dict(kind=kind,N=4,observations39_bitwise_equal=True,actual_neighbor_features_preserved=True,callback_bitwise_equal=True,native_nominal_entry_bitwise_equal=True,goal_stop=True,world_speed_bound=True,field_independent_of_other_robot_positions=True,no_physical_steps=True))
    p=np.zeros((4,2));g=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,0.]])
    expected=np.array([[0.,-1.],[1.,0.],[0.,1.],[0.,0.]])
    assert np.array_equal(clockwise_goal(p,g),expected)
    assert np.array_equal(clockwise_goal(p+[7.,-3.],g+[7.,-3.]),expected)
    source=ROOT/'diagnostics/blackbox_addendum_v1/trigger.py';assert sha(source)==sha(HERE/'trigger.py')
    assert not jax.config.jax_enable_x64
    write(HERE/'NOMINAL39_CALLBACK_UNIT_TEST.json',dict(passed=True,checks=checks,callback_sha256=sha(HERE/'nominal39_callback.py'),test_code_sha256=sha(Path(__file__)),generated_runtime_sha256=sha(HERE/'generated_runtime.py'),clockwise_goal_world_coordinates=True,translation_invariant_direction=True,trigger_source=str(source),trigger_source_sha256=sha(source),trigger_byte_identical=True,x64_enabled=False,only_untrained_unit_checkpoints=True,physical_steps=0,rollouts_registered=0))
    print(checks)

if __name__=='__main__':main()
