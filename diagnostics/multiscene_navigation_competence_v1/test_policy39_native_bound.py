from common import *
from policies39 import bound_and_stop
from types import SimpleNamespace
rng=np.random.default_rng(17);angles=rng.uniform(-np.pi,np.pi,10000)
raw=np.column_stack([np.cos(angles),np.sin(angles)]).astype(np.float32)*np.float32(.5)
env=SimpleNamespace(config=SimpleNamespace(max_speed=.5,goal_tolerance=.08),positions=np.zeros((len(raw),2)),goals=np.ones((len(raw),2)))
bounded=bound_and_stop(raw,env);assert np.linalg.norm(bounded,axis=1).max()<=.5+1e-10
assert np.allclose(bounded,raw,atol=5e-8,rtol=0)
env.goals[:3]=0;assert np.array_equal(bound_and_stop(raw,env)[:3],np.zeros((3,2)))
write(HERE/'POLICY39_NATIVE_BOUND_TEST.json',dict(passed=True,velocity_count=len(raw),maximum_raw_float64_speed=float(np.linalg.norm(raw.astype(np.float64),axis=1).max()),maximum_bounded_float64_speed=float(np.linalg.norm(bounded,axis=1).max()),goal_stop_preserved=True,rule='float64 radial bound then original goal stop; no plant change'))
print('native float64 action bound passed')
