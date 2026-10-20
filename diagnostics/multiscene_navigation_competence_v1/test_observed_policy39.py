from common import *
from policies39 import Policy,load_policy
from flax import serialization
from geometry import GeneratedGeometry
from bottleneck_family.scenario import Config
from single_agent import SingleAgentGeneratedEnv
from navigation_field import NavigationGrid
from batch_fields39 import IndependentFields
import jax,jax.numpy as jnp,tempfile
geo=GeneratedGeometry(Config(num_agents=4),[(-.3,-4,.3,-.3),(-.3,.3,.3,4)]);grid=NavigationGrid(geo)
envs=[SingleAgentGeneratedEnv(geo,[[-2.,0.]],[[1.,.8]]),SingleAgentGeneratedEnv(geo,[[-.5,1.]],[[1.,1.]])];fields=[grid.field(e.goals[0]) for e in envs];batch=IndependentFields(fields);observed=batch.observed(envs,np.arange(len(envs)));checks=[]
with tempfile.TemporaryDirectory(dir=HERE/'tmp') as folder:
 for kind in ['deterministic','mixture']:
  p=Path(folder)/kind;p.mkdir();net=Policy(kind);params=net.init(jax.random.PRNGKey(17),jnp.zeros((1,39),jnp.float32));(p/'best.msgpack').write_bytes(serialization.to_bytes(params));(p/'CONFIG.json').write_text(json.dumps(dict(family=kind,observation_dimensions=39)));(p/'NORMALIZATION.json').write_text(json.dumps(dict(obs_mean=[0]*39,obs_scale=[1]*39,act_mean=[0]*2,act_scale=[1]*2)))
  policy=load_policy(p/'best.msgpack')
  for i,env in enumerate(envs):
   key=jax.random.PRNGKey(17+i);a=policy(env,[fields[i]],key);b=policy.from_observation(env,observed[i:i+1],key);assert a.tobytes()==b.tobytes()
   checks.append(dict(kind=kind,case=i,actions_bitwise=True))
write(HERE/'OBSERVED_POLICY39_TEST.json',dict(passed=True,checks=checks,untrained_unit_checkpoints_only=True));print(checks)
