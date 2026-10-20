from common import *
from legacy_controllers import Nominal,M_GRID
from right_hand import RightHand
from historical_controllers import Observed
import jax
p=read(HERE/'gate_protocol.json');s=p['states'][0];checks=[]
for i,params in enumerate(M_GRID):
 rt=Nominal('M',params,p['native_manifest']);env=rt.make_env(s);a=rt.nominal(env,jax.random.PRNGKey(380))[1];b=rt.nominal(env,jax.random.PRNGKey(380))[1];assert a.shape==(4,2) and a.tobytes()==b.tobytes() and np.isfinite(a).all() and np.linalg.norm(a,axis=-1).max()<=env.config.max_speed+1e-12;checks.append('centralized_sampling'+str(i))
 # An inactive continuing rule preserves the exact centralized baseline action.
 observed=Observed(rt);rhr=RightHand(observed,dict(mode='progress',steps=1000000,speed=.35));key=jax.random.PRNGKey(380);x=observed.action(env,key,np.zeros(3),'old')[0];y=rhr.action(env,key,np.zeros(3),'old')[0];assert x.tobytes()==y.tobytes()
write(HERE/'LEGACY_AND_RIGHT_HAND_UNIT_TESTS.json',dict(utc=utc(),passed=True,controllers=checks,key_shape_speed=True,inactive_rule_first_action_bytes=True,physical_steps=0))
print('six legacy instances and inactive continuing rule PASS')
