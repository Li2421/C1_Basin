from core import *
from controllers import Nominal,M_GRID,P_GRID
import jax
jax.config.update('jax_enable_x64',False)
p=read(HERE/'base_protocol.json');s=read(HERE/'PILOT_STATES.json')[0];count=0
for kind,grid in [('M',M_GRID),('P',P_GRID)]:
 for params in grid:
  rt=Nominal(kind,params,p['native_manifest']);assert rt.config.to_dict()==p['native_manifest']['scenario_config'];env=rt.make_env(s);_,a,_=rt.nominal(env,jax.random.PRNGKey(7));assert a.shape==(4,2) and np.isfinite(a).all() and np.max(np.linalg.norm(a,axis=-1))<=rt.config.max_speed+1e-14;count+=1
write(HERE/'NATIVE_CONSTRUCTOR_TESTS.json',dict(utc=utc(),status='PASS',controllers=count,config_matches_immutable_native_manifest=True,no_simulator_steps=True,controller_code_sha256=sha(HERE/'controllers.py'),jax_enable_x64=bool(jax.config.x64_enabled)));print('Native constructor and actual-config nominal tests PASS')
