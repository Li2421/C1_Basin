from common import *
from controllers import distributed_shooting as original
from mpc_general_v2 import distributed_shooting
from ring_exchange.environment import RingExchangeEnv,LocalFrameConfig,Config
from audit_utils import source_inventory
import jax
jax.config.update('jax_enable_x64',False)
def main():
    states=read(R33/'EVAL_STATES.json');cfgdata=read(R33/'main_protocol.json')['native_manifest']['scenario_config'];cfg=(LocalFrameConfig if cfgdata['schema'].endswith('local_frame') else Config)(**cfgdata);checks=[]
    for j in [0,7,15,31]:
        env=RingExchangeEnv(cfg);s=states[j]['physical'];env.reset(np.asarray(s['positions']),velocities=np.asarray(s['velocities']),goals=np.asarray(s['goals']))
        for params in read(R33/'DESIGN_CONFIG.json')['DM']:
            for seed in [0,17]:
                key=jax.random.PRNGKey(seed);a=original(env,key,params);b=distributed_shooting(env,key,params);assert a.dtype==b.dtype and a.tobytes()==b.tobytes();checks.append(dict(state_index=j,parameters=params,key=seed,bitwise=True))
    write(HERE/'MPC_RING_EQUIVALENCE_V2.json',dict(utc=utc(),passed=True,checks=checks,new_physical_steps=0,jax_enable_x64=bool(jax.config.x64_enabled)));source_inventory('imports_mpc_unit_v2.json')
if __name__=='__main__':main()
