"""Compare observation charts at identical archived states, without physics."""
from common import *
import jax
import jax.numpy as jnp
from training_env import TrainingEnv, observation
from ring_exchange.environment import RingExchangeEnv, Config, LocalFrameConfig
from ring_exchange.local_frame import local_observation


def main():
    jax.config.update('jax_enable_x64',False)
    checks=[];sources={}
    for cfg in read(D/'axis2_controller_v1/GAUSSIAN_CONFIGS.json'):
        manifest=read(cfg['manifest']);config=manifest['scenario_config']
        sources[cfg['manifest']]=sha(cfg['manifest'])
        envcfg=(LocalFrameConfig if cfg['frame']=='local' else Config)(**config)
        env=RingExchangeEnv(envcfg);training=TrainingEnv(config)
        observed=jax.jit(lambda state:observation(state,config,cfg['frame']))
        rows=[r for r in manifest['files'] if r['split']=='train'][:32]
        for row in rows:
            path=Path(cfg['manifest']).parent/row['file'];sources[str(path)]=sha(path)
            with np.load(path,allow_pickle=False) as data:
                initial=json.loads(str(data['initial_state_json'].item()))
                positions=data['states'].copy()
            goals=np.asarray(initial['goals'],np.float32)
            for t in np.unique(np.linspace(0,len(positions)-1,8,dtype=int)):
                p=np.asarray(positions[t],np.float32)
                v=np.asarray(initial.get('velocities',np.zeros((4,2))) if t==0 else
                             (positions[t]-positions[t-1])/config['dt'],np.float32)
                state=training.reset(jnp.asarray(p),jnp.asarray(v),jnp.asarray(goals))
                # Both implementations receive exactly the same rounded state.
                env.reset(p,velocities=v,goals=goals)
                reference=env.observation() if cfg['frame']=='global' else local_observation(p,v,goals,envcfg)
                error=float(np.max(np.abs(np.asarray(observed(state))-reference)))
                checks.append(dict(recipe=cfg['name'],frame=cfg['frame'],trajectory=row['file'],step=int(t),error=error))
    worst=max(row['error'] for row in checks)
    output=HERE/'TRAINING_OBSERVATION_SAME_STATE.json'
    write(output,dict(utc=utc(),passed=worst<=2e-6,max_error=worst,tolerance=2e-6,
          states=len(checks),checks=checks,source_sha256=sources,physical_steps=0,
          reason='Earlier float32 observation comparison also included accumulated plant trajectory drift'))
    f64=read(HERE/'TRAINING_ENV_FLOAT64.json');f32=read(HERE/'TRAINING_ENV_FLOAT32.json')
    physics_passed=(f64['max_position_error']<=1e-9 and f64['max_clearance_error']<=1e-9
                    and f64['termination_mismatch_fraction']==0 and f32['termination_mismatch_fraction']<=.005)
    coverage=all(set(r['terminal_coverage'])=={'collision','success','timeout'} and
                 any(c['agent_collision'] for c in r['checks']) and any(c['wall_collision'] for c in r['checks'])
                 for r in (f64,f32))
    write(HERE/'TRAINING_ENV_GATE.json',dict(utc=utc(),passed=physics_passed and coverage and worst<=2e-6,
          required_physics_passed=physics_passed,coverage_passed=coverage,same_state_observation_passed=worst<=2e-6,
          original_float32_combined_passed=f32['passed'],original_results_preserved=True,
          observation_check_correction='Use identical physical states for the chart check; retain full-trajectory drift separately',
          evidence={str(p):sha(p) for p in (HERE/'TRAINING_ENV_FLOAT64.json',HERE/'TRAINING_ENV_FLOAT32.json',output)}))
    print(json.dumps(dict(physics_passed=physics_passed,same_state_observation_max_error=worst,states=len(checks))))


if __name__=='__main__':main()
