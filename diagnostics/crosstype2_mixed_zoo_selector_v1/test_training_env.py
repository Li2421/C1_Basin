"""Separate-process x64/f32 Ring equivalence using original training episodes."""
from common import *
import jax
import jax.numpy as jnp
from training_env import TrainingEnv,observation
from ring_exchange.local_frame import local_observation
from ring_exchange.environment import RingExchangeEnv,Config,LocalFrameConfig


def main(precision):
    assert precision in ('float64','float32')
    jax.config.update('jax_enable_x64',precision=='float64')
    checks=[];dtype=np.float64 if precision=='float64' else np.float32
    cfgs=read(D/'axis2_controller_v1/GAUSSIAN_CONFIGS.json')[:4]
    for group,cfg in enumerate(cfgs):
        manifest=read(cfg['manifest']);config=manifest['scenario_config']
        envcfg=(LocalFrameConfig if cfg['frame']=='local' else Config)(**config)
        training=TrainingEnv(config);step=jax.jit(training.step)
        observed=jax.jit(lambda st:observation(st,config,cfg['frame']))
        # Read existing train trajectories only: no new expert generation.
        records=[r for r in manifest['files'] if r['split']=='train'][:32]
        assert len(records)==32
        for mode in ('random','expert'):
            for number,row in enumerate(records):
                with np.load(Path(cfg['manifest']).parent/row['file'],allow_pickle=False) as data:
                    state=json.loads(str(data['initial_state_json'].item()))
                    positions=np.asarray(state['positions'],np.float64)
                    goals=np.asarray(state['goals'],np.float64)
                    velocities=np.asarray(state.get('velocities',np.zeros_like(positions)),np.float64)
                    expert_actions=data['actions'].copy()
                if mode=='random' and number==3:
                    positions=np.asarray([[2.,0.],[2.,.36],[-2.,0.],[0.,-2.]])
                    goals=positions[[1,0,2,3]].copy();velocities=np.zeros_like(positions)
                env=RingExchangeEnv(envcfg)
                env.reset(positions,velocities=velocities,goals=goals)
                st=training.reset(jnp.asarray(positions,dtype),jnp.asarray(velocities,dtype),jnp.asarray(goals,dtype))
                rng=np.random.default_rng(202610180300+group*100+number)
                max_position=max_clearance=max_observation=0.;reference_term=trial_term='running';trial_done=False
                reference_steps=trial_steps=None
                for t in range(config['max_steps']):
                    if mode=='expert':
                        action=expert_actions[t] if t<len(expert_actions) else np.zeros((4,2))
                    elif number%4==0 or number==3:
                        action=(env.goals-env.positions)
                    elif number%4==1:
                        action=np.zeros((4,2))
                    else:
                        action=rng.normal(size=(4,2))
                    action=action*np.minimum(1.,(.999*config['max_speed'])/np.maximum(np.linalg.norm(action,axis=-1,keepdims=True),1e-30))
                    ref_action=action.copy()
                    ref_action[np.linalg.norm(env.goals-env.positions,axis=-1)<=config['goal_tolerance']]=0
                    _,_,reference_done,info=env.step(ref_action)
                    if not trial_done:
                        st,_,done,trial=step(st,jnp.asarray(action,dtype))
                        trial_done=bool(done)
                        if trial_done:
                            trial_steps=t+1
                            trial_term=('running','collision','success','timeout')[int(trial['termination'])]
                        max_position=max(max_position,float(np.max(np.abs(np.asarray(st.positions)-env.positions))))
                        reference_obs=env.observation() if cfg['frame']=='global' else local_observation(env.positions,env.velocities,env.goals,env.config)
                        max_observation=max(max_observation,float(np.max(np.abs(np.asarray(observed(st))-reference_obs))))
                        for field in ('swept_obstacle_distance','swept_outer_distance','swept_agent_distances'):
                            max_clearance=max(max_clearance,float(np.max(np.abs(np.asarray(trial[field])-info[field]))))
                    if reference_done:
                        reference_term=info['termination'];reference_steps=t+1;break
                same=reference_term==trial_term and reference_steps==trial_steps
                checks.append(dict(group=group,mode=mode,trajectory=row['file'],
                    max_position_error=max_position,max_clearance_error=max_clearance,
                    max_observation_error=max_observation,
                    reference_termination=reference_term,training_termination=trial_term,
                    reference_steps=reference_steps,training_steps=trial_steps,termination_matches=same))
                checks[-1]['agent_collision']=bool(info['agent_collision'])
                checks[-1]['wall_collision']=bool(info['obstacle_collision'] or info['outer_collision'])
        print(precision,group,'complete',flush=True)
    mismatch=float(np.mean([not c['termination_matches'] for c in checks]))
    position=max(c['max_position_error'] for c in checks);clearance=max(c['max_clearance_error'] for c in checks)
    coverage=sorted(set(c['reference_termination'] for c in checks))
    passed=(position<=1e-9 and clearance<=1e-9 and mismatch==0) if precision=='float64' else mismatch<=.005
    passed=passed and set(coverage)=={'success','collision','timeout'}
    passed=passed and any(c['agent_collision'] for c in checks) and any(c['wall_collision'] for c in checks)
    passed=passed and max(c['max_observation_error'] for c in checks)<=2e-6
    write(HERE/f'TRAINING_ENV_{precision.upper()}.json',dict(utc=utc(),precision=precision,passed=passed,
          max_position_error=position,max_clearance_error=clearance,termination_mismatch_fraction=mismatch,
          terminal_coverage=coverage,episodes=len(checks),checks=checks,
          source_plant_sha256=sha(ROOT/'ring_exchange/environment.py'),new_expert_rollouts=0))
    print(precision,'passed',passed,'mismatch',mismatch,flush=True)

if __name__=='__main__':main(sys.argv[1])
