from common import *
from historical_controllers import GaussianNominal
from diagnostics.controller_q_transfer_round3_v1.flow_adapter import RingFlowAdapter
from ring_exchange.local_frame import world_actions_to_local,local_actions_to_world
import pickle,jax
jax.config.update('jax_enable_x64',False)
p=read(D/'axis2_controller_v1/main_protocol.json');s=p['states'][0];checks=[]
for cfg in read(D/'axis2_controller_v1/GAUSSIAN_CONFIGS.json'):
 manifest=read(cfg['manifest']);old=pickle.loads(Path(cfg['flow_checkpoint']).read_bytes())['config'];frame=cfg['frame'];rt=object.__new__(RingFlowAdapter);rt.frame=frame
 from ring_exchange.environment import Config,LocalFrameConfig,RingExchangeEnv
 env=RingExchangeEnv((Config if frame=='global' else LocalFrameConfig)(**manifest['scenario_config']));env.reset(**{k:np.asarray(v) for k,v in s['physical'].items() if k!='timestep'})
 obs=rt.observation(env);assert obs.shape==(4,23);assert np.asarray(old['obs_mean']).size==92;assert np.asarray(old['act_mean']).size==8
 rng=np.random.default_rng(3800);world=rng.normal(size=(4,2));chart=rt.to_local(world,env);assert np.allclose(rt.to_world(chart,env),world,rtol=0,atol=1e-12)
 angle=.731;rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]);rot=RingExchangeEnv(env.config);rot.reset(env.positions@rotation.T,velocities=env.velocities@rotation.T,goals=env.goals@rotation.T)
 if frame=='local':
  assert np.allclose(rt.observation(env),rt.observation(rot),atol=2e-6);assert np.allclose(rt.to_world(chart,rot),world@rotation.T,atol=1e-12)
 else:assert not np.allclose(rt.observation(env),rt.observation(rot))
 checks.append(dict(recipe=cfg['recipe'],frame=frame,observation_dim=92,action_dim=8,manifest_sha256=sha(cfg['manifest']),normalization_sha256=sha(cfg['flow_checkpoint']),inverse_and_rotation=True))
write(HERE/'FRAME_TESTS_NEW.json',dict(utc=utc(),passed=True,checks=checks,physical_steps=0,before_training=True,jax_enable_x64=bool(jax.config.x64_enabled)))
print('six chart tests PASS')
