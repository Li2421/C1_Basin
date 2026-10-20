from controllers import *
import types
cfg=types.SimpleNamespace(dt=.05,max_speed=.52,obstacle_radius=.9,outer_radius=3.25,agent_radius=.16,collision_margin=.005,agent_collision_margin=0.)
p=np.array([[2,0],[0,2],[-2,0],[0,-2]],float);env=types.SimpleNamespace(config=cfg,positions=p,velocities=np.zeros((4,2)),goals=-p)
for gains in P_GRID:
 a=potential(env,gains);assert np.isfinite(a).all() and np.max(np.linalg.norm(a,axis=-1))<=cfg.max_speed+1e-14
for spec in M_GRID:
 a=shooting(env,jax.random.PRNGKey(1),spec);b=shooting(env,jax.random.PRNGKey(1),spec);assert np.array_equal(a,b) and np.max(np.linalg.norm(a,axis=-1))<=cfg.max_speed+1e-14
actions=np.zeros((2,3,4,2));pos=np.broadcast_to(p,(2,3,4,2)).copy();pos[1]=0
assert shooting_cost(pos,actions,p,cfg,100,env.velocities)[1]>shooting_cost(pos,actions,p,cfg,100,env.velocities)[0]
assert np.allclose(p[None,None]+cfg.dt*np.cumsum(np.ones((1,3,4,2)),axis=1),p[None,None]+cfg.dt*np.arange(1,4)[None,:,None,None])
write(HERE/'UNIT_TESTS.json',dict(utc=utc(),status='PASS',deterministic_bounded_MPC=True,bounded_potential=True,collision_cost_and_model=True,jax_enable_x64=bool(jax.config.x64_enabled),no_simulator_steps=True));print('unit tests PASS')
