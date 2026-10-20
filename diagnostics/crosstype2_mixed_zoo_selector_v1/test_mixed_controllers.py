from common import *
from mixed_controllers import *
from types import SimpleNamespace


def single_agent_central_reference(env,key,parameters):
    """Original round38 CEM, with its hardcoded four-agent axis made dynamic.

    This test-only generalization is needed because the historical source cannot
    accept one robot. Optimization order, noise stream and elite update match.
    """
    horizon,iterations,samples=parameters;cfg=env.config;n=len(env.positions)
    rng=random_generator(key)
    mean=np.broadcast_to(bound(env.goals-env.positions,cfg.max_speed),(3,n,2)).copy();std=np.full_like(mean,.8*cfg.max_speed)
    for _ in range(iterations):
        knots=bound(mean[None]+rng.normal(size=(samples,3,n,2))*std[None],cfg.max_speed)
        actions=knots[:,np.minimum(np.arange(horizon)*3//horizon,2)]
        positions=env.positions[None,None]+cfg.dt*np.cumsum(actions,axis=1)
        if n==4:cost=shooting_cost(positions,actions,env.goals,cfg,100,env.velocities)
        else:
            assert n==1
            goal=np.square(positions-env.goals).sum(-1).mean((1,2))
            radius=np.linalg.norm(positions,axis=-1)
            inner=np.maximum(cfg.obstacle_radius+cfg.agent_radius+cfg.collision_margin-radius,0)**2
            outer=np.maximum(radius-(cfg.outer_radius-cfg.agent_radius-cfg.collision_margin),0)**2
            penalty=inner.mean((1,2))+outer.mean((1,2))
            changes=np.diff(np.concatenate((np.broadcast_to(env.velocities,(len(actions),1,n,2)),actions),axis=1),axis=1)
            smooth=np.square(changes).sum(-1).mean((1,2));cost=goal+100*penalty+.05*smooth
        elite=np.argsort(cost,kind='stable')[:int(np.ceil(.1*samples))]
        mean=knots[elite].mean(0);std=np.maximum(knots[elite].std(0),.001)
    return actions[int(np.argmin(cost)),0].copy()


def main():
    config=read(HERE/'NEW_CONTROLLER_CONFIGS.json');cfg=SimpleNamespace(**config['test_environment'])
    env=SimpleNamespace(config=cfg,positions=np.asarray([[1.,.4]]),velocities=np.asarray([[.1,-.05]]),goals=np.asarray([[2.,.8]]),step_count=0)
    checks=[]
    for parameters in config['CEM']:
        for seed in (17,41):
            key=jax.random.PRNGKey(seed);a=distributed_cross_entropy(env,key,parameters);b=single_agent_central_reference(env,key,parameters)
            error=float(np.max(np.abs(a-b)));assert error<=1e-9
            checks.append(dict(parameters=parameters,seed=seed,max_first_action_error=error))
    env.positions=np.asarray([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
    env.velocities=np.asarray([[.1,.2],[.2,.1],[-.2,.1],[.1,-.2]]);env.goals=-env.positions
    for parameters in config['CEM']:
        key=jax.random.PRNGKey(23);a=single_agent_central_reference(env,key,parameters);b=cross_entropy(env,key,parameters)
        assert a.tobytes()==b.tobytes()
    for robot in range(4):
        forecast=constant_velocity_predictions(env,10,robot);others=[i for i in range(4) if i!=robot]
        expected=env.positions[others][None]+cfg.dt*np.arange(1,11)[:,None,None]*env.velocities[others][None]
        assert forecast.tobytes()==expected.tobytes()
    rng=np.random.default_rng(202610182031);actions=rng.normal(size=(64,10,4,2));cost=np.arange(64,dtype=float)[::-1]
    cold=weighted_path_integral(actions,cost,1e-12);assert np.array_equal(cold,actions[cost.argmin()])
    manifest=read(D/'crosstype1b_generator_critic_pipeline_v1/BASE_PROTOCOL.json')['native_manifest']
    for parameters in config['MPPI']:
        one=CentralizedMPPI(parameters,manifest);two=CentralizedMPPI(parameters,manifest);key=jax.random.PRNGKey(41)
        a=one.nominal(env,key)[1];b=two.nominal(env,key)[1];assert a.tobytes()==b.tobytes()
        saved=one.mean.copy();env.step_count=1
        a=one.nominal(env,key)[1];b=two.nominal(env,key)[1];assert a.tobytes()==b.tobytes()
        assert saved.shape==(parameters[0],4,2);env.step_count=0
    write(HERE/'MIXED_CONTROLLER_TESTS.json',dict(utc=utc(),passed=True,distributed_single_agent_checks=checks,generalized_central_reference_matches_original_four_agent_bitwise=True,constant_velocity_forecast_bitwise=True,mppi_zero_temperature_min_sample=True,mppi_same_key_bitwise=True,mppi_receding_warm_start=True,physical_rollouts=0,source_cost_sha256=sha(D/'other_controller_classes_v1/controllers.py')))
    print('New optimizer tests passed; no task outcome labels')


if __name__=='__main__':main()
