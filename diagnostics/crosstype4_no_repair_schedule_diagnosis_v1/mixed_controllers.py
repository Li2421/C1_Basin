"""New black-box optimizers; original cost and deployment plant stay read-only."""
from common import *
from legacy_controllers import Nominal,bound,shooting_cost
from controllers import cross_entropy
import jax
jax.config.update('jax_enable_x64',False)


def random_generator(key):
    token=np.asarray(key,np.uint32).tobytes()
    return np.random.default_rng(int.from_bytes(hashlib.sha256(token).digest()[:8],'little'))


def constant_velocity_predictions(env,horizon,robot):
    neighbors=[j for j in range(len(env.positions)) if j!=robot]
    return env.positions[neighbors][None]+env.config.dt*np.arange(1,horizon+1)[:,None,None]*env.velocities[neighbors][None]


def scoped_shooting_cost(positions,actions,goal,cfg,weight,previous,forecast):
    # Literal round27 cost terms with reductions scoped to the optimized robot.
    goal=np.square(positions-goal).sum(-1).mean(1)
    radius=np.linalg.norm(positions,axis=-1)
    inner=np.maximum(cfg.obstacle_radius+cfg.agent_radius+cfg.collision_margin-radius,0)**2
    outer=np.maximum(radius-(cfg.outer_radius-cfg.agent_radius-cfg.collision_margin),0)**2
    dd=np.linalg.norm(positions[:,:,None]-forecast[None],axis=-1)
    pairs=np.maximum(2*cfg.agent_radius+cfg.agent_collision_margin-dd,0)**2
    pair_penalty=pairs.mean((1,2)) if forecast.shape[1] else np.zeros(len(actions))
    penalty=inner.mean(1)+outer.mean(1)+pair_penalty
    changes=np.diff(np.concatenate((np.broadcast_to(previous,(len(actions),1,2)),actions),axis=1),axis=1)
    smooth=np.square(changes).sum(-1).mean(1)
    return goal+weight*penalty+.05*smooth


def distributed_cross_entropy(env,key,parameters):
    horizon,iterations,samples=parameters;cfg=env.config;answer=[];count=len(env.positions)
    for robot in range(count):
        # A single-agent reference consumes exactly the original, unfolded key.
        local_key=key if count==1 else jax.random.fold_in(key,robot)
        rng=random_generator(local_key)
        mean=np.broadcast_to(bound((env.goals-env.positions)[robot],cfg.max_speed),(3,2)).copy()
        std=np.full_like(mean,.8*cfg.max_speed)
        forecast=constant_velocity_predictions(env,horizon,robot)
        for _ in range(iterations):
            knots=bound(mean[None]+rng.normal(size=(samples,3,2))*std[None],cfg.max_speed)
            actions=knots[:,np.minimum(np.arange(horizon)*3//horizon,2)]
            positions=env.positions[robot]+cfg.dt*np.cumsum(actions,axis=1)
            cost=scoped_shooting_cost(positions,actions,env.goals[robot],cfg,100,env.velocities[robot],forecast)
            elite=np.argsort(cost,kind='stable')[:int(np.ceil(.1*samples))]
            mean=knots[elite].mean(0);std=np.maximum(knots[elite].std(0),.001)
        answer.append(actions[int(np.argmin(cost)),0].copy())
    return np.asarray(answer)


def weighted_path_integral(actions,cost,temperature):
    assert temperature>0
    weights=np.exp(-(cost-cost.min())/temperature);weights/=weights.sum()
    return np.sum(weights.reshape((-1,)+(1,)*(actions.ndim-1))*actions,axis=0)


class DistributedCrossEntropy(Nominal):
    def __init__(self,parameters,manifest):super().__init__('DCEM',parameters,manifest)
    def nominal(self,env,key,traced=False):
        assert not traced
        return None,distributed_cross_entropy(env,key,self.parameters),None


class CentralizedMPPI(Nominal):
    def __init__(self,parameters,manifest):
        super().__init__('MPPI',parameters,manifest);self.episode_env=None;self.mean=None;self.last_step=None
    def nominal(self,env,key,traced=False):
        assert not traced
        horizon,samples,temperature=self.parameters;cfg=env.config
        if env is not self.episode_env or (env.step_count==0 and self.last_step!=0):
            self.episode_env=env;self.mean=np.broadcast_to(bound(env.goals-env.positions,cfg.max_speed),(horizon,len(env.positions),2)).copy()
        rng=random_generator(key)
        actions=bound(self.mean[None]+rng.normal(size=(samples,*self.mean.shape))*(.8*cfg.max_speed),cfg.max_speed)
        positions=env.positions[None,None]+cfg.dt*np.cumsum(actions,axis=1)
        cost=shooting_cost(positions,actions,env.goals,cfg,100,env.velocities)
        updated=weighted_path_integral(actions,cost,temperature);answer=updated[0].copy()
        self.mean=np.concatenate((updated[1:],updated[-1:]),axis=0);self.last_step=env.step_count
        return None,answer,None


def make(name,protocol,cache):
    payload=protocol['profiles'][name]['payload'];kind=payload.get('controller_class','Flow')
    if kind not in ('DCEM','MPPI','ASTAR_CORRECTED'):
        from zoo_runtime import make as historical_make
        return historical_make(name,protocol,cache)
    if name not in cache:
        if kind=='ASTAR_CORRECTED':
            from corrected_astar_reference import CorrectedPrioritizedAStar
            cache[name]=CorrectedPrioritizedAStar(payload['nominal_parameters'],protocol['native_manifest'])
        else:
            cls=DistributedCrossEntropy if kind=='DCEM' else CentralizedMPPI
            cache[name]=cls(payload['nominal_parameters'],protocol['native_manifest'])
    jax.config.update('jax_enable_x64',False)
    return cache[name]
