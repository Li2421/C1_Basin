from core import *
from pipeline import LocalSafety
from episode_extended import CountedRuntime
import jax
jax.config.update("jax_enable_x64",False)
def right_hand_tuned(rt,env,key,eta,counts,speed,trigger):
    nominal=np.stack([rt.nominal(env,key)[1][i] for i in range(4)])
    filters=[LocalSafety(env.positions,i,env.config,1.5) for i in range(4)];safe=np.stack([f.project(nominal[i:i+1])[0][0] for i,f in enumerate(filters)])
    target=safe.copy();active=np.zeros(4,bool)
    for i in range(4):
        if counts[i]>=trigger and np.linalg.norm(env.goals[i]-env.positions[i])>env.config.goal_tolerance:
            q=env.positions[i];target[i]+=speed*np.array([q[1],-q[0]])/max(np.linalg.norm(q),1e-30);active[i]=True
    executed=np.stack([f.project(target[i:i+1])[0][0] for i,f in enumerate(filters)])
    return executed,dict(projection_calls=sum(f.calls for f in filters),minimum_local_linear_residual=min(f.certificate(executed[i:i+1])['min_linear_residual'] for i,f in enumerate(filters)),neighbor_counts=[len(f.neighbor_indices) for f in filters],rhr_active=active)

class TunedRuntime(CountedRuntime):
    def __init__(self,base,speed,trigger):super().__init__(base);self.speed=speed;self.trigger=trigger
    def action(self,env,key,eta,chain='old'):
        if env.step_count==0:self.last_step=0;self.slow_counts[:]=0
        elif env.step_count!=self.last_step:
            mask=(np.linalg.norm(env.velocities,axis=1)<.01)&(np.linalg.norm(env.goals-env.positions,axis=1)>env.config.goal_tolerance);self.slow_counts=np.where(mask,self.slow_counts+1,0);self.last_step=env.step_count
        a,d=right_hand_tuned(self.base,env,key,eta,self.slow_counts,self.speed,self.trigger)
        return a,dict(min_linear_residual=d['minimum_local_linear_residual']),d['projection_calls']
