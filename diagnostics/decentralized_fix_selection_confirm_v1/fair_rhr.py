"""Private speed/progress histories and clockwise tangent after local CBF."""
from core import *
from pipeline import LocalSafety
import jax
jax.config.update('jax_enable_x64',False)
MODES=['always','slow','progress']
def grid():
    return [dict(mode=m,steps=t,speed=v) for m in MODES for t in ([0] if m=='always' else [20,40]) for v in [.10,.15,.25,.35]]
class Trigger:
    def __init__(self,rule):self.rule=rule;self.reset()
    def reset(self):self.last=-1;self.count=np.zeros(4,int);self.distances={};self.active=np.zeros(4,bool)
    def update(self,env):
        t=env.step_count
        if t==0 and self.last!=0:self.reset()
        if t==self.last:return self.active.copy()
        assert t==0 or t==self.last+1,'History must contain every physical step'
        distance=np.linalg.norm(env.goals-env.positions,axis=1);moving=np.linalg.norm(env.velocities,axis=1);not_goal=distance>env.config.goal_tolerance
        if t>0:self.count=np.where((moving<.05)&not_goal,self.count+1,0)
        self.distances[t]=distance.copy();m=self.rule['mode'];w=self.rule['steps']
        if m=='always':active=not_goal
        elif m=='slow':active=(self.count>=w)&not_goal
        else:active=(self.distances[t-w]-distance<.05)&not_goal if t>=w else np.zeros(4,bool)
        self.active=active;self.last=t
        for k in list(self.distances):
            if k<t-max(w,1):del self.distances[k]
        return active.copy()
class FairRuntime:
    def __init__(self,base,rule):self.base=base;self.rule=rule;self.trigger=Trigger(rule);self.active_log=[];self.neighbor_log=[]
    def __getattr__(self,name):return getattr(self.base,name)
    def action(self,env,key,eta,chain='old'):
        active=self.trigger.update(env);nominal=np.stack([self.base.nominal(env,key)[1][i] for i in range(4)])
        filters=[LocalSafety(env.positions,i,env.config,1.5) for i in range(4)];safe=np.stack([f.project(nominal[i:i+1])[0][0] for i,f in enumerate(filters)]);target=safe.copy()
        for i in range(4):
            if active[i]:q=env.positions[i];target[i]+=self.rule['speed']*np.array([q[1],-q[0]])/max(np.linalg.norm(q),1e-30)
        executed=np.stack([f.project(target[i:i+1])[0][0] for i,f in enumerate(filters)])
        self.active_log.append(active);self.neighbor_log.append([len(f.neighbor_indices) for f in filters])
        return executed,dict(min_linear_residual=min(f.certificate(executed[i:i+1])['min_linear_residual'] for i,f in enumerate(filters))),sum(f.calls for f in filters)
