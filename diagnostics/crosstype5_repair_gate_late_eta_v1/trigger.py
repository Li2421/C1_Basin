from core import *
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
