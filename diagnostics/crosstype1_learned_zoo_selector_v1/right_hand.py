"""Verbatim historical RightHand class; isolated safe imports."""
from common import *
from trigger import Trigger
from poc.field_controller import Runtime

class RightHand:
    def __init__(self,base,rule):self.base=base;self.rule=rule;self.trigger=Trigger(rule);self.last_safety=None;self.active_log=[]
    def __getattr__(self,k):return getattr(self.base,k)
    def safety(self,env):self.last_safety=self.base.safety(env);return self.last_safety
    def prepare(self,*args,**kw):return Runtime.prepare(self,*args,**kw)
    def action(self,env,key,eta,chain='old'):
        assert np.array_equal(np.asarray(eta),np.zeros(3));active=self.trigger.update(env);baseline,certificate,calls=Runtime.action(self,env,key,np.zeros(3),chain)
        if not active.any():self.active_log.append(active);return baseline,certificate,calls
        target=baseline.copy()
        for i in range(len(env.positions)):
            if active[i]:q=env.positions[i];target[i]+=self.rule['speed']*np.array([q[1],-q[0]])/max(np.linalg.norm(q),1e-30)
        action,_=self.last_safety.project(target);certificate=self.last_safety.assert_safe(action);self.active_log.append(active);return action,certificate,self.last_safety.calls
