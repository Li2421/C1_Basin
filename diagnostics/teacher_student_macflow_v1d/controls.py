"""Nonlearning nominal policies; frozen Runtime.action supplies centralized TT."""
from core import *
from poc.field_controller import Runtime
from shared_control.diagnostic_corrector import bounded_rows
import jax
jax.config.update('jax_enable_x64',False)
class Control(Runtime):
    def __init__(self,kind):
        from data import teacher
        base=teacher(1)
        for k in ['scenario','config','config_cls','env_cls','manifest','basis','cbf']:setattr(self,k,getattr(base,k))
        self.kind=kind;assert kind in ['Z','G'];self.checkpoint=HERE/'CONTROL_DEFINITIONS.json'
        jax.config.update('jax_enable_x64',False)
    def nominal(self,env,key,traced=False):
        assert not traced
        action=np.zeros_like(env.positions,dtype=np.float64) if self.kind=='Z' else bounded_rows(np.asarray(env.goals-env.positions,np.float64),env.config.max_speed)
        return self.observation(env),action,None
