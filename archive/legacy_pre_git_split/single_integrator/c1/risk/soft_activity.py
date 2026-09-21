"""Provisional R_risk_v0_1; hard TRO diagnostics remain R_risk_v0.

Local candidates use h <= rho + candidate_sigmas*tau_h. The outer shell
keeps membership fixed across h=rho, while excluding distant constraints.
Cone topology and this outer locality cutoff remain piecewise geometric
choices; this does not claim global smoothness at every cone transition.
"""
from dataclasses import dataclass
from functools import partial
import numpy as np
import jax.numpy as jnp
import jax
from .risk_function import RiskConfig, risk_diagnostics

@dataclass(frozen=True)
class SoftRiskConfig(RiskConfig):
    tau_h: float = .01
    tau_s: float = .01
    candidate_sigmas: float = 6.

    def __post_init__(self):
        super().__post_init__()
        if any(not np.isfinite(x) or x <= 0 for x in (self.tau_h,self.tau_s,self.candidate_sigmas)):
            raise ValueError('soft activity temperatures/local shell must be finite and positive')

@partial(jax.jit, static_argnames=('config',))
def soft_risk_diagnostics(task_force, safety_force_blocks, h, residual, config=None):
    config = config or SoftRiskConfig()
    h,residual = jnp.asarray(h),jnp.asarray(residual)
    blocks = jnp.asarray(safety_force_blocks).reshape(-1,2,2)
    candidate = h <= config.rho+config.candidate_sigmas*config.tau_h
    activity = jax.nn.sigmoid((config.rho-h)/config.tau_h)*jnp.exp(-(residual/config.tau_s)**2)
    relevant = candidate[:,None] & (jnp.sum(blocks*blocks,axis=-1)>0)
    local_activity = jnp.max(jnp.where(relevant,activity[:,None],0.),axis=0,initial=0.)
    cone = risk_diagnostics(task_force,blocks,candidate,config)
    agent_risk = local_activity*cone['agent_risk']
    hard_mask = (h<=config.rho)&(jnp.abs(residual)<=config.active_tol)
    hard = risk_diagnostics(task_force,blocks,hard_mask,config)
    return dict(risk=jnp.max(agent_risk),agent_risk=agent_risk,
                margins=cone['margins'],cone_types=cone['cone_types'],
                cone_risk=cone['agent_risk'],local_activity=local_activity,
                constraint_activity=activity,candidate_mask=candidate,residual=residual,
                active_mask=hard_mask,hard_risk=hard['risk'],
                hard_margins=hard['margins'],hard_cone_types=hard['cone_types'],
                hard_deadlock_geometry=(hard['cone_types']!=0)&(hard['margins']<=0))
