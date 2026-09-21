"""Mathematical primitives for C1's single-policy primal-dual objective.

The rollout code supplies executed controls and one trajectory risk per
trajectory. This module contains no policy snapshot, trajectory reweighting,
or regression-target construction.
"""
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class PrimalDualState:
    """Nonnegative dual state for the constraint ``J_live <= epsilon``."""

    dual: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.dual) or self.dual < 0:
            raise ValueError("dual must be finite and nonnegative")


def deviation_cost(executed, safe, metric=None, step_mask=None):
    """Return ``mean_{B,T} (u_HS-u_safe)^T M (u_HS-u_safe)``.

    Inputs have shape ``[B,T,D]``. ``metric`` is identity, diagonal ``[D]``,
    or matrix ``[D,D]``. The caller owns validation that a configured metric is
    positive semidefinite.
    """
    executed, safe = map(jnp.asarray, (executed, safe))
    if executed.ndim != 3 or executed.shape != safe.shape:
        raise ValueError("executed and safe must have matching shape [B,T,D]")
    delta = executed - safe
    if metric is None:
        per_step = jnp.sum(delta * delta, axis=-1)
    else:
        metric = jnp.asarray(metric)
        if metric.shape == (delta.shape[-1],):
            per_step = jnp.sum(delta * delta * metric, axis=-1)
        elif metric.shape == (delta.shape[-1], delta.shape[-1]):
            per_step = jnp.einsum("...i,ij,...j->...", delta, metric, delta)
        else:
            raise ValueError("metric must have shape [D] or [D,D]")
    if step_mask is None:
        return jnp.mean(per_step)
    mask=jnp.asarray(step_mask,per_step.dtype)
    if mask.shape!=per_step.shape:
        raise ValueError('step_mask must have shape [B,T]')
    return jnp.mean(jnp.sum(jnp.where(mask>0,per_step,0.),axis=-1)/jnp.maximum(jnp.sum(mask,axis=-1),1.))


def liveness_cost(trajectory_risk):
    """Return ``J_live = mean_B R_risk(tau)``."""
    risk = jnp.asarray(trajectory_risk)
    if risk.ndim == 0:
        return risk
    if risk.ndim != 1 or risk.size == 0:
        raise ValueError("trajectory_risk must be a scalar J_live or shape [B] with B > 0")
    return jnp.mean(risk)


def primal_loss(executed, safe, trajectory_risk, dual, epsilon, metric=None, step_mask=None):
    """Return the active C1 loss and its three directly logged components."""
    if not np.isfinite(float(dual)) or float(dual) < 0:
        raise ValueError("dual must be finite and nonnegative")
    if not np.isfinite(float(epsilon)):
        raise ValueError("epsilon must be finite")
    j_def = deviation_cost(executed, safe, metric,step_mask)
    j_live = liveness_cost(trajectory_risk)
    loss = j_def + jnp.asarray(dual, dtype=j_def.dtype) * (j_live - epsilon)
    return loss, {"J_def": j_def, "J_live": j_live, "constraint": j_live - epsilon}


def dual_update(state, trajectory_risk, epsilon, learning_rate):
    """Apply ``[lambda + eta_lambda * (J_live - epsilon)]_+``."""
    if not isinstance(state, PrimalDualState):
        raise TypeError("state must be PrimalDualState")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite")
    j_live = float(liveness_cost(trajectory_risk))
    if not np.isfinite(j_live):
        raise ValueError('liveness risk must be finite before dual update')
    return PrimalDualState(max(0.0, state.dual + learning_rate * (j_live - epsilon)))
