"""Restart stale Adam momentum only when its proposal is not a descent direction.

The fixed-batch C1 objective and primal-dual semantics do not change. Both
ordinary and restarted proposals still require an actual accepted backtrack;
failure commits neither parameters nor any optimizer state.
"""
import jax
import numpy as np
from single_integrator.c1.training.step_control import guarded_step as original, finite


def guarded_step(params, state, gradient, optimizer, objective, before, max_backtracks=8):
    if not finite((params, state, gradient, before)):
        raise FloatingPointError('nonfinite input to guarded optimizer')
    updates, _ = optimizer.update(gradient, state, params)
    if not finite(updates):
        raise FloatingPointError('nonfinite Adam proposal')
    gs, us = jax.tree_util.tree_leaves(gradient), jax.tree_util.tree_leaves(updates)
    dot = float(sum(np.vdot(np.asarray(g), np.asarray(u)) for g, u in zip(gs, us)))
    nonzero = any(np.any(np.asarray(g) != 0) for g in gs)
    restart = dot >= 0 and nonzero
    trial_state = optimizer.init(params) if restart else state
    candidate, candidate_state, after, info = original(
        params, trial_state, gradient, optimizer, objective, before, max_backtracks)
    info = dict(info, proposal='restart' if restart else 'ordinary', original_directional_derivative=dot)
    if info['status'] != 'accepted':
        return params, state, before, info
    return candidate, candidate_state, after, info
