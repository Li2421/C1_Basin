"""Transactional Adam proposal with fixed-batch, fixed-dual backtracking.

This changes the numerical solver, not the C1 objective. Acceptance guarantees
only nonincrease on the sampled batch, not feasibility or generalization.
"""
import jax
import numpy as np
import optax


def finite(tree):
    return all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(tree))


def guarded_step(params, state, gradient, optimizer, objective, before, max_backtracks=8):
    if not isinstance(max_backtracks, int) or max_backtracks < 0:
        raise ValueError('max_backtracks must be a nonnegative integer')
    if not finite((params, state, gradient, before)):
        raise FloatingPointError('nonfinite input to guarded optimizer')
    updates, proposed_state = optimizer.update(gradient, state, params)
    if not finite((updates, proposed_state)):
        raise FloatingPointError('nonfinite Adam proposal')
    attempts = []
    if all(np.all(np.asarray(x) == 0) for x in jax.tree_util.tree_leaves(updates)):
        return params, state, before, dict(status='no_op', scale=0., trials=attempts)
    for index in range(max_backtracks + 1):
        scale = 0.5 ** index
        candidate = optax.apply_updates(params, jax.tree_util.tree_map(lambda x: scale*x, updates))
        if not finite(candidate):
            raise FloatingPointError('nonfinite candidate parameters')
        after = objective(candidate)
        valid = finite(after)
        attempts.append(dict(scale=scale, finite=valid,
                             loss=float(after[0]) if valid else None))
        if valid and float(after[0]) <= float(before[0]):
            # Adam moments describe the accepted gradient observation; only the
            # parameter displacement is scaled. Rejected proposals commit neither.
            return candidate, proposed_state, after, dict(status='accepted', scale=scale, trials=attempts)
    return params, state, before, dict(status='rejected', scale=0., trials=attempts)
