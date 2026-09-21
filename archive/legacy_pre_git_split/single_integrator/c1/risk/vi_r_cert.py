"""Frozen direct and projection-optimality R_CERT trajectory construction.

The event logic lives in the rollout/authoritative monitor.  This module only
evaluates certificate margins on genuine action windows selected by saved
pre-action guards.  Witnesses are points in the same feasible control set;
they are never executed actions.
"""
import math

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


LOG_TWO = math.log(2.)
CONTROL_SCALE = .5
CONTROL_SCALE_SQUARED = .25
KAPPA = 1.
SIGMA = 1.
STRICT_FIRST_ACTION = 139
STRICT_LAST_ACTION = 849
STRICT_WINDOW = 101
STRICT_DIRECT_COUNT = 303
STRICT_AUGMENTED_COUNT = 2727
STALLED_DIRECT_COUNT = 42
STALLED_AUGMENTED_COUNT = 378


def _blocks(q):
    q = jnp.asarray(q, jnp.float64)
    if q.shape[-1] != 4:
        raise ValueError('joint controls must have final dimension four')
    return q.reshape(*q.shape[:-1], 2, 2)


def squared_distance_speed_bad_set(q, radius):
    """Distance squared to the product of two closed velocity balls."""
    blocks = _blocks(q)
    excess = jnp.maximum(jnp.linalg.norm(blocks, axis=-1) - radius, 0.)
    return jnp.sum(excess * excess, axis=-1)


def squared_distance_goal_bad_set(q, positions, goals, dt, tolerance):
    """Distance squared to controls leaving at least one agent outside goal.

    Under ``x+ = x + dt*q``, the bad set is the union of the exteriors of
    two control-space balls centered at ``(goal-x)/dt`` with radius
    ``tolerance/dt``.  Distance to the union is the minimum block distance.
    """
    blocks = _blocks(q)
    centers = (jnp.asarray(goals) - jnp.asarray(positions)) / dt
    radial = jnp.linalg.norm(blocks - centers, axis=-1)
    block_distance = jnp.maximum(tolerance / dt - radial, 0.)
    return jnp.min(block_distance * block_distance, axis=-1)


def squared_distance_progress_bad_set(q, positions, goals, anchors, dt,
                                      epsilon, agent=None):
    """Distance squared to dynamics-derived distance-persistence annuli.

    For each selected agent, the next goal distance must lie in
    ``[max(anchor-epsilon,0), anchor+epsilon]``.  The strict event atom is a
    subset of this closed set.  With ``agent=None`` both annuli are required;
    stalled terminal progress uses one selected agent per atom.
    """
    blocks = _blocks(q)
    centers = (jnp.asarray(goals) - jnp.asarray(positions)) / dt
    anchors = jnp.asarray(anchors)
    radial = jnp.linalg.norm(blocks - centers, axis=-1)
    lower = jnp.maximum(anchors - epsilon, 0.) / dt
    upper = (anchors + epsilon) / dt
    distance = jnp.maximum(jnp.maximum(lower - radial, radial - upper), 0.)
    if agent is not None:
        return distance[..., agent] ** 2
    return jnp.sum(distance * distance, axis=-1)


def _vi_margins(w, witnesses, distance_squared):
    midpoint = (w[None, :] + witnesses) / 2.
    separation = jnp.sum((w[None, :] - witnesses) ** 2, axis=-1) / 4.
    return (distance_squared(midpoint) - separation) / CONTROL_SCALE_SQUARED


def strict_atom_margins(before, after, w, applied, witnesses, goals, anchors,
                        *, dt=.05, goal_tolerance=.08,
                        progress_epsilon=.01, speed_limit=.025):
    """Return margins ``[goal/progress/speed, direct+8 VI]`` for one action."""
    before, after, w, applied, witnesses = map(
        lambda z: jnp.asarray(z, jnp.float64),
        (before, after, w, applied, witnesses))
    distances = jnp.linalg.norm(after - goals, axis=-1)
    direct_goal = (goal_tolerance - jnp.max(distances)) / goal_tolerance
    direct_progress = (jnp.max(jnp.abs(anchors - distances)) -
                       progress_epsilon) / progress_epsilon
    direct_speed = (squared_distance_speed_bad_set(applied, speed_limit) /
                    CONTROL_SCALE_SQUARED)
    goal_vi = _vi_margins(
        w, witnesses,
        lambda q: squared_distance_goal_bad_set(
            q, before, goals, dt, goal_tolerance))
    progress_vi = _vi_margins(
        w, witnesses,
        lambda q: squared_distance_progress_bad_set(
            q, before, goals, anchors, dt, progress_epsilon))
    speed_vi = _vi_margins(
        w, witnesses,
        lambda q: squared_distance_speed_bad_set(q, speed_limit))
    return jnp.stack((jnp.concatenate((jnp.array([direct_goal]), goal_vi)),
                      jnp.concatenate((jnp.array([direct_progress]), progress_vi)),
                      jnp.concatenate((jnp.array([direct_speed]), speed_vi))))


def stalled_progress_margins(before, after, w, witnesses, goals, anchors,
                             agent, *, dt=.05, epsilon=.02):
    distances = jnp.linalg.norm(after - goals, axis=-1)
    direct = (jnp.abs(distances[agent] - anchors[agent]) - epsilon) / epsilon
    vi = _vi_margins(
        w, witnesses,
        lambda q: squared_distance_progress_bad_set(
            q, before, goals, anchors, dt, epsilon, agent=agent))
    return jnp.concatenate((jnp.array([direct]), vi))


def stalled_speed_margins(w, applied, witnesses, *, speed_limit=.05):
    direct = (squared_distance_speed_bad_set(applied, speed_limit) /
              CONTROL_SCALE_SQUARED)
    vi = _vi_margins(
        w, witnesses,
        lambda q: squared_distance_speed_bad_set(q, speed_limit))
    return jnp.concatenate((jnp.array([direct]), vi))


def _cost(margin):
    return jax.nn.softplus(-margin / SIGMA) / LOG_TWO


def _normalized_softmin(costs, count):
    flat = jnp.ravel(costs)
    return -KAPPA * (logsumexp(-flat / KAPPA) - math.log(count))


def trajectory(before, after, w, applied, witnesses, goals, alive_pre,
               latch_pre, original_other_timeout, *, dt=.05,
               goal_tolerance=.08, progress_epsilon=.01,
               strict_speed_limit=.025, stalled_speed_limit=.05,
               stalled_progress_epsilon=.02):
    """Evaluate guarded direct-only and augmented risks on one 850-step trace.

    Invalid branches are not evaluated: each strict branch is computed inside
    a scalar ``lax.cond`` on its saved pre-action guard.  The stalled branch is
    evaluated only for an original timeout with all 850 genuine actions.
    """
    arrays = tuple(jnp.asarray(x) for x in
                   (before, after, w, applied, witnesses, alive_pre, latch_pre))
    before, after, w, applied, witnesses, alive_pre, latch_pre = arrays
    if before.shape != (850, 2, 2) or after.shape != before.shape:
        raise ValueError('frozen R_CERT requires 850 state transitions')
    if w.shape != (850, 4) or applied.shape != (850, 4):
        raise ValueError('frozen R_CERT requires 850 joint controls')
    if witnesses.shape != (850, 8, 4):
        raise ValueError('all eight witness indices are required at every action')
    if alive_pre.shape != (850,) or latch_pre.shape != (850,):
        raise ValueError('pre-action alive/latch traces must have length 850')
    states = jnp.concatenate((before[:1], after), axis=0)
    state_distances = jnp.linalg.norm(states - goals, axis=-1)
    times = jnp.arange(STRICT_FIRST_ACTION, STRICT_LAST_ACTION + 1,
                       dtype=jnp.int32)
    strict_guards = alive_pre[times] & ~latch_pre[times]

    def branch(t):
        guard = alive_pre[t] & ~latch_pre[t]
        def evaluate(_):
            start = t - (STRICT_WINDOW - 1)
            zero = jnp.array(0, jnp.int32)
            bs = jax.lax.dynamic_slice(before, (start, zero, zero),
                                       (STRICT_WINDOW, 2, 2))
            af = jax.lax.dynamic_slice(after, (start, zero, zero),
                                       (STRICT_WINDOW, 2, 2))
            ww = jax.lax.dynamic_slice(w, (start, zero),
                                       (STRICT_WINDOW, 4))
            uu = jax.lax.dynamic_slice(applied, (start, zero),
                                       (STRICT_WINDOW, 4))
            vv = jax.lax.dynamic_slice(witnesses, (start, zero, zero),
                                       (STRICT_WINDOW, 8, 4))
            anchor_start = start - 39
            aa = jax.lax.dynamic_slice(state_distances,
                                       (anchor_start, zero),
                                       (STRICT_WINDOW, 2))
            margins = jax.vmap(
                lambda b, a, z, u, v, anchor: strict_atom_margins(
                    b, a, z, u, v, goals, anchor, dt=dt,
                    goal_tolerance=goal_tolerance,
                    progress_epsilon=progress_epsilon,
                    speed_limit=strict_speed_limit))(bs, af, ww, uu, vv, aa)
            direct = _normalized_softmin(_cost(margins[:, :, 0]),
                                         STRICT_DIRECT_COUNT)
            augmented = _normalized_softmin(_cost(margins),
                                            STRICT_AUGMENTED_COUNT)
            return jnp.stack((direct, augmented))
        return jax.lax.cond(guard, evaluate,
                            lambda _: jnp.full((2,), -jnp.inf), operand=None)

    strict_risks = jax.lax.map(branch, times)
    enough_tail = jnp.sum(alive_pre) == 850
    stalled_guard = jnp.asarray(original_other_timeout, bool) & enough_tail

    def stalled_branch(_):
        progress = jnp.stack([
            stalled_progress_margins(
                before[849], after[849], w[849], witnesses[849], goals,
                state_distances[811], agent, dt=dt,
                epsilon=stalled_progress_epsilon)
            for agent in range(2)])
        speed = jax.vmap(
            lambda z, u, v: stalled_speed_margins(
                z, u, v, speed_limit=stalled_speed_limit))(
                    w[810:850], applied[810:850], witnesses[810:850])
        margins = jnp.concatenate((progress, speed), axis=0)
        direct = _normalized_softmin(_cost(margins[:, 0]),
                                     STALLED_DIRECT_COUNT)
        augmented = _normalized_softmin(_cost(margins),
                                        STALLED_AUGMENTED_COUNT)
        return jnp.stack((direct, augmented))

    stalled_risks = jax.lax.cond(
        stalled_guard, stalled_branch,
        lambda _: jnp.full((2,), -jnp.inf), operand=None)
    all_risks = jnp.concatenate((strict_risks, stalled_risks[None]), axis=0)
    any_guard = jnp.any(strict_guards) | stalled_guard
    maxima = jnp.max(all_risks, axis=0)
    maxima = jnp.where(any_guard, maxima, jnp.nan)
    active = jnp.argmax(all_risks, axis=0)
    return dict(direct_R_CERT=maxima[0], augmented_R_CERT=maxima[1],
                strict_direct_R_e=strict_risks[:, 0],
                strict_augmented_R_e=strict_risks[:, 1],
                stalled_direct_R_e=stalled_risks[0],
                stalled_augmented_R_e=stalled_risks[1],
                strict_guards=strict_guards, stalled_guard=stalled_guard,
                eligible=any_guard, active_event_index=active)
