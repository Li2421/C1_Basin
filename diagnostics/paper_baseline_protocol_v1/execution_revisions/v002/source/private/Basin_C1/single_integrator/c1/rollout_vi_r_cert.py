"""Full current-policy rollout for the frozen VI-augmented R_CERT diagnostic."""
import math

import jax
import jax.numpy as jnp

from single_integrator.c1.differentiable_rollout import barrier_constraints
from single_integrator.c1.risk.joint_frozen import controller_projection
from single_integrator.c1.risk.vi_r_cert import trajectory as certificate_trajectory
from single_integrator.c1.termination import monitor_step
from single_integrator.c1.train import observation
from single_integrator.environment import GiveWayEnv


WITNESS_TARGETS = jnp.reshape(
    jnp.stack((.5 * jnp.eye(4), -.5 * jnp.eye(4)), axis=1), (8, 4))


def rollout(params, field, initial, noise, plant, cbf, action_offsets=None):
    """Roll out one episode, preserving raw latch and enabled termination.

    ``action_offsets`` are diagnostic residual perturbations before the second
    projection.  Model parameters are inputs but are never updated here.
    """
    if plant.max_steps != 850 or plant.dt != .05 or plant.max_speed != .5:
        raise ValueError('frozen VI R_CERT constants require the 850-step direct-SI plant')
    if noise.shape != (850, 4):
        raise ValueError('one full-horizon Flow noise trace is required')
    if action_offsets is None:
        action_offsets = jnp.zeros((850, 4), jnp.float64)
    action_offsets = jnp.asarray(action_offsets, jnp.float64)
    if action_offsets.shape != (850, 4):
        raise ValueError('action offsets must have shape [850,4]')
    initial = jnp.asarray(initial, jnp.float64)
    env = GiveWayEnv(plant)
    jax.debug.callback(lambda x: GiveWayEnv(plant).reset(x),
                       jax.lax.stop_gradient(initial))
    goals, walls = jnp.asarray(env.goals), jnp.asarray(env.walls)
    history_size = int(math.ceil(
        plant.progress_window_seconds / plant.dt)) + 1
    history = jnp.zeros((history_size, 2), jnp.float64)
    history = history.at[0].set(jnp.linalg.norm(initial - goals, axis=-1))
    carry = (initial, jnp.zeros((2, 2), jnp.float64), jnp.array(True),
             jnp.array(0, jnp.int32), jnp.array(-1, jnp.int32),
             jnp.array(-1, jnp.int32), history)

    def step(carry, inputs):
        positions, velocity, alive, terminal_code, since, first_deadlock, history = carry
        t, xi, offset = inputs

        def active_control(_):
            obs = observation(positions[None], velocity[None], goals)
            A, b, h = barrier_constraints(positions, walls,
                                           plant.to_dict(), cbf)
            prepared = field.prepare(
                params, obs, xi[None], A[None], b[None], plant.max_speed,
                lambda value, matrix, lower, speed:
                    controller_projection(value, matrix, lower))
            safe = prepared['safe'][0]
            w = safe + prepared['correction'][0] + offset
            targets = jnp.concatenate((w[None], WITNESS_TARGETS), axis=0)
            projected = controller_projection(
                targets, jnp.broadcast_to(A, (9,) + A.shape),
                jnp.broadcast_to(b, (9,) + b.shape))
            return (safe, w, projected[0], projected[1:], A, b, h)

        def inactive_control(_):
            nan4 = jnp.full((4,), jnp.nan)
            return (nan4, nan4, nan4, jnp.full((8, 4), jnp.nan),
                    jnp.full((17, 4), jnp.nan), jnp.full((17,), jnp.nan),
                    jnp.full((17,), jnp.nan))

        safe, w, applied, witnesses, A, b, h = jax.lax.cond(
            alive, active_control, inactive_control, operand=None)
        physical = jnp.where(alive, applied, jnp.zeros_like(applied))
        after = positions + plant.dt * physical.reshape(2, 2)
        errors = jnp.linalg.norm(after - goals, axis=-1)
        slot = (t + 1) % history_size
        history = history.at[slot].set(jnp.where(alive, errors,
                                                 history[slot]))
        start = jnp.maximum(0., t + 1 -
                            plant.progress_window_seconds / plant.dt)
        lo = jnp.floor(start).astype(jnp.int32) % history_size
        hi = jnp.ceil(start).astype(jnp.int32) % history_size
        (raw_code, updated_since, updated_first, raw_deadlock,
         candidate, raw_flags) = monitor_step(
             positions[None], physical.reshape(1, 2, 2),
             history[lo][None], history[hi][None], since[None],
             first_deadlock[None], alive[None], t, plant)
        code = jnp.where(alive, raw_code[0], terminal_code)
        next_alive = alive & (raw_code[0] == 0)
        output = dict(before=positions, after=after, safe=safe, w=w,
                      applied=applied, witnesses=witnesses, A=A, b=b, h=h,
                      alive_pre=alive, latch_pre=first_deadlock >= 0,
                      first_deadlock_pre=first_deadlock,
                      first_deadlock_after=updated_first[0],
                      candidate=candidate[0], raw_deadlock=raw_deadlock[0],
                      event_code=raw_code[0], raw_flags=raw_flags[0])
        next_velocity = jnp.where(next_alive, physical.reshape(2, 2),
                                  jnp.zeros((2, 2), jnp.float64))
        next_carry = (after, next_velocity, next_alive, code,
                      updated_since[0], updated_first[0], history)
        return next_carry, output

    final, trace = jax.lax.scan(
        jax.checkpoint(step), carry,
        (jnp.arange(850, dtype=jnp.int32), noise, action_offsets))
    final_positions, _, _, terminal_code, _, first_deadlock, _ = final
    action_count = jnp.sum(trace['alive_pre'])
    original_timeout = terminal_code == 5
    speeds = jnp.linalg.norm(trace['applied'][810:850].reshape(40, 2, 2),
                             axis=-1)
    errors_811 = jnp.linalg.norm(trace['after'][810] - goals, axis=-1)
    errors_850 = jnp.linalg.norm(trace['after'][849] - goals, axis=-1)
    stalled = (original_timeout & (action_count == 850) &
               jnp.all(speeds < .05) &
               jnp.all(jnp.abs(errors_850 - errors_811) < .02))
    historical_strict = first_deadlock >= 0
    risk = certificate_trajectory(
        trace['before'], trace['after'], trace['w'], trace['applied'],
        trace['witnesses'], goals, trace['alive_pre'], trace['latch_pre'],
        original_timeout, dt=plant.dt,
        goal_tolerance=plant.goal_tolerance,
        progress_epsilon=plant.progress_epsilon,
        strict_speed_limit=plant.max_speed * plant.speed_epsilon_fraction)
    def_sq = jnp.sum((trace['applied'] - trace['safe']) ** 2, axis=-1)
    j_def = (jnp.sum(jnp.where(trace['alive_pre'], def_sq, 0.)) /
             jnp.maximum(action_count, 1))
    terms = dict(**risk, J_def=j_def,
                 historical_strict_deadlock=historical_strict,
                 stalled_deadlock=stalled,
                 primary_deadlock=historical_strict | stalled,
                 terminal_safe_deadlock=terminal_code == 4,
                 success=terminal_code == 1,
                 wall_collision=terminal_code == 2,
                 agent_collision=terminal_code == 3,
                 original_other_timeout=original_timeout,
                 terminal_code=terminal_code, action_count=action_count,
                 first_deadlock_step=first_deadlock,
                 final_positions=final_positions)
    return terms, trace
