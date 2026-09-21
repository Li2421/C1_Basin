"""Full-deadline V3 residual rollout; legacy joint rollout remains frozen."""
import math
import jax
import jax.numpy as jnp
import numpy as np

from single_integrator.c1.differentiable_rollout import barrier_constraints
from single_integrator.c1.train import observation
from single_integrator.c1.termination import event_step
from single_integrator.c1.risk.joint_frozen import controller_projection, geometry
from single_integrator.c1.risk.risk_v3 import trajectory
from single_integrator.environment import GiveWayEnv


def rollout(params, field, initial, noise, plant, cbf, prefix_steps=100):
    if (plant.dt != .05 or plant.max_speed != .5 or plant.goal_tolerance != .08
            or plant.terminate_on_deadlock or not plant.terminate_on_success
            or not plant.terminate_on_collision or plant.max_steps != 850):
        raise ValueError('V3 requires the frozen 42.5s joint-risk environment')
    if noise.shape != (plant.max_steps, 4) or not 0 <= prefix_steps < plant.max_steps:
        raise ValueError('V3 requires full-horizon noise and a valid prefix')
    env = GiveWayEnv(plant)
    initial = jnp.asarray(initial, jnp.float64)
    def check_start(x):
        GiveWayEnv(plant).reset(x)
    jax.debug.callback(check_start, jax.lax.stop_gradient(initial))
    goals, walls = jnp.asarray(env.goals), jnp.asarray(env.walls)
    window = plant.progress_window_seconds/plant.dt
    size = int(math.ceil(window))+1
    positions = initial[None]
    history = jnp.zeros((1, size, 2)).at[:, 0].set(jnp.linalg.norm(positions-goals, axis=-1))
    carry = (positions, jnp.zeros_like(positions), jnp.ones(1, bool),
             jnp.full((1,), -1, jnp.int32), history)

    def step(carry, inputs):
        positions, velocity, alive, since, history = carry
        t, xi = inputs
        # Finished episodes use a safe dummy state for unused solver calls.
        control_positions = jnp.where(alive[:, None, None], positions, initial[None])
        obs = observation(control_positions, velocity, goals)
        A, b, _ = jax.vmap(lambda x: barrier_constraints(x, walls, plant.to_dict(), cbf))(control_positions)
        def control(_):
            prepared = field.prepare(params, obs, xi[None], A, b, plant.max_speed,
                                     lambda v, a, b, s: controller_projection(v, a, b))
            candidate = prepared['safe'] + jnp.where(t >= prefix_steps, prepared['correction'], 0.)
            return candidate, controller_projection(candidate, A, b), prepared['safe']
        candidate, applied, safe = jax.lax.cond(alive[0], control,
            lambda _: (jnp.zeros((1, 4)),)*3, operand=None)
        after = positions + plant.dt*applied.reshape(1, 2, 2)
        history = history.at[:, (t+1) % size].set(jnp.linalg.norm(after-goals, axis=-1))
        start = jnp.maximum(0., t+1-window)
        lo, hi = jnp.floor(start).astype(jnp.int32), jnp.ceil(start).astype(jnp.int32)
        codes, since = event_step(positions, applied.reshape(1, 2, 2),
            history[:, lo % size], history[:, hi % size], since, alive, t, plant)
        def check_event(c):
            if np.any((c == 2) | (c == 3)):
                raise RuntimeError('V3 collision: reject rollout; do not assign a favorable score')
        jax.debug.callback(check_event, codes)
        success = (codes == 1) | ~alive
        next_alive = alive & (codes == 0)
        output = (positions[0], after[0], candidate[0], applied[0], safe[0],
                  A[0], b[0], success[0], alive[0], codes[0])
        return (after, jnp.where(next_alive[:, None, None], applied.reshape(1, 2, 2), 0.),
                next_alive, since, history), output

    _, data = jax.lax.scan(jax.checkpoint(step), carry,
                         (jnp.arange(plant.max_steps, dtype=jnp.int32), noise))
    before, after, candidate, applied, safe, A, b, success, alive, codes = data
    # Mask before geometry evaluation to avoid an undefined zero query after
    # success. Active zero queries retain the original explicit undefined g.
    query = jnp.where(alive[prefix_steps:, None], candidate[prefix_steps:], .1)
    g = geometry(query, A[prefix_steps:], b[prefix_steps:])
    terms = trajectory(before, after, g, goals, success, prefix_steps, plant.dt)
    mask = alive[prefix_steps:]
    deviation = jnp.sum(jnp.where(mask,
        jnp.sum((applied[prefix_steps:]-safe[prefix_steps:])**2, axis=-1), 0.))/jnp.maximum(mask.sum(), 1)
    return dict(J_live=terms['J_live'], J_def=deviation, P=terms['P'], g=terms['g'],
                success=jnp.any(codes == 1), steps=jnp.sum(alive)), dict(
        before=before, after=after, candidate=candidate, applied=applied,
        safe=safe, success=success, alive=alive, codes=codes, g=g)
