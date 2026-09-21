"""Progress-debt upper certificates; experimental, not a trained solution.

The default potential is Euclidean distance to the joint goal set in seconds.
Its Lipschitz constant under max-agent displacement is 2 / max_speed.
Do not substitute a navigation potential without supplying its proven bound.
"""
import math
import jax.numpy as jnp


def goal_potential(positions, goals, tolerance=.08, max_speed=.5):
    distance = jnp.sqrt(jnp.maximum(jnp.sum((positions-goals)**2, axis=-1), 1e-30))
    return jnp.sum(jnp.maximum(distance-tolerance, 0.), axis=-1)/max_speed


def certificate(potential, alive, *, dt=.05, target_rate=.4,
                lipschitz=4., low_speed=.025, hold_seconds=5.):
    """Return separate bounds and their maximum for a fixed deadline rollout.

    potential: [T+1], nonnegative, continuous-state potential, goal absorbed.
    alive: [T], true for every action up to and including success; a failure
    must be observed to the deadline. No deadlock early termination is allowed.
    Preconditions (caller responsibility): potential satisfies the supplied
    Lipschitz bound; trajectories obey SI dynamics; success is absorbing.
    Negative certificate margins are rejected rather than clipped to epsilon.
    """
    if potential.ndim != 1 or alive.shape != (len(potential)-1,):
        raise ValueError('expected potential[T+1] and alive[T]')
    if dt <= 0 or hold_seconds <= 0 or target_rate <= lipschitz*low_speed:
        raise ValueError('positive deadlock certificate margin required')
    count = int(math.ceil(hold_seconds/dt-1e-10))+1
    if len(alive) < count:
        raise ValueError('horizon cannot contain the detector hold interval')
    horizon = len(alive)*dt
    slack = target_rate*horizon-potential[0]
    # NaN deliberately makes an invalid deadline certificate unoptimizable.
    denominator = jnp.where(slack > 0, slack, jnp.nan)
    debt = jnp.where(alive, jnp.maximum(target_rate*dt+jnp.diff(potential), 0.), 0.)
    prefix = jnp.concatenate([jnp.zeros(1, debt.dtype), jnp.cumsum(debt)])
    windows = prefix[count:]-prefix[:-count]
    deadline = jnp.sum(debt)/denominator
    deadlock = jnp.max(windows)/((target_rate-lipschitz*low_speed)*count*dt)
    return dict(J_live=jnp.maximum(deadline, deadlock), timeout_bound=deadline,
                deadlock_bound=deadlock, debt=debt, deadline_slack=slack)


def trajectory(before, after, goals, alive, *, dt=.05, max_speed=.5,
               goal_tolerance=.08, target_rate=.4, hold_seconds=5.,
               speed_epsilon_fraction=.05):
    if before.shape != after.shape or after.ndim != 3 or after.shape[1:] != (2, 2):
        raise ValueError('expected before and after[T,2,2]')
    positions = jnp.concatenate([before[:1], after], axis=0)
    potential = goal_potential(positions, goals, goal_tolerance, max_speed)
    result = certificate(potential, alive, dt=dt, target_rate=target_rate,
                         lipschitz=2/max_speed,
                         low_speed=max_speed*speed_epsilon_fraction,
                         hold_seconds=hold_seconds)
    return {**result, 'potential': potential}
