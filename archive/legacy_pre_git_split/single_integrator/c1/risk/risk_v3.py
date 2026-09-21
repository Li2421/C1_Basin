"""V3 candidate: unchanged joint g and progress P over the full task horizon.

This is a new objective version, not a numerical rewrite of the old 5--15s
score. Success has absorbing zero cost. Deadlock remains recoverable, matching
the joint-risk experiments. The fixed denominator penalizes prolonged stalls;
it is not the legacy v2 first-event normalization or a liveness certificate.
"""
import jax
import jax.numpy as jnp


def progress_steps(before, after, goals, success, dt=.05, window=80):
    if before.shape != after.shape or after.ndim != 3 or after.shape[1:] != (2, 2):
        raise ValueError('positions must have shape [T,2,2]')
    if success.shape != (len(after),) or dt <= 0 or window < 1:
        raise ValueError('invalid progress support')
    # Same distance, unfinished mask, physical rate and thresholds as the
    # frozen joint P. Safe norms choose a finite zero subgradient at goals.
    norm = lambda x: jnp.sqrt(jnp.maximum(jnp.sum(x*x, axis=-1), 1e-30))
    initial = norm(before[0]-goals).sum()
    distances = norm(after-goals)
    potential = jnp.concatenate([initial[None], distances.sum(-1)])/(initial+1e-5)
    end = jnp.arange(1, len(after)+1)
    start = jnp.maximum(end-window, 0)
    rate = (potential[start]-potential[end])/((end-start)*dt)
    unfinished = 1-jnp.prod(1-jax.nn.sigmoid((distances-.08)/.02), axis=-1)
    # The successful action is included; only subsequent actions are absorbed.
    alive = jnp.concatenate([jnp.ones(1, bool), ~jnp.maximum.accumulate(success)[:-1]])
    return jnp.where(alive, unfinished*jax.nn.sigmoid((.005-rate)/.00125), 0.), alive


def trajectory(before, after, g, goals, success, score_start=100, dt=.05):
    if not 0 <= score_start < len(after) or g.shape != (len(after)-score_start,):
        raise ValueError('g must cover every action from score_start to the deadline')
    p, alive = progress_steps(before, after, goals, success, dt)
    geometric = jnp.where(alive[score_start:], g, 0.)
    P, G = jnp.mean(p[score_start:]), jnp.mean(geometric)
    return dict(P=P, g=G, J_live=P+G, progress_t=p, alive=alive)
