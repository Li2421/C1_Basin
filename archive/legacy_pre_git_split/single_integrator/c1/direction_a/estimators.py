"""Likelihood-ratio estimators for the frozen Direction A objective."""

import jax
import jax.numpy as jnp
import numpy as np

from .policy import log_probability_components


COMPONENTS = ("gate", "mean", "scale")


def _frozen_records(h_safe, s, observations, gate, residual, weights):
    values = (h_safe, s, observations, gate, residual, weights)
    return jax.tree_util.tree_map(lambda x: jax.lax.stop_gradient(jnp.asarray(x)),
                                  values)


def weighted_score_trees(model, params, h_safe, s, observations, gate,
                         residual, weights, distribution):
    """Return gate/mean/scale parameter scores for fixed sampled history."""
    h_safe, s, observations, gate, residual, weights = _frozen_records(
        h_safe, s, observations, gate, residual, weights)
    if h_safe.ndim != 2 or h_safe.shape[-1] != 4:
        raise ValueError("h_safe must have shape [T,4]")
    length = len(h_safe)
    if (s.shape != (length, 1) or observations.shape[0] != length
            or gate.shape != (length,) or residual.shape != (length, 4)
            or weights.shape != (length,)):
        raise ValueError("inconsistent trajectory record shapes")

    def totals(candidate_params):
        outputs = model.apply(candidate_params, h_safe, s, observations)
        terms = log_probability_components(outputs, gate, residual, distribution)
        return jnp.stack([jnp.sum(weights*term) for term in terms])

    jacobian = jax.jacrev(totals)(params)
    trees = tuple(jax.tree_util.tree_map(lambda x, i=i: x[i], jacobian)
                  for i in range(3))
    total = jax.tree_util.tree_map(lambda *xs: sum(xs), *trees)
    return dict(zip(COMPONENTS, trees)), total


def trajectory_scores(model, params, h_safe, s, observations, gate, residual,
                      pre_action_latch, distribution):
    """Return S_full and latch-truncated S_D without length normalization."""
    latch = np.asarray(pre_action_latch, dtype=bool)
    if latch.shape != (len(h_safe),):
        raise ValueError("pre_action_latch must have shape [T]")
    full_components, full = weighted_score_trees(
        model, params, h_safe, s, observations, gate, residual,
        np.ones(len(h_safe)), distribution)
    risk_components, risk = weighted_score_trees(
        model, params, h_safe, s, observations, gate, residual,
        (~latch).astype(float), distribution)
    return dict(full=full, full_components=full_components,
                risk=risk, risk_components=risk_components)


def deformation_costs(applied, safe, metric, *, dt=0.05):
    """Physical deformation cost at the actual state and Flow realization."""
    applied, safe, metric = map(jnp.asarray, (applied, safe, metric))
    if applied.shape != safe.shape or applied.ndim != 2 or applied.shape[-1] != 4:
        raise ValueError("applied and safe must both have shape [T,4]")
    if metric.shape != (4, 4) or not np.allclose(np.asarray(metric),
                                                np.asarray(metric).T):
        raise ValueError("deformation metric must be symmetric [4,4]")
    delta = applied-safe
    return dt*jnp.einsum("ti,ij,tj->t", delta, metric, delta)


def deformation_estimate(model, params, h_safe, s, observations, gate,
                         residual, applied, metric, distribution, *, dt=0.05):
    """Return J_def and its reward-to-go likelihood-ratio score."""
    costs = deformation_costs(applied, h_safe, metric, dt=dt)
    returns = jnp.flip(jnp.cumsum(jnp.flip(costs)))
    components, score = weighted_score_trees(
        model, params, h_safe, s, observations, gate, residual,
        returns, distribution)
    return dict(value=jnp.sum(costs), costs=costs, returns=returns,
                score=score, score_components=components)


def tree_dot(left, right):
    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda x, y: jnp.vdot(x, y), left, right))
    return sum(leaves, jnp.asarray(0.0))


def tree_squared_norm(tree):
    return tree_dot(tree, tree)


def pilot_baseline(events, risk_score_vectors):
    """Independent pilot baseline c_P; repeated scenarios are averaged first."""
    events = np.asarray(events, dtype=float)
    scores = np.asarray(risk_score_vectors, dtype=float)
    if events.ndim != 2 or scores.ndim != 3 or scores.shape[:2] != events.shape:
        raise ValueError("expected events[S,R] and scores[S,R,P]")
    norm2 = np.sum(scores*scores, axis=-1)
    # Each independent scenario contributes one equally weighted continuation
    # average to both sums.  Keep D and ||S_D||^2 paired within a continuation.
    numerator_means = (events*norm2).mean(axis=1)
    norm2_means = norm2.mean(axis=1)
    denominator = float(np.sum(norm2_means))
    if denominator == 0.0:
        return None
    return float(np.sum(numerator_means)/denominator)


def risk_gradient_contributions(events, risk_score_vectors, pilot_constant):
    """Return per-continuation G and the equal-scenario gradient estimate."""
    events = np.asarray(events, dtype=float)
    scores = np.asarray(risk_score_vectors, dtype=float)
    if events.ndim != 2 or scores.ndim != 3 or scores.shape[:2] != events.shape:
        raise ValueError("expected events[S,R] and scores[S,R,P]")
    if not np.isfinite(pilot_constant):
        raise ValueError("pilot baseline must be a fixed finite scalar")
    contributions = (events-pilot_constant)[..., None]*scores
    scenario_means = contributions.mean(axis=1)
    return dict(contributions=contributions, scenario_means=scenario_means,
                gradient=scenario_means.mean(axis=0))
