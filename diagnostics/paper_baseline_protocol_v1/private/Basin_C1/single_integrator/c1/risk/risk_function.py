"""R_risk_v0: piecewise differentiable per-agent planar cone geometry."""
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np

CONE_TYPES = ('empty', 'ray', 'line', 'wedge', 'half_plane', 'full_plane')

@dataclass(frozen=True)
class RiskConfig:
    rho: float = .05
    active_tol: float = 1e-7
    D0: float = .1
    kappa: float = 8.
    alpha: float = .5
    p: float = 4.
    smooth_max_temperature: float = .02  # legacy metadata only; max is exact

    def __post_init__(self):
        values = (self.rho, self.active_tol, self.D0, self.kappa, self.alpha, self.p)
        if not all(np.isfinite(v) for v in values):
            raise ValueError('nonfinite risk configuration')
        if self.rho < 0 or self.active_tol < 0 or self.D0 <= 0 or self.kappa <= 0 or not 0 <= self.alpha <= 1 or self.p < 1:
            raise ValueError('invalid risk configuration')

def _cross(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]

@jax.custom_jvp
def _norm(x):
    return jnp.sqrt(jnp.sum(x*x))

@_norm.defjvp
def _norm_jvp(primals, tangents):
    x, = primals
    dx, = tangents
    n = _norm(x)
    return n, jnp.where(n > 0, jnp.dot(x, dx)/jnp.maximum(n, 1e-30), 0.)

def cone_geometry(q, generators, active=None):
    """Circular gaps use valid directions only, including the wrap gap.
    Rank and angular coverage classify the cone. Selection is discrete;
    continuous selected boundary directions and query retain derivatives.
    """
    q, g = jnp.asarray(q), jnp.asarray(generators)
    if g.shape[0] == 0:
        return jnp.asarray(jnp.inf, q.dtype), jnp.asarray(0)
    squared = jnp.sum(g*g, axis=-1)
    lengths = jnp.sqrt(jnp.where(squared > 0, squared, 1.))
    valid = squared > 0
    if active is not None:
        valid = valid & jnp.asarray(active, bool)
    directions = g / lengths[:, None]
    # atan2(0,0) has undefined derivative: padding must stay innocuous.
    angle_directions = jnp.where(valid[:, None], directions, jnp.array([1., 0.]))
    angles = jnp.arctan2(angle_directions[:, 1], angle_directions[:, 0])
    order = jnp.argsort(jnp.where(valid, angles, jnp.inf))
    angles, directions = angles[order], directions[order]
    count = jnp.sum(valid.astype(jnp.int32))
    indices = jnp.arange(len(g))
    nxt = jnp.where(indices+1 < count, indices+1, 0)
    gaps = angles[nxt]-angles + jnp.where(indices == count-1, 2*jnp.pi, 0.)
    gaps = jnp.where(indices < count, gaps, -jnp.inf)
    end = jnp.argmax(gaps)
    start = jnp.where(end+1 < count, end+1, 0)
    a, b = directions[start], directions[end]
    omega = 2*jnp.pi-jnp.max(gaps)
    tol = 1e-7
    rank_one = jnp.max(jnp.where(indices < count, jnp.abs(_cross(directions[0], directions)), 0.)) < tol
    kind = jnp.where(count == 0, 0,
            jnp.where(rank_one, jnp.where(omega < jnp.pi/2, 1, 2),
            jnp.where(omega < jnp.pi-tol, 3, jnp.where(omega <= jnp.pi+tol, 4, 5))))
    kind = jax.lax.stop_gradient(kind).astype(jnp.int32)
    def ray_distance(ray):
        return _norm(q-jnp.maximum(jnp.dot(q, ray), 0.)*ray)
    def wedge():
        inside = (_cross(a, q) >= 0) & (_cross(q, b) >= 0)
        return jnp.where(inside, -jnp.minimum(_cross(a, q), _cross(q, b)),
                         jnp.minimum(ray_distance(a), ray_distance(b)))
    margin = jax.lax.switch(kind, (
        lambda: jnp.asarray(jnp.inf, q.dtype),
        lambda: ray_distance(a),
        lambda: _norm(q-jnp.dot(q, a)*a),
        wedge,
        lambda: -_cross(a, q),
        lambda: jnp.asarray(-jnp.inf, q.dtype)))
    return margin, kind

def signed_cone_margin(x, generators, active):
    return cone_geometry(jnp.asarray(x), generators, active)[0]

def risk_diagnostics(task_force, safety_force_blocks, active_mask, config=None):
    config = config or RiskConfig()
    task_force = jnp.asarray(task_force).reshape(2, 2)
    blocks = jnp.asarray(safety_force_blocks).reshape(-1, 2, 2)
    margins, kinds = jax.vmap(lambda q, g: cone_geometry(q, g, active_mask))(-task_force, blocks.transpose(1, 0, 2))
    finite_margin = jnp.where(jnp.isfinite(margins), margins, 0.)
    agent_risk = jax.nn.sigmoid(config.kappa*(1-finite_margin/config.D0))
    agent_risk = jnp.where(kinds == 0, 0., jnp.where(kinds == 5, 1., agent_risk))
    return dict(margins=margins, cone_types=kinds, agent_risk=agent_risk,
                risk=jnp.max(agent_risk), active_mask=active_mask)

def instantaneous_risk(task_force, safety_force_blocks, active_mask, config=None):
    d = risk_diagnostics(task_force, safety_force_blocks, active_mask, config)
    return d['risk'], d['margins']

def trajectory_risk(step_risk, config=None, step_mask=None):
    config = config or RiskConfig()
    r = jnp.asarray(step_risk)
    mask=jnp.ones_like(r) if step_mask is None else jnp.asarray(step_mask,r.dtype)
    count=jnp.maximum(jnp.sum(mask,axis=-1),1.)
    # Normalizing first also prevents tiny positive risks underflowing in r**p.
    scale=jnp.max(jnp.where(mask>0,jnp.abs(r),0.),axis=-1)
    safe_scale=jnp.where(scale>0,scale,1.)
    normalized=jnp.where(mask>0,r,0.)/safe_scale[...,None]
    moment = jnp.sum(normalized**config.p*mask,axis=-1)/count
    safe_moment = jnp.where(moment > 0, moment, 1.)
    root = jnp.where(moment > 0, safe_moment**(1/config.p), 0.)
    return config.alpha*scale*root+(1-config.alpha)*jnp.sum(jnp.where(mask>0,r,0.),axis=-1)/count
