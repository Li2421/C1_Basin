"""Historical nominal-input Stage-1 deadlock-geometry instrumentation.

For the current single-integrator plant, ``f = 0`` and ``g = I``.  Thus the
TRO no-CLF task force is ``F_V = f + g * u_nominal = u_nominal``.  It is
deliberately the bounded nominal Flow-BC reference input to the CBF filter,
not the projected hard-safe velocity.

This historical extractor is not the active C1 training risk path. Current
C1 uses the pre-final-projection candidate (u_safe + G) as task input, and
computes its geometry with the differentiable barrier/risk modules. Callers
of this nominal-input diagnostic must label that convention explicitly.
"""
from dataclasses import asdict, dataclass

import numpy as np

from single_integrator.cbf import barrier_constraints


@dataclass(frozen=True)
class DeadlockGeometryConfig:
    """Numerical selection settings for \(H_\rho\).

    A scalar rho is intentionally applied to every CBF constraint in this
    first instrumentation pass.  The returned ``rho`` vector keeps the
    per-constraint representation required for a later heterogeneous choice.
    """

    rho: float = 0.05
    active_tol: float = 1e-7

    def __post_init__(self):
        for name in ("rho", "active_tol"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")

    def to_dict(self):
        return asdict(self)


def _constraint_identity():
    """Stable identities matching ``single_integrator.cbf.barrier_constraints``.

    Row zero is the joint pairwise CBF.  Rows 1--16 are the eight finite wall
    segment CBFs for agent 0 followed by those for agent 1.
    """
    labels = [("pairwise", -1, -1)]
    labels.extend(("wall", agent, wall) for agent in range(2) for wall in range(8))
    return labels


def extract_deadlock_geometry(snapshot, nominal_velocity, safe_velocity, cbf_config,
                              config=None):
    """Return confirmed C1 geometry at one pre-action state.

    The CBF constraints are \(A_i u \ge b_i\).  Their exact values are:

    * pairwise: \(h_0=\|p_0-p_1\|^2-d_{safe}^2\),
      \(\nabla h_0=[2(p_0-p_1),-2(p_0-p_1)]\);
    * wall row \((a,j)\): \(h_{a,j}=d(p_a,\mathrm{segment}_j)-r_a-r_w-
      m_w-\text{buffer}\), with the gradient in agent \(a\)'s two control
      coordinates equal to the outward unit vector from its nearest segment
      point and zeros in the other agent block.

    Since \(f=0\), \(g=I\), and the CBF implementation uses
    \(\alpha_i(h_i)=\gamma_i h_i\), the residual used for activity is
    \(\mathrm{CBFResidual}_i=A_i u_{safe}-b_i= L_fh_i+L_gh_i u_{safe}+
    \alpha_i(h_i)\).  A constraint enters \(H_\rho\) iff
    \(h_i\le\rho_i\) and \(|\mathrm{CBFResidual}_i|\le\texttt{active_tol}\).

    ``safety_force_blocks`` has shape ``[17,2,2]``.  Each row retains the two
    per-agent control blocks of \(F_{h_i}=G\nabla h_i=\nabla h_i\), with
    ``G=I``.  Selected generators have shape ``[K,2,2]``.  No numerical cone
    distance is generated when ``K == 0``; the explicit state label is
    ``completely_deadlock_free``.
    """
    config = config or DeadlockGeometryConfig()
    nominal = np.asarray(nominal_velocity, dtype=np.float64)
    safe = np.asarray(safe_velocity, dtype=np.float64)
    if nominal.shape != (2, 2) or safe.shape != (2, 2):
        raise ValueError("Expected nominal and hard-safe joint velocities [2,2]")
    if not np.isfinite(nominal).all() or not np.isfinite(safe).all():
        raise ValueError("Nonfinite nominal or hard-safe velocity")

    A, b, geometry = barrier_constraints(snapshot, cbf_config)
    identities = _constraint_identity()
    if A.shape != (17, 4) or b.shape != (17,) or len(identities) != len(b):
        raise AssertionError("Unexpected current two-agent CBF constraint layout")

    h = np.concatenate(([geometry["pairwise_h"]], geometry["wall_h"].reshape(-1)))
    gradients = A.reshape(17, 2, 2)
    residual = A @ safe.reshape(4) - b
    rho = np.full(17, config.rho, dtype=np.float64)
    selected = (h <= rho) & (np.abs(residual) <= config.active_tol)
    if not all(np.isfinite(array).all() for array in (h, gradients, residual, rho)):
        raise FloatingPointError("Nonfinite C1 deadlock geometry")

    # f=0, g=I: F_V is exactly the bounded nominal reference velocity.
    task_force = nominal.copy()
    safety_force_blocks = gradients.copy()  # F_hi = G grad(h_i), G = I.
    selected_indices = np.flatnonzero(selected)
    selected_generators = safety_force_blocks[selected_indices]
    selected_identity = np.asarray([identities[i] for i in selected_indices], dtype=object)
    empty = len(selected_indices) == 0
    return dict(
        task_force=task_force,
        h=h,
        cbf_residual=residual,
        rho=rho,
        safety_force_blocks=safety_force_blocks,
        constraint_kind=np.asarray([x[0] for x in identities]),
        constraint_agent=np.asarray([x[1] for x in identities], dtype=np.int64),
        constraint_wall=np.asarray([x[2] for x in identities], dtype=np.int64),
        active_mask=selected,
        active_indices=selected_indices,
        active_generators=selected_generators,
        active_constraint_kind=np.asarray([x[0] for x in selected_identity]),
        active_constraint_agent=np.asarray([x[1] for x in selected_identity], dtype=np.int64),
        active_constraint_wall=np.asarray([x[2] for x in selected_identity], dtype=np.int64),
        completely_deadlock_free=empty,
        cone_distance_defined=False,
        cone_distance=None,
    )
