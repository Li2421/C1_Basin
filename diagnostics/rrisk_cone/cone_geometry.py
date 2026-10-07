"""Exact small-dimensional cone geometry for the R_risk diagnostic only.

The controller never imports this module.  Rows in ``generators`` are rays of
the cone.  Signed distance uses the ordinary ambient topology, not a relative-
interior convention that would manufacture a negative interior for a
lower-dimensional cone.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import nnls


def _rank(matrix: np.ndarray, tolerance: float) -> int:
    matrix = np.asarray(matrix, np.float64)
    if matrix.size == 0:
        return 0
    singular = np.linalg.svd(matrix, compute_uv=False)
    return int(np.count_nonzero(singular > tolerance * max(1.0, singular[0])))


def cone_projection(vector, generators, zero_tolerance=1e-12):
    """Project ``vector`` onto a finitely generated closed convex cone."""
    vector = np.asarray(vector, np.float64).reshape(-1)
    generators = np.asarray(generators, np.float64).reshape(-1, len(vector))
    norms = np.linalg.norm(generators, axis=1)
    generators = generators[norms > zero_tolerance]
    if not len(generators):
        return np.zeros_like(vector), np.empty(0), float(np.linalg.norm(vector))
    coefficients, _ = nnls(generators.T, vector, maxiter=10000)
    projection = generators.T @ coefficients
    return projection, coefficients, float(np.linalg.norm(vector - projection))


def cone_facets(generators, facet_tolerance=1e-10,
                zero_tolerance=1e-12):
    """Unit inward facet normals for a full-dimensional polyhedral cone.

    In four dimensions every facet contains at least three independent
    generator directions.  Enumeration is deliberately exact and transparent
    because the diagnostic has at most 17 active safety rows.
    """
    generators = np.asarray(generators, np.float64)
    if generators.ndim != 2:
        raise ValueError("generators must be a matrix")
    dimension = generators.shape[1]
    norms = np.linalg.norm(generators, axis=1)
    generators = generators[norms > zero_tolerance]
    if _rank(generators, facet_tolerance) < dimension:
        return np.empty((0, dimension), np.float64)
    facets = []
    for indices in combinations(range(len(generators)), dimension - 1):
        subset = generators[list(indices)]
        if _rank(subset, facet_tolerance) != dimension - 1:
            continue
        _, _, vh = np.linalg.svd(subset, full_matrices=True)
        normal = vh[-1]
        normal /= np.linalg.norm(normal)
        dots = generators @ normal
        scale = max(1.0, float(np.max(np.abs(dots), initial=0.0)))
        tol = facet_tolerance * scale
        if np.all(dots >= -tol):
            pass
        elif np.all(dots <= tol):
            normal = -normal
            dots = -dots
        else:
            continue
        # It is a facet only if the generators on the hyperplane span d-1.
        if _rank(generators[np.abs(dots) <= tol], facet_tolerance) != dimension - 1:
            continue
        if not any(abs(float(old @ normal)) >= 1.0 - 1e-9 for old in facets):
            facets.append(normal)
    return (np.asarray(facets, np.float64).reshape(-1, dimension)
            if facets else np.empty((0, dimension), np.float64))


def lineality_dimension(generators, membership_tolerance=1e-9,
                        zero_tolerance=1e-12):
    """Dimension of the represented cone's lineality space."""
    generators = np.asarray(generators, np.float64)
    if generators.ndim != 2:
        raise ValueError("generators must be a matrix")
    directions = []
    for generator in generators:
        if np.linalg.norm(generator) <= zero_tolerance:
            continue
        _, _, distance = cone_projection(-generator, generators,
                                          zero_tolerance)
        if distance <= membership_tolerance:
            directions.append(generator)
    return _rank(np.asarray(directions).reshape(-1, generators.shape[1])
                 if directions else np.empty((0, generators.shape[1])),
                 membership_tolerance)


def signed_cone_margin(vector, generators, membership_tolerance=1e-9,
                       facet_tolerance=1e-10, zero_tolerance=1e-12):
    """Ambient signed distance with positive-outside convention.

    ``margin`` is finite except for a point inside the full-space cone, whose
    boundary is empty.  In that case ``margin`` is ``None`` and
    ``extended_margin`` is ``-inf``.  This avoids silently assigning a large
    finite risk to a mathematical degeneracy.
    """
    vector = np.asarray(vector, np.float64).reshape(-1)
    generators = np.asarray(generators, np.float64).reshape(-1, len(vector))
    norms = np.linalg.norm(generators, axis=1)
    generators = generators[norms > zero_tolerance]
    dimension = len(vector)
    rank = _rank(generators, membership_tolerance)
    lineality = lineality_dimension(generators, membership_tolerance,
                                    zero_tolerance) if len(generators) else 0
    projection, coefficients, distance = cone_projection(
        vector, generators, zero_tolerance)
    inside = distance <= membership_tolerance
    base = dict(
        dimension=dimension, generator_count=int(len(generators)), rank=rank,
        lineality_dimension=lineality, pointed=bool(lineality == 0),
        projection=projection.tolist(), coefficients=coefficients.tolist(),
        distance_to_cone=distance, inside=bool(inside),
        low_dimensional=bool(rank < dimension), full_space=False,
        facet_count=0, facet_normals=[], nearest_facet_indices=[],
        boundary_distance=None, extended_margin=None,
    )
    if not len(generators):
        margin = 0.0 if inside else distance
        return dict(base, status=("empty_active_apex" if inside
                                  else "empty_active_outside"), margin=margin,
                    risk=-margin, boundary=bool(inside))
    if not inside:
        return dict(base, status="outside", margin=distance, risk=-distance,
                    boundary=False)
    if rank < dimension:
        # A proper lower-dimensional closed cone has empty ambient interior.
        return dict(base, status="boundary_low_dimensional", margin=0.0,
                    risk=-0.0, boundary=True)
    facets = cone_facets(generators, facet_tolerance, zero_tolerance)
    base["facet_count"] = int(len(facets))
    base["facet_normals"] = facets.tolist()
    if not len(facets):
        # A full-dimensional finitely generated cone with no supporting facet
        # is R^d.  Its topological boundary is empty.
        return dict(base, status="full_space", margin=None, risk=None,
                    extended_margin="-inf", extended_risk="+inf",
                    full_space=True, boundary=False)
    distances = facets @ vector
    distances[np.abs(distances) <= membership_tolerance] = 0.0
    boundary_distance = float(max(0.0, np.min(distances)))
    nearest = np.flatnonzero(np.isclose(
        distances, boundary_distance, atol=membership_tolerance, rtol=1e-7))
    margin = -boundary_distance
    return dict(base, status=("boundary" if boundary_distance == 0.0
                              else "inside"), margin=margin, risk=-margin,
                boundary=bool(boundary_distance == 0.0),
                boundary_distance=boundary_distance,
                nearest_facet_indices=nearest.astype(int).tolist())


def projection_kkt_audit(nominal, projected, A, b, speed=0.5,
                         active_tolerance=1e-7):
    """Reconstruct multipliers for the exact joint projection KKT system.

    Linear constraints use ``A u >= b``.  A speed ball is represented at its
    boundary by the equivalent tangent row ``-u_i/||u_i||`` in the same
    greater-than orientation.  Thus ``p-y = J_active.T lambda``.
    """
    nominal = np.asarray(nominal, np.float64).reshape(-1)
    projected = np.asarray(projected, np.float64).reshape(-1)
    A = np.asarray(A, np.float64).reshape(-1, len(nominal))
    b = np.asarray(b, np.float64).reshape(-1)
    linear_slack = A @ projected - b
    active_linear = np.flatnonzero(linear_slack <= active_tolerance)
    blocks = projected.reshape(-1, 2)
    block_speed = np.linalg.norm(blocks, axis=1)
    speed_slack = speed - block_speed
    active_speed = np.flatnonzero(speed_slack <= active_tolerance)
    rows = [A[active_linear]]
    labels = [("safety", int(i)) for i in active_linear]
    for agent in active_speed:
        row = np.zeros_like(projected)
        if block_speed[agent] > 0:
            row[2 * agent:2 * agent + 2] = -blocks[agent] / block_speed[agent]
            rows.append(row[None])
            labels.append(("speed", int(agent)))
    jacobian = (np.concatenate(rows, axis=0) if rows
                else np.empty((0, len(projected))))
    gradient = projected - nominal
    if len(jacobian):
        multipliers, _ = nnls(jacobian.T, gradient, maxiter=10000)
        residual = gradient - jacobian.T @ multipliers
    else:
        multipliers = np.empty(0)
        residual = gradient
    return dict(
        linear_slack=linear_slack.tolist(),
        speed_slack=speed_slack.tolist(),
        active_linear=active_linear.astype(int).tolist(),
        active_speed=active_speed.astype(int).tolist(),
        active_labels=[[kind, index] for kind, index in labels],
        multipliers=multipliers.tolist(),
        stationarity_inf=float(np.linalg.norm(residual, ord=np.inf)),
        projection_residual=(nominal - projected).tolist(),
        projection_residual_norm=float(np.linalg.norm(nominal - projected)),
        nominal_norm=float(np.linalg.norm(nominal)),
        projected_norm=float(np.linalg.norm(projected)),
        min_linear_slack=float(np.min(linear_slack, initial=np.inf)),
        max_speed=float(np.max(block_speed, initial=0.0)),
    )


def self_test():
    eye = np.eye(4)
    inside = signed_cone_margin(np.ones(4), eye)
    outside = signed_cone_margin(np.array([1., 1., 1., -2.]), eye)
    boundary = signed_cone_margin(np.array([1., 1., 1., 0.]), eye)
    assert inside["status"] == "inside" and np.isclose(inside["margin"], -1.)
    assert outside["status"] == "outside" and np.isclose(outside["margin"], 2.)
    assert boundary["status"] == "boundary" and boundary["margin"] == 0.
    ray = signed_cone_margin(np.array([2., 0., 0., 0.]), eye[:1])
    assert ray["status"] == "boundary_low_dimensional" and ray["margin"] == 0.
    off_ray = signed_cone_margin(np.array([2., 3., 0., 0.]), eye[:1])
    assert off_ray["status"] == "outside" and np.isclose(off_ray["margin"], 3.)
    full = signed_cone_margin(np.ones(4), np.r_[eye, -eye])
    assert full["status"] == "full_space" and full["extended_margin"] == "-inf"
    empty = signed_cone_margin(np.ones(4), np.empty((0, 4)))
    assert empty["status"] == "empty_active_outside" and np.isclose(empty["margin"], 2.)
    kkt = projection_kkt_audit([-1., 0., 0., 0.], [0., 0., 0., 0.],
                               eye[:1], [0.])
    assert kkt["active_linear"] == [0] and kkt["stationarity_inf"] < 1e-12


if __name__ == "__main__":
    self_test()
    print("cone_geometry self-test: PASS")
