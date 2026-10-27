"""Prototype: same Clarabel problem, cheaper CSC construction. Checks matrices and trajectories are bit-identical."""
import sys, os, json, time
os.environ['JAX_PLATFORMS'] = 'cpu'
H = '/home/zhihan/.claude/jobs/7eda8ddd/tmp/scene_search/harness'; sys.path.insert(0, H)
import bootstrap  # noqa
from common import actions
from design import controllers
from scenes import make_new
from execution import execute
import numpy as np, clarabel
from scipy import sparse
import shared_control.hard_projection as HP
import swap_safety
ORIG = HP.project_velocity
_P = {}
def fast_project_velocity(nominal, matrix, lower, max_speed, config, check=False):
    nominal = np.asarray(nominal, dtype=np.float64)
    if nominal.ndim != 2 or nominal.shape[1] != 2: raise ValueError("nominal must have shape [N,2]")
    num_agents = len(nominal); target = nominal.reshape(-1)
    matrix = np.asarray(matrix, dtype=np.float64); lower = np.asarray(lower, dtype=np.float64)
    if not all(np.isfinite(v).all() for v in (target, matrix, lower)): raise ValueError("nonfinite projection input")
    if matrix.shape != (len(lower), 2 * num_agents) or not np.isfinite(max_speed) or max_speed <= 0: raise ValueError("invalid projection dimensions or speed limit")
    if matrix.size and np.min(matrix @ target - lower) >= 0 and np.max(np.linalg.norm(nominal, axis=-1)) <= max_speed:
        d = HP._diagnostics(target, target, matrix, lower, max_speed)
        d.update(backend="clarabel_exact_socp", solver_status="nominal_feasible", accepted=True, conic_iterations=0, projection_input=target.tolist())
        return nominal.copy(), "nominal_feasible", d
    m = len(lower); n = 2 * num_agents; neg = -matrix
    data = []; idx = []; ptr = [0]
    for j in range(n):                      # column-major, rows increasing: linear block, then the agent's speed-cone rows
        r = np.flatnonzero(neg[:, j] != 0); data.append(neg[r, j]); idx.append(r)
        agent, k = divmod(j, 2); data.append(np.array([-1.0])); idx.append(np.array([m + 3 * agent + 1 + k]))
        ptr.append(ptr[-1] + len(r) + 1)
    cone_matrix = sparse.csc_matrix((np.concatenate(data), np.concatenate(idx).astype(np.int32), np.asarray(ptr, np.int32)), shape=(m + 3 * num_agents, n))
    if check:   # identical arrays to the original construction
        blocks = [sparse.csc_matrix(-matrix)]
        for agent in range(num_agents):
            b = np.zeros((3, n)); b[1:, 2 * agent:2 * agent + 2] = -np.eye(2); blocks.append(sparse.csc_matrix(b))
        ref = sparse.vstack(blocks, format="csc")
        assert ref.shape == cone_matrix.shape and np.array_equal(ref.indptr, cone_matrix.indptr) and np.array_equal(ref.indices, cone_matrix.indices) \
            and np.array_equal(ref.data, cone_matrix.data) and ref.indices.dtype == cone_matrix.indices.dtype and ref.data.dtype == cone_matrix.data.dtype, 'CSC mismatch'
    cones = [clarabel.NonnegativeConeT(m)] + [clarabel.SecondOrderConeT(3) for _ in range(num_agents)]
    cone_rhs = np.concatenate([-lower] + [np.asarray([max_speed, 0.0, 0.0])] * num_agents)
    if n not in _P: e = sparse.csc_matrix(np.eye(n)); _P[n] = (e.data.copy(), e.indices.copy(), e.indptr.copy())
    P = sparse.csc_matrix((_P[n][0].copy(), _P[n][1].copy(), _P[n][2].copy()), shape=(n, n))
    s = clarabel.DefaultSettings(); s.verbose = False; s.max_iter = config.max_iterations
    s.tol_feas = min(config.feasibility_tol, config.speed_tol); s.tol_gap_abs = config.solver_ftol; s.tol_gap_rel = config.solver_ftol
    sol = clarabel.DefaultSolver(P, -target, cone_matrix, cone_rhs, cones, s).solve()
    cand = np.asarray(sol.x, dtype=np.float64); d = HP._diagnostics(cand, target, matrix, lower, max_speed)
    dual = np.asarray(sol.z, dtype=np.float64); slack = np.asarray(sol.s, dtype=np.float64)
    stat = float(np.linalg.norm(cand - target + cone_matrix.T @ dual, ord=np.inf)); comp = float(abs(np.dot(slack, dual))); status = str(sol.status)
    d.update(backend="clarabel_exact_socp", solver_status=status, accepted=False, conic_iterations=int(sol.iterations), primal_residual=float(sol.r_prim),
             dual_residual=float(sol.r_dual), stationarity_residual=stat, complementarity_residual=comp, projection_input=target.tolist())
    ok = np.isfinite(cand).all() and d["min_linear_residual"] >= -config.feasibility_tol and d["max_speed_excess"] <= config.speed_tol and max(stat, comp) <= config.optimality_tol
    if not ok: raise HP.CBFSolverError("conic_failed", d)
    d["accepted"] = True
    return cand.reshape(num_agents, 2).copy(), "solved" if status == "Solved" else "solved_kkt_certified", d
