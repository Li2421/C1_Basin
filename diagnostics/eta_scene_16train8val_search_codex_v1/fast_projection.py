"""Exact CSC construction; unchanged solver, tolerances, diagnostics, exceptions."""
import numpy as np,clarabel
from scipy import sparse
from shared_control.hard_projection import HardProjectionConfig,CBFSolverError,_diagnostics,project_velocity as _original
import accel_mode
_P={}
def _fast_project_velocity(
    nominal: np.ndarray,
    matrix: np.ndarray,
    lower: np.ndarray,
    max_speed: float,
    config: HardProjectionConfig,
) -> tuple[np.ndarray, str, dict]:
    """Project onto all linear CBF rows and one 2-D speed SOC per agent."""
    nominal = np.asarray(nominal, dtype=np.float64)
    if nominal.ndim != 2 or nominal.shape[1] != 2:
        raise ValueError("nominal must have shape [N,2]")
    num_agents = len(nominal)
    target = nominal.reshape(-1)
    matrix = np.asarray(matrix, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    if not all(np.isfinite(value).all() for value in (target, matrix, lower)):
        raise ValueError("nonfinite projection input")
    if matrix.shape != (len(lower), 2 * num_agents) or not np.isfinite(max_speed) or max_speed <= 0:
        raise ValueError("invalid projection dimensions or speed limit")

    if matrix.size and np.min(matrix @ target - lower) >= 0 and np.max(
        np.linalg.norm(nominal, axis=-1)
    ) <= max_speed:
        diagnostics = _diagnostics(target, target, matrix, lower, max_speed)
        diagnostics.update(
            backend="clarabel_exact_socp",
            solver_status="nominal_feasible",
            accepted=True,
            conic_iterations=0,
            projection_input=target.tolist(),
        )
        return nominal.copy(), "nominal_feasible", diagnostics

    m=len(lower); n=2*num_agents; neg=-matrix; data=[]; indices=[]; ptr=[0]
    for j in range(n):
        rows=np.flatnonzero(neg[:,j]!=0)
        data.extend((neg[rows,j],np.asarray([-1.])))
        indices.extend((rows,np.asarray([m+3*(j//2)+1+j%2])))
        ptr.append(ptr[-1]+len(rows)+1)
    cone_matrix=sparse.csc_matrix((np.concatenate(data),np.concatenate(indices).astype(np.int32),np.asarray(ptr,np.int32)),shape=(m+3*num_agents,n))
    if accel_mode.CHECK:
        blocks=[sparse.csc_matrix(-matrix)]
        for agent in range(num_agents):
            block=np.zeros((3,2*num_agents),dtype=np.float64)
            block[1:,2*agent:2*agent+2]=-np.eye(2);blocks.append(sparse.csc_matrix(block))
        ref=sparse.vstack(blocks,format='csc')
        assert ref.shape==cone_matrix.shape
        for field in ('data','indices','indptr'):
            aa=getattr(ref,field);bb=getattr(cone_matrix,field)
            assert aa.dtype==bb.dtype and aa.tobytes()==bb.tobytes(),field
        accel_mode.MATRIX_CHECKS+=1
    cone_rhs=np.concatenate([-lower]+[np.asarray([max_speed,0.,0.]) for _ in range(num_agents)])
    cones=[clarabel.NonnegativeConeT(m)]+[clarabel.SecondOrderConeT(3) for _ in range(num_agents)]
    if n not in _P:
        eye=sparse.csc_matrix(np.eye(n));_P[n]=(eye.data.copy(),eye.indices.copy(),eye.indptr.copy())
    cached=_P[n];quadratic=sparse.csc_matrix((cached[0].copy(),cached[1].copy(),cached[2].copy()),shape=(n,n))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_iter = config.max_iterations
    settings.tol_feas = min(config.feasibility_tol, config.speed_tol)
    settings.tol_gap_abs = config.solver_ftol
    settings.tol_gap_rel = config.solver_ftol
    solution = clarabel.DefaultSolver(
        quadratic,
        -target,
        cone_matrix,
        cone_rhs,
        cones,
        settings,
    ).solve()
    candidate = np.asarray(solution.x, dtype=np.float64)
    diagnostics = _diagnostics(candidate, target, matrix, lower, max_speed)
    dual = np.asarray(solution.z, dtype=np.float64)
    slack = np.asarray(solution.s, dtype=np.float64)
    stationarity = float(np.linalg.norm(candidate - target + cone_matrix.T @ dual, ord=np.inf))
    complementarity = float(abs(np.dot(slack, dual)))
    status = str(solution.status)
    diagnostics.update(
        backend="clarabel_exact_socp",
        solver_status=status,
        accepted=False,
        conic_iterations=int(solution.iterations),
        primal_residual=float(solution.r_prim),
        dual_residual=float(solution.r_dual),
        stationarity_residual=stationarity,
        complementarity_residual=complementarity,
        projection_input=target.tolist(),
    )
    certified = (
        np.isfinite(candidate).all()
        and diagnostics["min_linear_residual"] >= -config.feasibility_tol
        and diagnostics["max_speed_excess"] <= config.speed_tol
        and max(stationarity, complementarity) <= config.optimality_tol
    )
    if not certified:
        raise CBFSolverError("conic_failed", diagnostics)
    diagnostics["accepted"] = True
    return (
        candidate.reshape(num_agents, 2).copy(),
        "solved" if status == "Solved" else "solved_kkt_certified",
        diagnostics,
    )

def project_velocity(*a,**kw):
    return (_fast_project_velocity if accel_mode.ENABLED else _original)(*a,**kw)
