"""Three-dimensional, held-velocity simulator with exact swept collision checks.

This isolated prototype mirrors the repository's Euclidean hard projection.
The route controller is an explicitly privileged feasibility witness, not Flow.
"""
from dataclasses import asdict, dataclass
from itertools import combinations

import clarabel
import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class Config:
    dt: float = 0.1
    max_steps: int = 800
    radius: float = 0.18
    max_speed: float = 0.5
    pair_xy: float = 0.5
    pair_z: float = 0.8
    gamma: float = 1.0
    buffer: float = 1e-4
    goal_tol: float = 0.04
    deadlock_window: int = 50
    deadlock_speed: float = 0.005
    deadlock_path: float = 0.02
    deadlock_pair_gap: float = 0.02


def geometry():
    """Physical solids, in metres: low corridor and one raised side alcove."""
    bounds = np.array([[-3., -.3, 0.], [3., .9, 1.1]])
    obstacles = np.array([
        [[-3., -.3, .8], [-1.5, .9, 1.1]],
        [[-.2, -.3, .8], [3., .9, 1.1]],
        [[-3., .3, 0.], [-1.5, .9, .8]],
        [[-.2, .3, 0.], [3., .9, .8]],
        [[-1.5, .3, 0.], [-.2, .9, .55]],
    ], dtype=float)
    return bounds, obstacles


def initial_state(seed):
    # These three pilots preserve head-on symmetry; not a robustness sample.
    rng = np.random.default_rng(seed)
    y, z = rng.uniform(-.012, .012, 2) + [0., .4]
    positions = np.array([[-2.3, y, z], [2.3, y, z]])
    return dict(positions=positions.tolist(), velocities=np.zeros((2, 3)).tolist(),
                goals=positions[::-1].copy().tolist())


def segment_box_distance(start, end, box):
    """Exact segment-to-AABB distance, including interior minima between frames."""
    delta = end - start
    cuts = [0., 1.]
    for k in range(3):
        if abs(delta[k]) > 1e-15:
            cuts.extend(float(t) for t in (box[:, k] - start[k]) / delta[k]
                        if 0. < t < 1.)
    cuts = sorted(set(cuts))
    candidates = list(cuts)
    for a, b in zip(cuts[:-1], cuts[1:]):
        midpoint = start + ((a + b) / 2) * delta
        active = (midpoint < box[0]) | (midpoint > box[1])
        edge = np.clip(midpoint, box[0], box[1])
        q = (start - edge)[active]
        v = delta[active]
        if v @ v > 1e-30:
            candidates.append(float(np.clip(-(q @ v) / (v @ v), a, b)))
    points = start + np.asarray(candidates)[:, None] * delta
    return float(np.linalg.norm(points - np.clip(points, box[0], box[1]), axis=1).min())


def swept_pair_gap(start, end, scale):
    relative = (start[0] - start[1]) / scale
    delta = ((end[0] - end[1]) - (start[0] - start[1])) / scale
    t = float(np.clip(-(relative @ delta) / (delta @ delta), 0., 1.)) if delta @ delta > 1e-30 else 0.
    return float(np.linalg.norm(relative + t * delta) - 1.)


def swept_clearances(start, end, config):
    bounds, obstacles = geometry()
    points = np.concatenate([start, end])
    wall = float(min((points - bounds[0]).min(), (bounds[1] - points).min()) - config.radius)
    for p, q in zip(start, end):
        for box in obstacles:
            wall = min(wall, segment_box_distance(p, q, box) - config.radius)
    pair = swept_pair_gap(start, end, np.array([config.pair_xy, config.pair_xy, config.pair_z]))
    physical = swept_pair_gap(start, end, np.full(3, 2 * config.radius)) * (2 * config.radius)
    return np.array([wall, physical, pair])  # m, m, dimensionless


def constraints(positions, config):
    n, dim = positions.shape
    assert dim == 3 and n == 2
    if not (0 < config.gamma * config.dt <= 1):
        raise ValueError("held-velocity safety requires 0 < gamma*dt <= 1")
    bounds, obstacles = geometry()
    rows, lower = [], []
    scale = np.array([config.pair_xy, config.pair_xy, config.pair_z])
    for i, j in combinations(range(n), 2):
        delta = positions[i] - positions[j]
        h = float(np.sum((delta / scale) ** 2) - (1. + config.buffer) ** 2)
        row = np.zeros((n, dim))
        row[i] = 2 * delta / scale ** 2
        row[j] = -row[i]
        rows.append(row.ravel()); lower.append(-config.gamma * h)
    for i, p in enumerate(positions):
        for k in range(dim):
            for sign, boundary in [(1., bounds[0, k]), (-1., bounds[1, k])]:
                row = np.zeros((n, dim)); row[i, k] = sign
                h = sign * (p[k] - boundary) - config.radius - config.buffer
                rows.append(row.ravel()); lower.append(-config.gamma * h)
        for box in obstacles:
            delta = p - np.clip(p, box[0], box[1])
            distance = np.linalg.norm(delta)
            if distance <= 1e-12:
                raise ValueError("agent centre inside obstacle")
            row = np.zeros((n, dim)); row[i] = delta / distance
            rows.append(row.ravel())
            lower.append(-config.gamma * (distance - config.radius - config.buffer))
    return np.array(rows), np.array(lower)


def project(target, positions, config):
    """min .5 ||u-target||², A u >= b, one true 3-D speed ball per agent."""
    matrix, lower = constraints(positions, config)
    flat = np.asarray(target, dtype=float).ravel()
    if not np.isfinite(flat).all():
        raise ValueError("nonfinite action")
    if (matrix @ flat >= lower).all() and np.linalg.norm(target, axis=1).max() <= config.max_speed:
        return target.copy()
    n = len(positions); size = 3 * n
    blocks, rhs = [sparse.csc_matrix(-matrix)], [-lower]
    cones = [clarabel.NonnegativeConeT(len(lower))]
    for i in range(n):
        block = np.zeros((4, size)); block[1:, i*3:i*3+3] = -np.eye(3)
        blocks.append(sparse.csc_matrix(block)); rhs.append(np.array([config.max_speed, 0., 0., 0.]))
        cones.append(clarabel.SecondOrderConeT(4))
    conic = sparse.vstack(blocks, format="csc")
    settings = clarabel.DefaultSettings(); settings.verbose = False
    settings.max_threads = 1
    settings.tol_feas = settings.tol_gap_abs = settings.tol_gap_rel = 1e-11
    solution = clarabel.DefaultSolver(sparse.eye(size, format="csc"), -flat, conic,
                                     np.concatenate(rhs), cones, settings).solve()
    u = np.asarray(solution.x).reshape(n, 3)
    stationarity = np.linalg.norm(u.ravel() - flat + conic.T @ np.asarray(solution.z), np.inf)
    complementarity = abs(np.asarray(solution.s) @ np.asarray(solution.z))
    if (str(solution.status) != "Solved" or not np.isfinite(u).all()
            or (matrix @ u.ravel() - lower).min() < -1e-8
            or np.linalg.norm(u, axis=1).max() > config.max_speed + 1e-8
            or max(stationarity, complementarity) > 2e-6):
        raise RuntimeError(f"projection failed: {solution.status}")
    return u


def nominal(positions, targets, config):
    u = .8 * (targets - positions)
    return u * np.minimum(1., config.max_speed / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-30))


def blocked(path, actions, goals, config):
    w = config.deadlock_window
    if len(actions) < w:
        return False
    p = path[-1]
    speeds = np.linalg.norm(np.asarray(actions[-w:]), axis=2)
    motion = np.linalg.norm(np.diff(np.asarray(path[-w-1:]), axis=0), axis=2).sum(axis=0)
    gap = swept_pair_gap(p, p, np.array([config.pair_xy, config.pair_xy, config.pair_z]))
    desired = nominal(p, goals, config)
    closing = (p[0] - p[1]) @ (desired[0] - desired[1]) < -.05
    return bool((np.linalg.norm(p-goals, axis=1) > config.goal_tol).all()
                and speeds.max() < config.deadlock_speed and motion.max() < config.deadlock_path
                and gap < config.deadlock_pair_gap and closing)


def route_targets(positions, goals, phase, hold):
    """Privileged sequential witness; its phase is explicit controller state."""
    waypoints = [np.array([-.85, 0., .4]), np.array([-.85, 0., .87]),
                 np.array([-.85, .60, .87]), goals[1],
                 np.array([-.85, 0., .87]), np.array([-.85, 0., .4]), goals[0]]
    active = 1 if phase == 3 else 0
    target = hold.copy(); target[active] = waypoints[phase]
    if np.linalg.norm(positions[active] - target[active]) <= .025:
        hold = positions.copy(); phase += 1
        if phase == len(waypoints):
            return goals.copy(), phase, hold
        return route_targets(positions, goals, phase, hold)
    return target, phase, hold


def rollout(state, controller, config):
    positions = np.array(state["positions"], float); goals = np.array(state["goals"], float)
    if np.min(swept_clearances(positions, positions, config)) <= 0:
        raise ValueError("initial state is unsafe")
    path, actions, clearances, phases = [positions.copy()], [], [], [0]
    event_step = None; phase = -1; hold = None; error = None
    termination = "timeout"
    for step in range(config.max_steps):
        targets = goals
        if controller == "route_witness" and phase >= 0 and phase < 7:
            targets, phase, hold = route_targets(positions, goals, phase, hold)
        try:
            u = project(nominal(positions, targets, config), positions, config)
        except (ValueError, RuntimeError) as exc:
            termination = "numerical_failure"; error = str(exc); break
        next_positions = positions + config.dt * u
        margins = swept_clearances(positions, next_positions, config)
        actions.append(u); clearances.append(margins); path.append(next_positions.copy()); phases.append(phase)
        positions = next_positions
        if margins.min() < -1e-9:
            termination = "collision"; break
        if (np.linalg.norm(positions-goals, axis=1) <= config.goal_tol).all():
            termination = "success"; break
        if event_step is None and blocked(path, actions, goals, config):
            event_step = step + 1
            if controller == "goal_cbf":
                termination = "deadlock"; break
            phase = 0; hold = positions.copy()
    trace = dict(positions=np.asarray(path), actions=np.asarray(actions), goals=goals,
                 swept_clearance=np.asarray(clearances), phases=np.asarray(phases),
                 steps=np.arange(len(path)), bounds=geometry()[0], obstacles=geometry()[1])
    mins = np.min(clearances, axis=0).tolist() if clearances else [None] * 3
    result = dict(outcome=termination, success=termination == "success", deadlock=termination == "deadlock",
                  timeout=termination == "timeout", collision=termination == "collision",
                  numerical_failure=termination == "numerical_failure", numerical_error=error,
                  episode_length=len(actions), deadlock_event_step=event_step,
                  deadlock_criterion="mutual_block_v1: both unfinished, 5s speed<0.005m/s, path<0.02m, ellipsoid gap<0.02, nominal closing",
                  minimum_wall_clearance=mins[0], minimum_agent_clearance=mins[1], minimum_pair_ellipsoid_gap=mins[2],
                  final_goal_distances=np.linalg.norm(positions-goals, axis=1).tolist(),
                  config=asdict(config), controller=controller, checkpoint=None,
                  safety_model="sphere vs solids + anisotropic pair exclusion; no aerodynamic downwash simulation")
    return trace, result
