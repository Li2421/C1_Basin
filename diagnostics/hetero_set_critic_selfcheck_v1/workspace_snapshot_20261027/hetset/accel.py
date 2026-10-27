"""Four exact engineering speed-ups (copied from verified prototype); enable() patches in-process."""
import sys,os
sys.path.insert(0,'/home/zhihan/.claude/jobs/7eda8ddd/tmp/speed');sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax
import shared_control.hard_projection as HP
import swap_safety, swap_safety_user_fix_20261023 as SFIX, controllers_new as CN, runtime_new as RN
from fastproj import fast_project_velocity
import np_threefry
ORIG = dict(pv=swap_safety.project_velocity, lp=SFIX.original_problem, wp=CN.wall_penalties, fold=jax.random.fold_in)
# --- (3) controller wall penalty: drop walls that cannot come within the margin of an agent's sampled points
def wall_penalties_pruned(p, env, xp=np):
    c = env.config
    if xp is not np or not c.scene.startswith('xsc_'): return ORIG['wp'](p, env, xp)
    margin = c.agent_radius + c.wall_collision_margin + c.wall_radius; zero = np.zeros(p.shape[:-1])
    walls = np.asarray(env.walls)
    if not len(walls): return zero, zero
    a_all = walls[:, 0]; ab_all = walls[:, 1] - walls[:, 0]; inner = np.zeros(p.shape[:-1])
    for j in range(p.shape[-2]):
        q = p[..., j, :]; lo = q.reshape(-1, 2).min(0); hi = q.reshape(-1, 2).max(0); ctr = (lo + hi) / 2; rad = np.linalg.norm(hi - lo) / 2
        tt = np.clip(((ctr - a_all) * ab_all).sum(-1) / (ab_all * ab_all).sum(-1), 0, 1); dc = np.linalg.norm(ctr - (a_all + tt[:, None] * ab_all), axis=-1)
        keep = np.flatnonzero(dc - rad <= margin + 1e-9)
        if not len(keep): continue
        a = a_all[keep]; ab = ab_all[keep]; d = q[..., None, :] - a                      # same per-element formula as the original
        t = np.clip(np.sum(d * ab, axis=-1) / np.sum(ab * ab, axis=-1), 0, 1); delta = d - t[..., None] * ab
        distance = np.sqrt(np.maximum(np.sum(delta * delta, axis=-1), 1e-30))
        inner[..., j] = np.maximum(margin - np.min(distance, axis=-1), 0) ** 2
    return inner, zero
# --- (4) safety-layer local rows: vectorized conservative pre-screen, then the original scalar loop on kept walls only
def local_problem_prescreen(snapshot, i):
    walls = np.asarray(snapshot['walls'])
    if len(walls) == 0: return ORIG['lp'](snapshot, i)
    p = np.asarray(snapshot['positions']); a = walls[:, 0]; b = walls[:, 1]; ab = b - a
    t = np.clip(((p[i] - a) * ab).sum(-1) / (ab * ab).sum(-1), 0, 1); dist = np.linalg.norm(p[i] - (a + t[:, None] * ab), axis=-1)
    keep = np.flatnonzero(dist <= .5 + 1e-9)
    sub = dict(snapshot, walls=walls[keep]); A, lower, labels = ORIG['lp'](sub, i)
    labels = [['wall', lab[1], int(keep[lab[2]])] if lab[0] == 'wall' else lab for lab in labels]   # restore original wall indices
    return A, lower, labels
def set_mode(fast):
    swap_safety.project_velocity = fast_project_velocity if fast else ORIG['pv']
    SFIX.original_problem = local_problem_prescreen if fast else ORIG['lp']
    CN.wall_penalties = wall_penalties_pruned if fast else ORIG['wp']
    jax.random.fold_in = (lambda k, d: np_threefry.fold_in(k, d)) if fast else ORIG['fold']
def enable():set_mode(True)
