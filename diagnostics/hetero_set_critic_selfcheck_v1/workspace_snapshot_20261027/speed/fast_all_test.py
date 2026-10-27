"""Apply four engineering changes in-process and check bit-identical trajectories against the original code."""
import sys, os, json, time, math
os.environ['JAX_PLATFORMS'] = 'cpu'
H = '/home/zhihan/.claude/jobs/7eda8ddd/tmp/scene_search/harness'; sys.path.insert(0, H); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bootstrap  # noqa
from common import actions, uid
from design import controllers
from scenes import make_new
from execution import execute
from legacy_states import LiteratureConfig
import numpy as np, jax
import shared_control.hard_projection as HP
import swap_safety, swap_safety_user_fix_20261023 as SFIX, controllers_new as CN, runtime_new as RN
from fast_projection_test import fast_project_velocity
import np_threefry
from bottleneck_family.scenario import Config, Geometry
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
A = actions(); C = controllers(); keys = [x['key'] for x in A]
S = {s['name']: s for f in ['scenes_explore.json', 'scenes_explore2.json', 'scenes_explore3.json'] for s in json.load(open(f'/home/zhihan/.claude/jobs/7eda8ddd/tmp/scene_search/{f}'))}
def swap_scene(name, N, K=None):
    th = np.arange(N) * 2 * np.pi / N; init = (N / 8) * np.stack([np.cos(th), np.sin(th)], -1); rng = np.random.default_rng(np.random.SeedSequence([2026102661, N] + ([K] if K else [])))
    st = [dict(index=i, N=N, positions=(init if i == 0 else init + rng.uniform(-.1, .1, init.shape)).tolist(), goals=(-init).tolist(), velocities=np.zeros((N, 2)).tolist(), scene=name) for i in range(2)]
    walls = []
    if K:
        a_, rho = .07, .75; R0 = N / 8; Hh = max(2.0, round(R0 + .9, 2)); ang = [2 * math.pi * i / N + math.pi / N for i in range(N)]; pick = ang if K == N else ang[::2]
        bc = Config(num_agents=N, half_length=Hh, half_height=Hh, barrier_x=(Hh - .3,), openings=(((0., 2 * Hh - .4),),), barrier_thickness=.1,
                    extra_rectangles=tuple((rho*R0*math.cos(t)-a_, rho*R0*math.sin(t)-a_, rho*R0*math.cos(t)+a_, rho*R0*math.sin(t)+a_) for t in pick), agent_radius=.05, dt=.05, max_speed=.5, max_steps=400).to_dict()
        walls = Geometry(Config(**bc)).walls.tolist()
        for s in st: s['bottleneck_config'] = dict(bc)
    longest = max(np.linalg.norm(np.asarray(s['goals']) - np.asarray(s['positions']), axis=-1).max() for s in st) / .5
    sc = dict(name=name, family='speed_test', factor='speed_test', config=LiteratureConfig(name, N, .05, .05, .5, int(math.ceil(4 * longest / .05))).to_dict(), walls=walls, states=st)
    if K: sc['bottleneck_config'] = st[0]['bottleneck_config']
    sc['definition_sha256'] = uid('scene_definition', sc); return sc
S['xsc_speed_obst_n08_k8'] = swap_scene('xsc_speed_obst_n08_k8', 8, 8); S['xsc_speed_swap_n10'] = swap_scene('xsc_speed_swap_n10', 10)
cases = [(sc, ci, si, a) for sc in ('xsc_x_box_n08_r015_f25', 'xsc_x_plus_w025', 'xsc_x_time_f30_r005', 'xsc_speed_obst_n08_k8', 'xsc_speed_swap_n10') for ci in range(3) for si in (0, 1) for a in ('safety', 'sE22', 'sE05')]
set_mode(False); execute(make_new(S[cases[0][0]], C[0], S[cases[0][0]]['states'][0], A[0], 0))
per = {}; same = 0
for sc, ci, si, a in cases:
    c = make_new(S[sc], C[ci], S[sc]['states'][si], A[keys.index(a)], 0)
    set_mode(False); t = time.time(); r0, a0 = execute(c); t0 = time.time() - t
    set_mode(True); t = time.time(); r1, a1 = execute(c); t1 = time.time() - t
    eq = all(np.array_equal(a0[k], a1[k]) for k in ('positions', 'velocities', 'actions', 'safe', 'nominal')) and all(r0[k] == r1[k] for k in ('success', 'outcome', 'episode_length', 'collision', 'constraint_violation_steps'))
    same += eq; v = per.setdefault((sc, ci), [0., 0., 0]); v[0] += t0; v[1] += t1; v[2] += eq
set_mode(False)
names = ['集中式随机采样', '梯度下降模型预测', '分布式模型预测']
for (sc, ci), (t0, t1, e) in per.items(): print(f'{sc:24s} {names[ci]:8s} identical {e}/6  original {t0:5.1f}s  optimized {t1:5.1f}s  saving {(1 - t1 / t0) * 100:4.0f}%')
T0 = sum(v[0] for v in per.values()); T1 = sum(v[1] for v in per.values())
print(f'TOTAL {len(cases)} rollouts: bit-identical {same}/{len(cases)}; original {T0:.1f}s, optimized {T1:.1f}s, saving {(1 - T1 / T0) * 100:.0f}% (speed-up x{T0 / T1:.2f})')
