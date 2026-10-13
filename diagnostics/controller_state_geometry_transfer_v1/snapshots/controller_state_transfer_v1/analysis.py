"""Pre-registered analysis (PREREGISTRATION.md). Writes results.json.

  python -m diagnostics.controller_state_transfer_v1.analysis
"""
from __future__ import annotations

import dataclasses, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / 'controller_ff_factorial_repair_v1'
CONTROLLERS = ('flow_isotropic', 'flow_anisotropic')
SEEDS = (17, 23, 41)
BOOT, PERM = 2000, 200
ENT_STEPS, MLP_STEPS = 6000, 3000
read = lambda p: json.loads(Path(p).read_text())


def load():
    p = read(HERE / 'protocol.json'); cases = read(HERE / 'cases.json')
    W = sorted(float(w) for w in p['widths']); roles = {float(w): r for w, r in p['widths'].items()}
    states = p['train_states'] + p['new_states']; E = p['eta_indices']
    Y = np.full((len(W), len(states), 2, len(E), 4), np.nan); missing = nf = 0
    for c in cases:
        f = HERE / 'raw' / (c['rollout_uid'] + '.json')
        if not f.exists():
            missing += 1; continue
        r = read(f)
        if r['numerical_failure']:
            nf += 1; continue
        Y[W.index(c['width']), states.index(c['parent_state_uid']), CONTROLLERS.index(c['controller_name']), E.index(c['eta_index']), c['seed']] = r['success']
    arch = {s['state_uid']: s for s in read(ARCHIVE / 'protocol.json')['states']}
    phys = [arch[u]['physical'] for u in states]
    Xs = np.array([np.concatenate([np.ravel(ph[k]) for k in ('positions', 'goals', 'velocities')]) for ph in phys])
    eta = np.array([p['eta'][str(j)] for j in E], float)
    return dict(Y=Y, W=W, roles=roles, phys=phys, Xs=Xs, eta=eta, n_train_states=len(p['train_states']),
                missing=missing, numerical_failures=nf)


def corr(a, b):
    a = np.ravel(a) - np.mean(a); b = np.ravel(b) - np.mean(b); d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 1e-12 else float('nan')


def sxe(Q):                      # Q [S, K] -> state x eta pattern
    return Q - Q.mean(0, keepdims=True) - Q.mean(1, keepdims=True) + Q.mean()


def nanfill(Q):                  # rare numerical-failure cells: fill with (state-mean + eta-mean - grand)
    Q = Q.copy()
    if np.isnan(Q).any():
        r, c, g = np.nanmean(Q, 1, keepdims=True), np.nanmean(Q, 0, keepdims=True), np.nanmean(Q)
        Q = np.where(np.isnan(Q), r + c - g, Q)
    return Q


# ----------------------------------------------------------------------------- learners
def entities(D):
    from single_integrator.environment import Config, GiveWayEnv
    from diagnostics.orthoflow3_unified_representation_v1 import representation as rep
    env = read(ARCHIVE / 'protocol.json')['environments']['toy_give_way']['config']['config']
    items = []
    for w in D['W']:
        cfg = dataclasses.replace(Config(**env), corridor_width=float(w)); walls = GiveWayEnv(cfg).walls
        obs = [dict(a=x[0].tolist(), b=x[1].tolist(), radius=float(cfg.wall_radius), curvature=0., interior_sign=0.) for x in walls]
        for ph in D['phys']:
            n = len(ph['positions'])
            items.append(rep.entities(dict(positions=ph['positions'], velocities=ph['velocities'], goals=ph['goals'],
                flow=np.zeros((n, 2)).tolist(), radius=[cfg.agent_radius] * n, goal_tolerance=[cfg.goal_tolerance] * n, obstacles=obs,
                policy_origin=[0., 0.], policy_axis=[1., 0.], policy_axis_required=True, flow_committed=False,
                remaining_fraction=1., remaining_seconds=cfg.max_steps * cfg.dt, dt=cfg.dt, max_speed=cfg.max_speed,
                max_seconds=cfg.max_steps * cfg.dt, wall_margin=cfg.wall_collision_margin, agent_margin=cfg.agent_collision_margin,
                monitor_active=bool(cfg.terminate_on_deadlock), progress_window_seconds=cfg.progress_window_seconds,
                deadlock_hold_seconds=cfg.deadlock_hold_seconds, progress_epsilon=cfg.progress_epsilon,
                speed_epsilon_fraction=cfg.speed_epsilon_fraction, monitor_elapsed=0., monitor_history_empty=True)))
    X = rep.batch(items); assert np.allclose(X['globals'][0], X['globals'][-1])
    return X


def train_rows(D):
    Y = D['Y']; idx, y = [], []
    for wi, w in enumerate(D['W']):
        if D['roles'][w] != 'train':
            continue
        for si in range(D['n_train_states']):
            for ci in range(2):
                for ei in range(Y.shape[3]):
                    for sd in (0, 1):
                        v = Y[wi, si, ci, ei, sd]
                        if np.isfinite(v):
                            idx.append((wi, si, ci, ei)); y.append(v)
    return np.array(idx, int), np.array(y, float)


def all_cells(D):
    Y = D['Y']
    return np.array([(wi, si, ci, ei) for wi in range(Y.shape[0]) for si in range(Y.shape[1]) for ci in range(2) for ei in range(Y.shape[3])], int)


def fit_ent(D, X, tr, y, cells, seed):
    import jax, jax.numpy as jnp, optax
    from diagnostics.controller_q_diversity_v1.train import Critic
    S = D['Y'].shape[1]; Xj = {k: jnp.asarray(v, jnp.float32) for k, v in X.items()}
    onehot = jnp.eye(2, dtype=jnp.float32); eta = jnp.asarray(D['eta'], jnp.float32); na = X['agents'].shape[1]
    net = Critic('state_field_compact')

    def inputs(ix):
        g = ix[:, 0] * S + ix[:, 1]; x = {k: v[g] for k, v in Xj.items()}
        c = jnp.broadcast_to(onehot[ix[:, 2]][:, None, :], (len(g), na, 2)) * x['agent_mask'][..., None]
        return x, c, eta[ix[:, 3]]

    trj, yy = jnp.asarray(tr, jnp.int32), jnp.asarray(y, jnp.float32)
    params = net.init(jax.random.PRNGKey(seed), *inputs(trj[:2]))
    opt = optax.chain(optax.clip_by_global_norm(5.), optax.adamw(8e-4, weight_decay=1e-4)); st = opt.init(params)

    @jax.jit
    def block(params, st, draws):
        def step(carry, b):
            pp, oo = carry
            l, g = jax.value_and_grad(lambda q: optax.sigmoid_binary_cross_entropy(net.apply(q, *inputs(trj[b])), yy[b]).mean())(pp)
            u, oo = opt.update(g, oo, pp); return (optax.apply_updates(pp, u), oo), l
        return jax.lax.scan(step, (params, st), draws)

    rng = np.random.default_rng(seed)
    for _ in range(ENT_STEPS // 500):
        (params, st), _ = block(params, st, jnp.asarray(rng.integers(0, len(y), (500, 120)), jnp.int32))
    pred = jax.jit(lambda q, ix: jax.nn.sigmoid(net.apply(q, *inputs(ix)))); cj = jnp.asarray(cells, jnp.int32)
    return np.concatenate([np.asarray(pred(params, cj[k:k + 1024])) for k in range(0, len(cells), 1024)])


def fit_mlp(D, tr, y, cells, seed):
    import jax, jax.numpy as jnp, optax
    import flax.linen as nn
    W = np.asarray(D['W']); mk = lambda ix: np.concatenate([W[ix[:, 0], None], D['Xs'][ix[:, 1]], D['eta'][ix[:, 3]]], 1)
    C = mk(tr); mu, sd = C.mean(0), np.maximum(C.std(0), 1e-6)
    enc = lambda ix: jnp.asarray(np.concatenate([(mk(ix) - mu) / sd, np.eye(2)[ix[:, 2]]], 1), jnp.float32)
    Xtr, ytr, Xte = enc(tr), jnp.asarray(y, jnp.float32), enc(cells)

    class MLP(nn.Module):
        @nn.compact
        def __call__(self, x):
            for _ in range(3):
                x = nn.silu(nn.Dense(128)(x))
            return nn.Dense(1)(x)[:, 0]

    net, opt = MLP(), optax.adamw(1e-3, weight_decay=1e-4)
    params = net.init(jax.random.PRNGKey(seed), Xtr[:2]); st = opt.init(params)

    @jax.jit
    def block(params, st, draws):
        def step(carry, b):
            pp, oo = carry
            l, g = jax.value_and_grad(lambda q: optax.sigmoid_binary_cross_entropy(net.apply(q, Xtr[b]), ytr[b]).mean())(pp)
            u, oo = opt.update(g, oo, pp); return (optax.apply_updates(pp, u), oo), l
        return jax.lax.scan(step, (params, st), draws)

    rng = np.random.default_rng(seed)
    for _ in range(MLP_STEPS // 500):
        (params, st), _ = block(params, st, jnp.asarray(rng.integers(0, len(y), (500, 256)), jnp.int32))
    return np.asarray(jax.nn.sigmoid(net.apply(params, Xte)))


def lookup(D):
    """Same training state, nearest training width, observed seed-0/1 Q (non-learned reference)."""
    Y = D['Y']; W = D['W']; trw = [i for i, w in enumerate(W) if D['roles'][w] == 'train']
    P = np.full(Y.shape[:4], np.nan)
    for wi in range(len(W)):
        wj = min(trw, key=lambda k: (abs(W[k] - W[wi]), W[k]))
        P[wi] = np.nanmean(Y[wj][..., :2], -1)
    return P


# ----------------------------------------------------------------------------- metrics
def evaluate(D, P, wis, sis, rng):
    """P [W, S, 2, K] predictions. Returns m1, ceiling, m2, selection gains with state-bootstrap CIs."""
    Y = D['Y']; sis = np.asarray(sis)
    def stats(ss):
        m1, ceil, rel, num, den, gain, mm, orc = [], [], [], [], [], [], [], []
        for wi in wis:
            for ci in range(2):
                A = nanfill(np.nanmean(Y[wi, ss, ci][..., :2], -1)); B = nanfill(np.nanmean(Y[wi, ss, ci][..., 2:], -1))
                T = nanfill(np.nanmean(Y[wi, ss, ci], -1)); pr = P[wi, ss, ci]
                pa, pb, pt, pp = sxe(A), sxe(B), sxe(T), sxe(pr)
                m1.append(corr(pp, pt)); r_ = corr(pa, pb); rel.append(r_); ceil.append(np.sqrt(max(2 * r_ / (1 + r_), 0)) if r_ > -1 else 0.)
                num.append(np.mean((pp - pa) * (pp - pb))); den.append(np.mean(pa * pb))
                ad = pr.argmax(1); fx = pr.mean(0).argmax()
                gain.append(np.mean(B[np.arange(len(ss)), ad] - B[:, fx]))
                orc.append(np.mean(B[np.arange(len(ss)), A.argmax(1)] - B[:, A.mean(0).argmax()]))
                mis = []
                for _ in range(PERM):
                    perm = rng.permutation(len(ss))
                    while len(ss) > 1 and np.any(perm == np.arange(len(ss))):
                        perm = rng.permutation(len(ss))
                    mis.append(np.mean(B[np.arange(len(ss)), pr[perm].argmax(1)]))
                mm.append(np.mean(B[np.arange(len(ss)), ad]) - np.mean(mis))
        den_ = float(np.mean(den))
        return dict(m1=float(np.nanmean(m1)), ceiling=float(np.nanmean(ceil)), reliability=float(np.nanmean(rel)), energy=den_,
                    m2=float(1 - np.mean(num) / den_) if den_ > 0 else float('nan'),
                    adaptive_gain=float(np.mean(gain)), oracle_adaptive_gain=float(np.mean(orc)), matched_minus_mismatched=float(np.mean(mm)))
    point = stats(sis)
    boots = [stats(rng.choice(sis, len(sis))) for _ in range(200)]
    ci = lambda k: [float(np.nanpercentile([b[k] for b in boots], 2.5)), float(np.nanpercentile([b[k] for b in boots], 97.5))]
    point.update({f'{k}_CI95': ci(k) for k in ('m1', 'm2', 'energy', 'adaptive_gain', 'matched_minus_mismatched')})
    return point


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else 'results.json'
    D = load()
    if name == 'results.json':
        assert D['missing'] == 0 and ENT_STEPS == 6000 and MLP_STEPS == 3000
    rng = np.random.default_rng(20261011)
    W = D['W']; n_tr = D['n_train_states']; S = D['Y'].shape[1]
    trw = [i for i, w in enumerate(W) if D['roles'][w] == 'train']; hw = [i for i, w in enumerate(W) if D['roles'][w] == 'held']
    interp = [i for i in hw if W[i] < max(W[k] for k in trw)]; extrap = [i for i in hw if i not in interp]
    tr, y = train_rows(D); cells = all_cells(D); shape = D['Y'].shape[:4]
    X = entities(D)
    preds = {'ENT': np.mean([fit_ent(D, X, tr, y, cells, s) for s in SEEDS], 0).reshape(shape),
             'MLP': np.mean([fit_mlp(D, tr, y, cells, s) for s in SEEDS], 0).reshape(shape),
             'LOOKUP': lookup(D)}
    sets = dict(T1=(trw, list(range(n_tr, S))), T2=(interp, list(range(n_tr))), T2_extrap=(extrap, list(range(n_tr))),
                T3=(interp, list(range(n_tr, S))), T3_extrap=(extrap, list(range(n_tr, S))))
    res = {m: {k: evaluate(D, P, *v, rng) for k, v in sets.items() if not (m == 'LOOKUP' and k in ('T1', 'T3', 'T3_extrap'))}
           for m, P in preds.items()}
    def verdict(m):
        r = res[m]; L = res['LOOKUP']['T2']['m1']
        evaluable = res['LOOKUP']['T2']['reliability'] >= 0.4 and r['T1']['energy_CI95'][0] > 0 and r['T3']['energy_CI95'][0] > 0
        d1 = L >= 0.5 and r['T2']['m1'] <= 0.5 * L
        d2 = r['T1']['m1'] <= 0.2 and r['T1']['matched_minus_mismatched_CI95'][0] <= 0
        d3 = (r['T3']['m1'] < 0.5 * r['T1']['m1']) or (r['T3']['matched_minus_mismatched_CI95'][0] <= 0 < r['T1']['matched_minus_mismatched_CI95'][0])
        ok = (r['T2']['m1'] >= 0.5 * L and r['T2']['m1'] > 0.3 and r['T1']['m1'] > 0.3 and r['T3']['m1'] > 0.3
              and r['T3']['matched_minus_mismatched_CI95'][0] > 0)
        return dict(evaluable=bool(evaluable), D1_cross_geometry_state_transfer_defect=bool(d1), D2_state_generalization_defect=bool(d2),
                    D3_new_state_new_geometry_defect=bool(d3), method_defect_excluded=bool(ok))
    out = dict(missing=D['missing'], numerical_failures=D['numerical_failures'], results=res,
               verdict={m: verdict(m) for m in ('ENT', 'MLP')})
    (HERE / name).write_text(json.dumps(out, indent=2) + '\n')
    for m, r in res.items():
        for k, v in r.items():
            print(f"{m:6s} {k:9s} m1 {v['m1']:+.3f} [{v['m1_CI95'][0]:+.2f},{v['m1_CI95'][1]:+.2f}] ceil {v['ceiling']:.2f} "
                  f"m2 {v['m2']:+.2f} energy {v['energy']:.4f} gain {v['adaptive_gain']:+.3f} (oracle {v['oracle_adaptive_gain']:+.3f}) "
                  f"match-mis {v['matched_minus_mismatched']:+.3f} [{v['matched_minus_mismatched_CI95'][0]:+.3f},{v['matched_minus_mismatched_CI95'][1]:+.3f}]")
    print(json.dumps(out['verdict'], indent=1))


if __name__ == '__main__':
    main()
