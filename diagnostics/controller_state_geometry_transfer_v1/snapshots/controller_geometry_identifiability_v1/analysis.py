"""Pre-registered analysis (PREREGISTRATION.md sections 4-5). No tuning on test widths.

  python -m diagnostics.controller_geometry_identifiability_v1.analysis

Writes results.json and prints a summary.
"""
from __future__ import annotations

import itertools, json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent / 'controller_four_scene_mechanism_v1'
FAMILIES = ('isotropic', 'anisotropic', 'rotate_positive', 'rotate_negative', 'coupled')
ETA_INDICES = [0, 5, 10, 15]
LEARNER_SEEDS = [17, 23, 41, 53, 71]
BOOT = 2000
STEPS = 3000   # pre-registered; unit tests may lower it, final analysis asserts 3000
read = lambda p: json.loads(Path(p).read_text())


# ----------------------------------------------------------------------------- data
def load():
    p = read(HERE / 'protocol_main.json'); cases = read(HERE / 'cases_main.json')
    widths = sorted(float(w) for w in p['widths'])
    roles = {float(w): r for w, r in p['widths'].items()}
    parents = p['parent_states']
    arch = {s['state_uid']: s for s in read(ARCHIVE / 'protocol.json')['states']}
    Y = np.full((len(widths), len(parents), 5, 4, 4), np.nan)
    mech = {w: [] for w in widths}
    nf = missing = 0
    for c in cases:
        f = HERE / 'raw' / (c['rollout_uid'] + '.json')
        if not f.exists():
            missing += 1; continue
        r = read(f)
        wi = widths.index(c['width']); si = parents.index(c['parent_state_uid'])
        ci = FAMILIES.index(c['controller_name']); ei = ETA_INDICES.index(c['eta_index'])
        if r['numerical_failure']:
            nf += 1; continue
        Y[wi, si, ci, ei, c['seed']] = r['success']
        s = r['summary'] or {}
        mech[c['width']].append((r['outcome'], bool(s.get('ever_candidate_deadlock')), s.get('first_success_step')))
    X_state = np.array([np.concatenate([np.ravel(arch[u]['physical'][k]) for k in ('positions', 'goals', 'velocities')]) for u in parents])
    eta = np.array([p['eta'][str(j)] for j in ETA_INDICES], float)
    g = p['main_grid']
    return dict(Y=Y, widths=widths, roles=roles, X_state=X_state, eta=eta, grid=g, mech=mech,
                numerical_failures=nf, missing=missing, n_cases=len(cases))


def corr(x, y):
    x = np.ravel(x) - np.mean(x); y = np.ravel(y) - np.mean(y)
    d = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / d) if d > 1e-12 else float('nan')


def pattern(Q):          # Q [states, C, eta] -> operator-effect pattern [C, eta]
    t = np.nanmean(Q, 0); return t - t.mean(0, keepdims=True)


def interaction(Q):      # pure C x eta interaction
    t = np.nanmean(Q, 0); return t - t.mean(0, keepdims=True) - t.mean(1, keepdims=True) + t.mean()


# ----------------------------------------------------------------------------- Test A
def test_a(D, rng):
    Y, W = D['Y'], D['widths']
    q = lambda wi, sd: Y[wi][..., sd]                                    # [states, C, eta]
    rel = {W[i]: corr(pattern(q(i, 0)), pattern(q(i, 1))) for i in range(len(W))}
    info = [w for w in W if np.isfinite(rel[w]) and rel[w] >= 0.5]
    R = {}
    for w1, w2 in itertools.combinations(info, 2):
        i, j = W.index(w1), W.index(w2)
        R[f'{w1:.2f}-{w2:.2f}'] = 0.5 * (corr(pattern(q(i, 0)), pattern(q(j, 1))) + corr(pattern(q(i, 1)), pattern(q(j, 0))))
    adjacent = {k: v for k, v in R.items() if info.index(float(k.split('-')[1])) - info.index(float(k.split('-')[0])) == 1}
    drot = {}
    ns = Y.shape[1]
    for i, w in enumerate(W):
        Q = np.nanmean(Y[i][..., :2], -1)                                   # seeds 0/1, [states, C, eta]
        d_state = np.nanmean(0.5 * (Q[:, 2] + Q[:, 3]) - Q[:, 0], -1)      # per state
        boot = [np.nanmean(d_state[rng.integers(0, ns, ns)]) for _ in range(BOOT)]
        drot[f'{w:.2f}'] = dict(mean=float(np.nanmean(d_state)), CI95=[float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))])
    neg = [w for w, v in drot.items() if v['CI95'][1] < 0]
    pos = [w for w, v in drot.items() if v['CI95'][0] > 0]
    a1 = bool(R) and min(R.values()) <= 0.3 and bool(adjacent) and max(adjacent.values()) >= 0.7
    a2 = bool(neg) and bool(pos)
    f1 = bool(R) and min(R.values()) >= 0.7 and not a2
    mech = {}
    for w in W:
        v = D['mech'][w]
        out = np.array([x[0] for x in v]); fs = np.array([np.nan if x[2] is None else x[2] for x in v], float)
        mech[f'{w:.2f}'] = dict(success=float(np.mean(out == 'success')), timeout=float(np.mean(out == 'timeout')),
                                deadlock=float(np.mean(out == 'deadlock')),
                                ever_candidate_deadlock=float(np.mean([x[1] for x in v])),
                                median_success_step=float(np.nanmedian(fs)) if np.isfinite(fs).any() else None)
    per_width = {f'{w:.2f}': dict(role=D['roles'][w], reliability=rel[w], mean_Q=float(np.nanmean(Y[W.index(w)][..., :2])),
                                  Q_by_operator=np.round(np.nanmean(Y[W.index(w)][..., :2], (0, 2, 3)), 3).tolist(),
                                  Q_by_eta=np.round(np.nanmean(Y[W.index(w)][..., :2], (0, 1, 3)), 3).tolist())
                 for w in W}
    return dict(per_width=per_width, informative_widths=info, cross_width_R=R, adjacent_R=adjacent, delta_rot=drot,
                significant_negative=neg, significant_positive=pos, A1_supported=a1, A2_sign_reversal=a2,
                A2_partial=(bool(neg) or bool(pos)) and not a2, F1_falsified=f1, mechanism=mech)


# ----------------------------------------------------------------------------- Test B learner
def features(D, wi_list, si_list):
    W, Xs, eta = D['widths'], D['X_state'], D['eta']
    rows = []
    for wi in wi_list:
        for si in si_list:
            for ci in range(5):
                for ei in range(4):
                    rows.append((wi, si, ci, ei))
    rows = np.array(rows, int)
    cont = np.concatenate([np.array(W)[rows[:, 0], None], Xs[rows[:, 1]], eta[rows[:, 3]]], 1)
    onehot = np.eye(5)[rows[:, 2]]
    return rows, cont, onehot


def training_rows(D, spec):
    """spec: list of (width_index, state_indices, seeds) -> per-rollout rows."""
    idx, y = [], []
    Y = D['Y']
    for wi, sis, seeds in spec:
        for si in sis:
            for ci in range(5):
                for ei in range(4):
                    for sd in seeds:
                        v = Y[wi, si, ci, ei, sd]
                        if np.isfinite(v):
                            idx.append((wi, si, ci, ei)); y.append(v)
    return np.array(idx, int), np.array(y, float)


def fit_predict(D, train_idx, y, pred_idx):
    import jax, jax.numpy as jnp, optax
    import flax.linen as nn
    W, Xs, eta = np.array(D['widths']), D['X_state'], D['eta']
    mk = lambda ix: np.concatenate([W[ix[:, 0], None], Xs[ix[:, 1]], eta[ix[:, 3]]], 1)
    Ctr = mk(train_idx); mu = Ctr.mean(0); sd = np.maximum(Ctr.std(0), 1e-6)
    enc = lambda ix: np.concatenate([(mk(ix) - mu) / sd, np.eye(5)[ix[:, 2]]], 1).astype(np.float32)
    Xtr, Xte = jnp.asarray(enc(train_idx)), jnp.asarray(enc(pred_idx)); ytr = jnp.asarray(y, jnp.float32)

    class MLP(nn.Module):
        @nn.compact
        def __call__(self, x):
            for _ in range(3):
                x = nn.silu(nn.Dense(128)(x))
            return nn.Dense(1)(x)[:, 0]

    model = MLP(); opt = optax.adamw(1e-3, weight_decay=1e-4)

    @jax.jit
    def step(params, state, key):
        b = jax.random.randint(key, (256,), 0, Xtr.shape[0])
        def loss(pr):
            return optax.sigmoid_binary_cross_entropy(model.apply(pr, Xtr[b]), ytr[b]).mean()
        l, g = jax.value_and_grad(loss)(params)
        u, state = opt.update(g, state, params)
        return optax.apply_updates(params, u), state, l

    probs = []
    for seed in LEARNER_SEEDS:
        key = jax.random.PRNGKey(seed)
        params = model.init(key, Xtr[:2]); state = opt.init(params)
        for t in range(STEPS):
            key, k = jax.random.split(key)
            params, state, _ = step(params, state, k)
        probs.append(np.asarray(jax.nn.sigmoid(model.apply(params, Xte))))
    return np.mean(probs, 0)


def evaluate(D, P, test_wis, train_idx, y):
    """P: predictions [len(test_wis), S, 5, 4]. Metrics of PREREGISTRATION.md B1-B3."""
    Y, S = D['Y'], D['Y'].shape[1]
    prior = np.full((5, 4), np.nan)
    for ci in range(5):
        for ei in range(4):
            m = (train_idx[:, 2] == ci) & (train_idx[:, 3] == ei)
            prior[ci, ei] = y[m].mean() if m.any() else y.mean()
    tr_w = sorted(set(train_idx[:, 0].tolist()))
    near = {}
    for wi in test_wis:
        wj = min(tr_w, key=lambda k: abs(D['widths'][k] - D['widths'][wi]))
        tab = np.zeros((5, 4))
        for ci in range(5):
            for ei in range(4):
                m = (train_idx[:, 0] == wj) & (train_idx[:, 2] == ci) & (train_idx[:, 3] == ei)
                tab[ci, ei] = y[m].mean()
        near[wi] = tab

    def metrics(states):
        out = dict(brier_model=[], brier_prior=[], brier_near=[], B2=[], B2_interaction=[], ceiling=[],
                   sel_model=[], sel_oracle=[], sel_prior=[], sel_near=[], sel_random=[],
                   eta_sel_model=[], eta_sel_oracle=[], eta_sel_prior=[])
        for k, wi in enumerate(test_wis):
            obs = Y[wi][states]                                       # [S, C, eta, seeds]
            pr = P[k][states]
            for sd in range(4):
                o = obs[..., sd]; m = np.isfinite(o)
                out['brier_model'].append(((pr - o)[m] ** 2).sum()); out['brier_prior'].append(((prior[None] - o)[m] ** 2).sum())
                out['brier_near'].append(((near[wi][None] - o)[m] ** 2).sum())
            qall = np.nanmean(obs, -1)
            out['B2'].append(corr(pattern(pr), pattern(qall)))
            out['B2_interaction'].append(corr(interaction(pr), interaction(qall)))
            out['ceiling'].append(np.sqrt(max(corr(pattern(np.nanmean(obs[..., :2], -1)), pattern(np.nanmean(obs[..., 2:], -1))), 0)))
            q01, q23 = np.nanmean(obs[..., :2], -1), np.nanmean(obs[..., 2:], -1)
            for si in range(len(states)):
                flat = lambda a: a.reshape(-1)
                out['sel_model'].append(flat(q23[si])[np.argmax(flat(pr[si]))])
                out['sel_oracle'].append(flat(q23[si])[np.argmax(np.nan_to_num(flat(q01[si]), nan=-1))])
                out['sel_prior'].append(flat(q23[si])[np.argmax(flat(prior))])
                out['sel_near'].append(flat(q23[si])[np.argmax(flat(near[wi]))])
                out['sel_random'].append(np.nanmean(q23[si]))
                for ci in range(5):
                    out['eta_sel_model'].append(q23[si, ci][np.argmax(pr[si, ci])])
                    out['eta_sel_oracle'].append(q23[si, ci][np.argmax(np.nan_to_num(q01[si, ci], nan=-1))])
                    out['eta_sel_prior'].append(q23[si, ci][np.argmax(prior[ci])])
        n = sum(np.isfinite(Y[wi][states]).sum() for wi in test_wis)
        res = dict(B1_skill=1 - sum(out['brier_model']) / sum(out['brier_prior']),
                   B1_near_skill=1 - sum(out['brier_near']) / sum(out['brier_prior']),
                   brier_model=sum(out['brier_model']) / n, brier_prior=sum(out['brier_prior']) / n,
                   B2=float(np.nanmean(out['B2'])), B2_per_width=[float(v) for v in out['B2']],
                   B2_interaction=float(np.nanmean(out['B2_interaction'])), B2_ceiling=float(np.nanmean(out['ceiling'])))
        for k in ('sel_model', 'sel_oracle', 'sel_prior', 'sel_near', 'sel_random', 'eta_sel_model', 'eta_sel_oracle', 'eta_sel_prior'):
            res[k] = float(np.nanmean(out[k]))
        return res

    return metrics, list(range(S))


def test_b(D, rng):
    Y, W, g = D['Y'], D['widths'], D['grid']
    wi = lambda w: W.index(w)
    test = [wi(w) for w in g['test']]; pool = [wi(w) for w in g['pool']]; few = [wi(w) for w in g['few']]
    S = Y.shape[1]; allS = list(range(S))
    t = sorted(g['test'])
    specs = {
        'MANY': [(p_, [(5 * i + k) % S for k in range(5)], [0, 1]) for i, p_ in enumerate(pool)],
        'FEW': [(f, allS, [0, 1]) for f in few],
        'FEW_x2': [(f, allS, [0, 1, 2, 3]) for f in few],
        'ALL_POOL': [(p_, allS, [0, 1]) for p_ in pool],
        'EXTRAP_A': [(wi(w), allS, [0, 1]) for w in g['pool'] if w < t[1]],
        'EXTRAP_B': [(wi(w), allS, [0, 1]) for w in g['pool'] if w > t[2]],
    }
    targets = {k: test for k in specs}
    targets['EXTRAP_A'] = [wi(t[2]), wi(t[3])]; targets['EXTRAP_B'] = [wi(t[0]), wi(t[1])]
    results, metric_fns, preds = {}, {}, {}
    for name, spec in specs.items():
        tr_idx, y = training_rows(D, spec)
        all_t = sorted(set(test) | set(targets[name]))
        pidx, _, _ = features(D, all_t, allS)
        p = fit_predict(D, tr_idx, y, pidx).reshape(len(all_t), S, 5, 4)
        preds[name] = {w_: p[k] for k, w_ in enumerate(all_t)}
        for tag, tw in (('own_targets', targets[name]), ('extrapA_targets', targets['EXTRAP_A']), ('extrapB_targets', targets['EXTRAP_B'])):
            P = np.stack([preds[name][w_] for w_ in tw])
            fn, _ = evaluate(D, P, tw, tr_idx, y)
            metric_fns[(name, tag)] = fn
            results.setdefault(name, {})[tag] = fn(allS)
        results[name]['train_rollouts'] = int(len(y)); results[name]['train_widths'] = [W[k] for k in sorted({s[0] for s in spec})]
    # paired state bootstrap for the pre-registered contrasts (same resampled states for both arms)
    KEYS = ('B2', 'sel_model', 'B1_skill', 'eta_sel_model')
    samples = [rng.integers(0, S, S).tolist() for _ in range(BOOT)]

    def boot(a, b, label, tag='own_targets'):
        diffs = {k: [] for k in KEYS}
        for st in samples:
            ma, mb = metric_fns[(a, tag)](st), metric_fns[(b, tag)](st)
            for k in KEYS:
                diffs[k].append(ma[k] - mb[k])
        for k in KEYS:
            v = np.asarray(diffs[k], float)
            contrasts[f'{label}:{k}'] = dict(mean=float(np.nanmean(v)), CI95=[float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))])

    contrasts = {}
    boot('MANY', 'FEW', 'MANY-FEW'); boot('FEW_x2', 'FEW', 'FEWx2-FEW'); boot('MANY', 'FEW_x2', 'MANY-FEWx2')
    # extrapolation targets compared with interpolating regimes on the same widths
    boot('MANY', 'EXTRAP_A', 'MANY-EXTRAP_A@A', 'extrapA_targets'); boot('MANY', 'EXTRAP_B', 'MANY-EXTRAP_B@B', 'extrapB_targets')
    M, F, F2, EA, EB = (results[k]['own_targets'] for k in ('MANY', 'FEW', 'FEW_x2', 'EXTRAP_A', 'EXTRAP_B'))
    gap = M['B2'] - F['B2']
    verdict = dict(
        B_P1_many_learns=bool(M['B2'] >= 0.6 and M['B1_skill'] > 0),
        B_P2_many_beats_few=bool(gap >= 0.15 and contrasts['MANY-FEW:B2']['CI95'][0] > 0),
        B_P3_more_data_not_substitute=bool((F2['B2'] - F['B2']) <= 0.5 * gap) if gap > 0 else None,
        B_P4_extrapolation_fails=bool(EA['B2'] <= 0.3 and EB['B2'] <= 0.3 and EA['B1_skill'] <= 0 and EB['B1_skill'] <= 0),
        F2_within_data_is_bottleneck=bool(contrasts['MANY-FEWx2:B2']['CI95'][1] <= 0 and contrasts['MANY-FEWx2:sel_model']['CI95'][1] <= 0),
        F3_not_learnable_even_with_diversity=bool(M['B2'] <= 0.3 and M['B1_skill'] <= 0),
        F4_invariant_across_regimes=bool(min(EA['B2'], EB['B2']) >= 0.7 and min(EA['B1_skill'], EB['B1_skill']) > 0.3),
        extrapolation_detail=dict(EXTRAP_A=dict(B2=EA['B2'], B1=EA['B1_skill']), EXTRAP_B=dict(B2=EB['B2'], B1=EB['B1_skill'])))
    return dict(results=results, contrasts=contrasts, verdict=verdict)


def main():
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else 'results.json'
    rng = np.random.default_rng(20261011)
    D = load()
    if name == 'results.json':
        assert D['missing'] == 0, f"{D['missing']} raw files missing; final results require the complete batch"
        assert STEPS == 3000 and BOOT == 2000 and LEARNER_SEEDS == [17, 23, 41, 53, 71], 'pre-registered settings'
    A = test_a(D, rng)
    B = test_b(D, rng)
    out = dict(n_cases=D['n_cases'], numerical_failures=D['numerical_failures'], missing_raw=D['missing'],
               complete=D['missing'] == 0, grid=D['grid'], test_A=A, test_B=B)
    (HERE / name).write_text(json.dumps(out, indent=2, allow_nan=True) + '\n')
    print(json.dumps(dict(A=dict(info=A['informative_widths'], A1=A['A1_supported'], A2=A['A2_sign_reversal'], F1=A['F1_falsified'],
                                 neg=A['significant_negative'], pos=A['significant_positive']),
                          B=B['verdict']), indent=1))


if __name__ == '__main__':
    main()
