"""Circle state-conditioned selector: features, models, leave-one-state-out selection, held-out evaluation.
Sealed (sha256 in PROTOCOL_ADDENDUM_20261024_circle_selector.md) before any learn_* outcome is read."""
from pathlib import Path
import sys
S2 = Path(__file__).resolve().parent
sys.path.insert(0, str(S2))
from ceiling_common import *  # noqa: F401,F403
from ceiling_driver2 import training_rows, stage_rows2, stage_name, seed0_stages
from learn_states import new_states, TRAIN_NEW, HELDOUT_NEW
from statistics_supplement_user_fix_20261023 import infer
from collections import defaultdict
from itertools import combinations

CIRCLE = 'lit_circle_five'
MENU = ('sc_rf_circle_1_0__eta9__th25', 'sE61', 'rf_circle_1_3')
REFS = ('eta9', 'eta13', 'eta9_to_3')
RADIUS = 2.0
LAMBDAS = (.1, 1., 10., 100.)
KS = (3, 5, 9, 15)
FEATURES = ('min_closest_approach', 'mean_closest_approach', 'conflict_pairs', 'crossing_pairs', 'mean_path', 'path_spread',
            'head_on_conflicts', 'conflict_timing', 'midpoint_crowding', 'midpoint_centrality')


def _intersect(a, b, c, d):
    def orient(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0


def features(s):
    p = np.asarray(s['positions'], float); g = np.asarray(s['goals'], float); d = g - p; L = np.linalg.norm(d, axis=1); u = d / L[:, None]
    dstar, tstar, cosang, cross = [], [], [], 0
    for i, j in combinations(range(len(p)), 2):
        r0 = p[i] - p[j]; w = u[i] - u[j]; T = min(L[i], L[j])
        t = 0. if w @ w < 1e-12 else float(np.clip(-(r0 @ w) / (w @ w), 0., T))
        dstar.append(np.linalg.norm(r0 + w * t)); tstar.append(t / L.mean()); cosang.append(u[i] @ u[j]); cross += int(_intersect(p[i], g[i], p[j], g[j]))
    dstar, tstar, cosang = map(np.asarray, (dstar, tstar, cosang)); conflict = dstar < 3 * RADIUS
    mid = p + d / 2; md = min(np.linalg.norm(mid[i] - mid[j]) for i, j in combinations(range(len(p)), 2))
    return np.array([dstar.min() / RADIUS, dstar.mean() / RADIUS, conflict.sum(), cross, L.mean() / RADIUS, L.std() / RADIUS,
                     ((cosang < -.8) & conflict).sum(), tstar[conflict].mean() if conflict.any() else 1., md / RADIUS,
                     np.linalg.norm(mid, axis=1).mean() / RADIUS], float)


def state_table(rows, keys):
    """Per state: mean success over the training controllers for each action (equal controller weight)."""
    by = defaultdict(list)
    for r in rows:
        if r['action'] in keys:
            by[(r['state'], r['action'])].append(r['success'])
    states = sorted({s for s, _ in by})
    Y = np.array([[np.mean(by[(s, a)]) for a in keys] for s in states])
    n = {len(by[(s, a)]) for s in states for a in keys}; assert n == {7}, n
    return states, Y


def ridge_fit(X, Y, lam):
    Xa = np.hstack([X, np.ones((len(X), 1))]); P = np.eye(Xa.shape[1]) * lam; P[-1, -1] = 0.
    return np.linalg.solve(Xa.T @ Xa + P, Xa.T @ Y)


def ridge_pick(W, X):
    pred = np.hstack([X, np.ones((len(X), 1))]) @ W
    return np.array([int(np.flatnonzero(np.isclose(r, r.max()))[0]) for r in pred])


def knn_pick(Xtr, Ytr, X, k):
    out = []
    for x in X:
        nn = np.argsort(np.linalg.norm(Xtr - x, axis=1), kind='stable')[:k]; m = Ytr[nn].mean(axis=0)
        out.append(int(np.flatnonzero(np.isclose(m, m.max()))[0]))
    return np.array(out)


def loso(X, Y, fit_pick):
    vals = []
    for i in range(len(X)):
        keep = np.arange(len(X)) != i; mu = X[keep].mean(0); sd = X[keep].std(0) + 1e-9
        a = fit_pick((X[keep] - mu) / sd, Y[keep], ((X[i] - mu) / sd)[None])[0]; vals.append(Y[i, a])
    return float(np.mean(vals))


def train():
    keep = {s['index'] for s in train_states(CIRCLE)} | set(TRAIN_NEW); ctls = {c['controller'] for c in train_ctls(CIRCLE)}; parent_raw_index.cache_clear()
    rows = {r['rollout_uid']: r for st in seed0_stages('circle') + ('learn_train',) for r in stage_rows2(st)
            if r['scene'] == CIRCLE and r['seed'] == 0 and r['state'] in keep and r['controller'] in ctls and r['action'] in MENU}
    states, Y = state_table(list(rows.values()), MENU)
    assert len(states) == 64, len(states)
    lookup = {s['index']: s for s in train_states(CIRCLE) + new_states(TRAIN_NEW)}
    X = np.array([features(lookup[s]) for s in states])
    ridge_cv = {lam: loso(X, Y, lambda Xt, Yt, Xq, lam=lam: ridge_pick(ridge_fit(Xt, Yt, lam), Xq)) for lam in LAMBDAS}
    knn_cv = {k: loso(X, Y, lambda Xt, Yt, Xq, k=k: knn_pick(Xt, Yt, Xq, k)) for k in KS}
    lam = max(LAMBDAS, key=lambda v: (ridge_cv[v], -LAMBDAS.index(v))); k = max(KS, key=lambda v: (knn_cv[v], -KS.index(v)))
    mu = X.mean(0); sd = X.std(0) + 1e-9; Z = (X - mu) / sd
    model = dict(states=states, Y=Y.tolist(), mu=mu.tolist(), sd=sd.tolist(), lam=lam, k=k, W=ridge_fit(Z, Y, lam).tolist(), Z=Z.tolist(),
                 ridge_cv=ridge_cv, knn_cv=knn_cv, always_first=float(Y[:, 0].mean()), per_state_oracle=float(Y.max(1).mean()),
                 label_counts={MENU[i]: int(c) for i, c in zip(*np.unique(Y.argmax(1), return_counts=True))})
    return model


def choose(model, s):
    z = (features(s) - np.asarray(model['mu'])) / np.asarray(model['sd'])
    ridge = ridge_pick(np.asarray(model['W']), z[None])[0]; knn = knn_pick(np.asarray(model['Z']), np.asarray(model['Y']), z[None], model['k'])[0]
    return MENU[ridge], MENU[knn]


def evaluate(model):
    held = [r for r in stage_rows2('learn_heldout') if r['scene'] == CIRCLE]
    states = {s['index']: s for s in eval_states(CIRCLE) + new_states(HELDOUT_NEW)}
    picks = {i: choose(model, s) for i, s in states.items()}
    out = {}
    for name, idx in (('evaluation_0_15_all_instances', [s['index'] for s in eval_states(CIRCLE)]), ('new_states_80_99_training_controllers', list(HELDOUT_NEW))):
        rr = [r for r in held if r['state'] in idx]
        by = {(r['controller'], r['state'], r['action']): r for r in rr}
        synth = []
        for (ctl, st, a), r in by.items():
            if a == MENU[0]:
                for arm, pick in (('learned_ridge', picks[st][0]), ('learned_knn', picks[st][1])):
                    synth.append(dict(by[(ctl, st, pick)], action=arm))
                means = {m: np.mean([x['success'] for x in rr if x['state'] == st and x['action'] == m]) for m in MENU}
                best = max(MENU, key=lambda m: (means[m], -MENU.index(m)))
                synth.append(dict(by[(ctl, st, best)], action='menu_oracle'))
        arms = ['learned_ridge', 'learned_knn', 'menu_oracle'] + list(MENU) + list(REFS)
        out[name] = dict(arms=arms, inference=infer(rr + synth, arms, (0,)), picks={st: picks[st] for st in idx},
                         pick_counts={m: sum(picks[st][0] == m for st in idx) for m in MENU})
    return out


def main():
    model = train(); result = dict(utc=utc(), menu=MENU, features=FEATURES, model=model, heldout=evaluate(model), analysis_sha256=sha(Path(__file__)))
    write2(S2 / 'LEARN_RESULTS.json', result); print('learn analysis written', flush=True)


if __name__ == '__main__':
    main()
