"""EXPLORATORY (not pre-registered) follow-ups on the same main labels; zero new rollouts.

E1  eta-ranking reversal along width: Q(eta10) - Q(eta15) per width, state bootstrap CI.
E2  Test B with DISJOINT physical states: train on states 0-7, evaluate on unseen
    states 8-15 at the held test widths (closer to LOSO: new geometry AND new states).
Writes exploratory_results.json.
"""
import json
import numpy as np
from . import analysis as A


def eta_reversal(D, rng):
    Y, W = D['Y'], D['widths']; S = Y.shape[1]; out = {}
    for i, w in enumerate(W):
        Q = np.nanmean(Y[i][..., :2], -1)                    # [S, C, eta], seeds 0/1
        d = np.nanmean(Q[:, :, 2] - Q[:, :, 3], 1)           # eta10 - eta15 per state (C-averaged)
        b = [np.nanmean(d[rng.integers(0, S, S)]) for _ in range(A.BOOT)]
        out[f'{w:.2f}'] = dict(mean=float(np.nanmean(d)), CI95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
    return out


def disjoint_states(D, rng):
    W, g = D['widths'], D['grid']; wi = lambda w: W.index(w)
    test = [wi(w) for w in g['test']]; pool = [wi(w) for w in g['pool']]; few = [wi(w) for w in g['few']]
    TR, TE = list(range(8)), list(range(8, 16)); t = sorted(g['test'])
    specs = {
        'MANY': [(p_, [(3 * i + k) % 8 for k in range(3)], [0, 1]) for i, p_ in enumerate(pool)],
        'FEW': [(f, TR, [0, 1]) for f in few],
        'FEW_x2': [(f, TR, [0, 1, 2, 3]) for f in few],
        'EXTRAP_A': [(wi(w), TR, [0, 1]) for w in g['pool'] if w < t[1]],
        'EXTRAP_B': [(wi(w), TR, [0, 1]) for w in g['pool'] if w > t[2]],
    }
    targets = {k: test for k in specs}
    targets['EXTRAP_A'] = [wi(t[2]), wi(t[3])]; targets['EXTRAP_B'] = [wi(t[0]), wi(t[1])]
    res, fns = {}, {}
    S = D['Y'].shape[1]
    for name, spec in specs.items():
        tr, y = A.training_rows(D, spec)
        pidx, _, _ = A.features(D, test, list(range(S)))
        p = A.fit_predict(D, tr, y, pidx).reshape(len(test), S, 5, 4)
        for tag, tw in (('own', targets[name]), ('at_A', targets['EXTRAP_A']), ('at_B', targets['EXTRAP_B'])):
            P = np.stack([p[test.index(w_)] for w_ in tw])
            fn, _ = A.evaluate(D, P, tw, tr, y)
            fns[(name, tag)] = fn
            res.setdefault(name, {})[tag] = fn(TE)                 # unseen states only
        res[name]['train_rollouts'] = int(len(y)); res[name]['state_width_units'] = int(sum(len(s[1]) for s in spec))
    samples = [rng.choice(TE, len(TE)).tolist() for _ in range(A.BOOT)]
    con = {}
    for a, b, tag in (('MANY', 'FEW', 'own'), ('FEW_x2', 'FEW', 'own'), ('MANY', 'FEW_x2', 'own'),
                      ('MANY', 'EXTRAP_A', 'at_A'), ('MANY', 'EXTRAP_B', 'at_B')):
        for k in ('B2', 'sel_model', 'B1_skill', 'eta_sel_model'):
            v = [fns[(a, tag)](st)[k] - fns[(b, tag)](st)[k] for st in samples]
            con[f'{a}-{b}@{tag}:{k}'] = dict(mean=float(np.nanmean(v)), CI95=[float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))])
    return dict(results=res, contrasts=con)


def main():
    rng = np.random.default_rng(20261012)
    D = A.load(); assert D['missing'] == 0
    out = dict(label='EXPLORATORY, not pre-registered', eta10_minus_eta15=eta_reversal(D, rng), disjoint_states=disjoint_states(D, rng))
    (A.HERE / 'exploratory_results.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out['eta10_minus_eta15']))
    for k, v in out['disjoint_states']['results'].items():
        o = v['own']
        print(k, {m: round(o[m], 3) for m in ('B2', 'B1_skill', 'sel_model', 'sel_oracle', 'sel_prior', 'eta_sel_model', 'eta_sel_oracle', 'eta_sel_prior')}, v['train_rollouts'])
    for k, v in out['disjoint_states']['contrasts'].items():
        print(f"  {k:36s} {v['mean']:+.3f} [{v['CI95'][0]:+.3f},{v['CI95'][1]:+.3f}]")


if __name__ == '__main__':
    main()
