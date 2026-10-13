"""Zero-rollout, zero-training audit of WHY the controller-Q relation does not transfer.

Read-only use of diagnostics/controller_ff_factorial_repair_v1 (raw labels, dataset,
saved source-only predictions, raw rollout summaries). Nothing in that directory is
modified. Run from the repository root:

  /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python \
      diagnostics/controller_fundamental_cause_audit_20261011/analysis.py

Sections
  1  operator algebra: what each affine controller does to the correction
  2  split-half unbiased variance decomposition of Q[scene, state, C, eta]
  3  do the SAME operators have the same effect in different scenes?
  4  failure mechanism per scene x eta (raw rollout summaries)
  5  few-shot identifiability of the held-scene eta choice
  6  immediate executed-action difference vs outcome difference
  7  what the source-only full model predicts on each held scene
"""
import glob, itertools, json
import numpy as np
from scipy.stats import spearmanr

D = 'diagnostics/controller_ff_factorial_repair_v1'
SCENES = ['toy_give_way', 'double_bottleneck', 'four_way_intersection', 'ring_exchange']
SHORT = dict(zip(SCENES, ['Toy', 'DB', 'Four', 'Ring']))
OPS = ['isotropic', 'anisotropic', 'rotate_positive', 'rotate_negative', 'coupled']
FOLD = dict(zip(SCENES, ['held_toy', 'held_db', 'held_four', 'held_ring']))

r = np.load(D + '/raw_labels.npz', allow_pickle=True)
z = np.load(D + '/dataset.npz', allow_pickle=True)
SUC, OBS = r['success'].astype(float), r['observed'].astype(float)
K = [i for i in range(SUC.shape[1]) if OBS[:, i].sum() > 0]          # eta indices [0,5,10,15]
SUC, OBS = SUC[:, K], OBS[:, K]                                       # [row, eta4, seed4]
scene, ctl, st = z['scene'], z['controller'], z['state_uid']
ETA = np.asarray(json.load(open(D + '/protocol.json'))['eta'])[K]


def half(seeds):
    s, o = SUC[:, :, seeds].sum(-1), OBS[:, :, seeds].sum(-1)
    return np.where(o > 0, s / np.maximum(o, 1), np.nan)


QA, QB, QF = half([0, 2]), half([1, 3]), half([0, 1, 2, 3])          # independent seed halves


def tensor(Q, kind):
    """[4 scenes, 40 states, 5 C, 4 eta]; rare unknowns -> (scene,C,eta) mean of the same half."""
    out = np.zeros((4, 40, 5, 4))
    for a, sc in enumerate(SCENES):
        for b, s in enumerate(sorted(set(st[scene == sc]))):
            for c, op in enumerate(OPS):
                m = (scene == sc) & (st == s) & (ctl == f'{kind}_{op}')
                assert m.sum() == 1
                out[a, b, c] = Q[m][0]
        for c in range(5):
            for e in range(4):
                v = out[a, :, c, e]
                v[np.isnan(v)] = np.nanmean(v)
    return out


def corr(x, y):
    x = np.ravel(x) - np.mean(x); y = np.ravel(y) - np.mean(y)
    d = np.sqrt((x @ x) * (y @ y))
    return float(x @ y / d) if d > 1e-12 else float('nan')


def effects(Y):
    """Balanced ANOVA (state nested in scene; C, eta crossed), effects broadcast to the grid."""
    m = Y.mean(); M = lambda ax: Y.mean(axis=ax, keepdims=True)
    sc, e, c = M((1, 2, 3)), M((0, 1, 2)), M((0, 1, 3))
    sce, scc, ce, scce = M((1, 2)), M((1, 3)), M((0, 1)), M(1)
    scs, scse, scsc = M((2, 3)), M(2), M(3)
    E = {'eta (shared)': e - m, 'scene': sc - m, 'scene x eta': sce - sc - e + m,
         'C (shared)': c - m, 'scene x C': scc - sc - c + m,
         'C x eta (shared)': ce - c - e + m,
         'scene x C x eta': scce - sce - scc - ce + sc + c + e - m,
         'state|scene': scs - sc, 'state x eta|scene': scse - scs - sce + sc,
         'state x C|scene': scsc - scs - scc + sc,
         'state x C x eta|scene': Y - scsc - scse - scce + scs + scc + sce - sc}
    return {k: np.broadcast_to(v, Y.shape) for k, v in E.items()}


def section(t):
    print('\n' + '=' * 78 + '\n' + t + '\n' + '=' * 78)


section('1. operator algebra (endpoint-matched families, controller_response_families_v1/runtime.py)')
I2, J = np.eye(2), np.array([[0., -1.], [1., 0.]])
R = lambda M: np.linalg.solve(M, I2 - np.linalg.matrix_power(I2 - M / 10, 10))
for name, M in [('isotropic', 2.5 * I2), ('rotate_positive', 2.5 * I2 + 2.5 * J),
                ('rotate_negative', 2.5 * I2 - 2.5 * J), ('anisotropic (goal frame)', np.diag([5., .8]))]:
    Rm = R(M)
    print(f'  {name:26s} executed-correction map R(M) = {np.round(Rm, 3).tolist()}; '
          f'rotation {np.degrees(np.arctan2(Rm[1, 0], Rm[0, 0])):+.1f} deg; gains {np.round(np.linalg.svd(Rm)[1], 3)}')
print('  With delta=0 and no active constraint every family returns the same nominal action;')
print('  a controller acts ONLY through how its field maps the correction and meets the constraints.')

section('2. split-half unbiased energy decomposition of Q (probability^2)')
for kind in ['flow', 'goal']:
    YA, YB = tensor(QA, kind), tensor(QB, kind)
    EA, EB = effects(YA), effects(YB)
    tot = float(np.mean((YA - YA.mean()) * (YB - YB.mean())))
    en = {k: float(np.mean(EA[k] * EB[k])) for k in EA}
    print(f'\n  [{kind}-reference] total signal variance {tot:.4f}')
    for k, v in en.items():
        print(f'    {k:24s} {v:+.5f} ({100 * v / tot:5.1f}%)')
    ck = ['C (shared)', 'scene x C', 'C x eta (shared)', 'scene x C x eta']
    ctot = sum(max(en[k], 0) for k in ck); csh = max(en['C (shared)'], 0) + max(en['C x eta (shared)'], 0)
    print(f'    controller-dependent energy {ctot:.4f}; shared across scenes {100 * csh / ctot:.1f}%')

section('3. same operator, different scenes: operator-effect pattern [C x eta] after removing scene x eta')
rng = np.random.default_rng(7)
for kind in ['goal', 'flow']:
    YA, YB, YF = tensor(QA, kind), tensor(QB, kind), tensor(QF, kind)
    op = lambda Y: Y.mean(0) - Y.mean(0).mean(0, keepdims=True)          # Y [states, C, eta]
    seed_rel = {SHORT[s]: round(corr(op(YA[i]), op(YB[i])), 3) for i, s in enumerate(SCENES)}
    within = {s: [] for s in SCENES}; cross = {}
    for _ in range(500):                                                  # disjoint 20/20 state halves
        p = [rng.permutation(40) for _ in range(4)]
        H1 = [op(YF[i][p[i][:20]]) for i in range(4)]; H2 = [op(YF[i][p[i][20:]]) for i in range(4)]
        for i, s in enumerate(SCENES):
            within[s].append(corr(H1[i], H2[i]))
        for i, j in itertools.combinations(range(4), 2):
            cross.setdefault(f'{SHORT[SCENES[i]]}-{SHORT[SCENES[j]]}', []).append(corr(H1[i], H2[j]))
    mean = lambda v: round(float(np.nanmean(v)), 3) if np.isfinite(v).any() else None
    print(f'\n  [{kind}] reliability across seed halves {seed_rel}')
    print(f'  [{kind}] reliability across disjoint states {{{", ".join(f"{SHORT[s]}: {mean(v)}" for s, v in within.items())}}}')
    print(f'  [{kind}] cross-scene correlation (disjoint states) {{{", ".join(f"{k}: {mean(v)}" for k, v in cross.items())}}}')
    for i, s in enumerate(SCENES):
        print(f'    {SHORT[s]:4s} Q by operator {OPS}:', ' '.join(f'{x:.2f}' for x in YF[i].mean((0, 2))))

section('4. failure mechanism per scene x eta (flow-reference, raw rollout summaries)')
print('  eta =', {k: np.round(e, 3).tolist() for k, e in zip(K, ETA)})
agg = {}
for f in glob.glob(D + '/raw/*.json'):
    d = json.load(open(f)); c = d['case']
    if c['controller_name'].startswith('flow'):
        s = d['summary'] or {}
        agg.setdefault((c['scene'], c['eta_index']), []).append(
            (d['outcome'], d['episode_length'], bool(s.get('ever_candidate_deadlock')), s.get('first_success_step')))
print('  scene eta | n | success timeout deadlock | ever-candidate-deadlock | median success step / horizon')
for sc in SCENES:
    for e in K:
        v = agg[(sc, e)]; out = np.array([x[0] for x in v])
        fs = np.array([np.nan if x[3] is None else x[3] for x in v], float)
        L = int(max(x[1] for x in v))
        msg = f'{np.nanmedian(fs):.0f}' if np.isfinite(fs).any() else '-'
        print(f'  {SHORT[sc]:4s} {e:2d} | {len(v)} | {np.mean(out == "success"):.2f} {np.mean(out == "timeout"):.2f} '
              f'{np.mean(out == "deadlock"):.2f} | {np.mean([x[2] for x in v]):.2f} | {msg}/{L}')

section('5. few-shot: k held-scene states (one seed half) choose eta; evaluate on the other states')
YF, YAl = tensor(QF, 'flow'), tensor(QA, 'flow')
rng = np.random.default_rng(20261011)
for i, s in enumerate(SCENES):
    prof = YF[i].mean((0, 1))
    vote = int(np.mean([YF[j].mean((0, 1)) for j in range(4) if j != i], 0).argmax())
    out = f'  {SHORT[s]:4s} oracle fixed {prof.max():.3f} | LOSO source vote {prof[vote]:.3f} |'
    for k in [1, 2, 4, 8]:
        vals = []
        for _ in range(3000):
            idx = rng.choice(40, k, replace=False); rest = np.setdiff1d(np.arange(40), idx)
            vals.append(YF[i][rest][:, :, int(YAl[i][idx].mean((0, 1)).argmax())].mean())
        out += f' k={k} {np.mean(vals):.3f}'
    print(out)

section('6. immediate executed-action difference vs outcome difference (flow-reference)')
AF, amask = z['action_feature'][:, K], z['entity_agent_mask']
for s in SCENES:
    dA, dQ = [], []
    for u in sorted(set(st[scene == s])):
        rows = [np.where((scene == s) & (st == u) & (ctl == f'flow_{op}'))[0][0] for op in OPS]
        n = int(amask[rows[0]].sum())
        for e in range(4):
            for a, b in itertools.combinations(rows, 2):
                if np.isfinite(QF[a, e]) and np.isfinite(QF[b, e]):
                    dA.append(np.linalg.norm(AF[a, e, :n, :2] - AF[b, e, :n, :2], axis=-1).mean())
                    dQ.append(abs(QF[a, e] - QF[b, e]))
    dA, dQ = np.asarray(dA), np.asarray(dQ); big = dQ >= .5
    print(f'  {SHORT[s]:4s} controller pairs {len(dA)}: Spearman(|da|,|dQ|) {spearmanr(dA, dQ)[0]:+.3f}; '
          f'|dQ|>=0.5: {big.sum()}, of which below-median |da| {int((dA[big] < np.median(dA)).sum())}; '
          f'smallest |da| with |dQ|>=0.5 {dA[big].min():.4f} vmax (median |da| {np.median(dA):.4f})')

section('7. source-only FULL model (encounter_same_all_labels) on each held scene, flow-reference')
P = np.load(D + '/development_predictions.npz', allow_pickle=True)
YAf, YBf = tensor(QA, 'flow'), tensor(QB, 'flow')
sxe = lambda Y: (lambda t: t - t.mean(0, keepdims=True) - t.mean(1, keepdims=True) + t.mean())(Y.mean(1))
for i, s in enumerate(SCENES):
    pre = f'{FOLD[s]}__{s}__all_held__flow__'
    cs, us = list(P[pre + 'controllers']), list(P[pre + 'state_uid'])
    pr = P[pre + 'encounter_same_all_labels__probability'][[us.index(u) for u in sorted(set(st[scene == s]))]]
    pr = pr[:, [cs.index(f'flow_{op}') for op in OPS]]                  # [40, 5, 4]
    true = YF[i]; src = [j for j in range(4) if j != i]
    op = lambda Y: Y.mean(0) - Y.mean(0).mean(0, keepdims=True)
    src_op = np.mean([op(YF[j]) for j in src], 0)
    print(f'  {SHORT[s]:4s} eta profile: predicted {np.round(pr.mean((0, 1)), 2)} true {np.round(true.mean((0, 1)), 2)} '
          f'source mean {np.round(np.mean([YF[j].mean((0, 1)) for j in src], 0), 2)}')
    print(f'       operator pattern corr: pred~held truth {corr(op(pr), op(true)):+.3f}; pred~source {corr(op(pr), src_op):+.3f}; '
          f'held truth~source {corr(op(true), src_op):+.3f}')
    print(f'       state x eta: corr pred~truth {corr(sxe(pr), sxe(true)):+.3f}; predicted energy {np.mean(sxe(pr) ** 2):.4f} '
          f'vs unbiased true energy {np.mean(sxe(YAf[i]) * sxe(YBf[i])):.4f}')
