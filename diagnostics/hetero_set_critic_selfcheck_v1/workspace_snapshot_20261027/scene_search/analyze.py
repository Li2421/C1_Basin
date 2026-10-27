"""Scratch analysis: one global eta versus each scene's own best eta (honest half-splits)."""
import json, itertools, sys, collections
import numpy as np

KEYS = ['safety'] + [f'eta{i}' for i in range(16)] + [f'sE{i:02d}' for i in range(32)]
acts = json.load(open('/home/zhihan/research/Basin_C1/diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1/best_correction_search_v1/ACTIONS.json'))['lit_swap8_free']
ETA = {k: acts[k]['eta_a'] for k in KEYS}
CTL = ('centralized_sampling0', 'gradient_mpc0', 'DM0')
NAMES_CN = {'lit_circle_five': '圆形五机', 'lit_crowded_random': '拥挤随机', 'lit_narrow_passage': '窄通道', 'lit_swap8_free': '八机无障碍',
            'lit_swap8_moving': '八机移动障碍', 'lit_swap8_obstacles': '八机静态障碍', 'xsc_swap_n4': '对穿4机', 'xsc_swap_n6': '对穿6机',
            'xsc_swap_n10': '对穿10机', 'xsc_swap_n12': '对穿12机', 'xsc_swap_n8_dense': '对穿8机加密'}


def load(paths):
    M = {}
    lab = json.load(open('/home/zhihan/research/Basin_C1/diagnostics/cross_scene_correction_v1/LABEL_TABLES.json'))
    for n, t in lab.items():
        if not all(c in t['controllers'] for c in CTL): continue
        ci = [t['controllers'].index(c) for c in CTL]
        y = np.where(np.asarray(t['unknown'])[ci] > 0, 0, np.asarray(t['y'], float)[ci])
        M[n] = y.mean(0)  # states x 49, controller-equal
    for p in paths:
        cell = collections.defaultdict(dict)
        for line in open(p):
            d = json.loads(line)
            if d['solo'] is not None: continue
            cell[(d['scene'], d['state'])].setdefault(d['action'], []).append(0 if d['numerical'] else d['success'])
        by = collections.defaultdict(dict)
        for (s, st), v in cell.items():
            if len(v) == 49 and all(len(x) == 3 for x in v.values()): by[s][st] = [np.mean(v[k]) for k in KEYS]
        for s, rows in by.items(): M[s] = np.array([rows[k] for k in sorted(rows)])
    return M


def splits(M, reps=200, seed=0):
    rng = np.random.default_rng(seed)
    return {n: [rng.permutation(len(M[n])) for _ in range(reps)] for n in M}


def own_honest(M, S, n):
    v = []
    for p in S[n]:
        h = len(p) // 2; a, b = p[:h], p[h:]
        for x, y in ((a, b), (b, a)):
            s = M[n][x].mean(0); t = np.flatnonzero(s >= s.max() - 1e-12); v.append(M[n][y][:, t].mean())
    return float(np.mean(v))


def single_honest(M, S, sub):
    v = []
    for r in range(len(S[sub[0]])):
        tr = []; sc = []
        for n in sub:
            p = S[n][r]; h = len(p) // 2; tr.append(M[n][p[:h]].mean(0)); sc.append(M[n][p[h:]].mean(0))
        a = int(np.argmax(np.mean(tr, 0))); v.append(np.mean([s[a] for s in sc]))
    return float(np.mean(v))


def cross_loss(M, S, own, i, j):
    """use i's own best (chosen on i's training half) on j's scoring half, versus j's own honest best"""
    v = []
    for r in range(len(S[i])):
        p = S[i][r]; s = M[i][p[:len(p) // 2]].mean(0); a = int(np.argmax(s))
        q = S[j][r]; v.append(M[j][q[len(q) // 2:]][:, a].mean())
    return own[j] - float(np.mean(v))


if __name__ == '__main__':
    M = load(sys.argv[1:])
    S = splits(M)
    own = {n: own_honest(M, S, n) for n in M}
    print(f"{'scene':26s} {'states':>6s} {'own':>6s}  best in-sample eta")
    for n in sorted(M):
        q = M[n].mean(0); b = int(np.argmax(q))
        top = np.argsort(-q)[:3]
        print(f"{n:26s} {len(M[n]):6d} {100*own[n]:6.1f}  " + ' | '.join(f"{KEYS[i]} {np.round(ETA[KEYS[i]],2).tolist()} {100*q[i]:.0f}" for i in top))
    json.dump(dict(M={n: M[n].tolist() for n in M}, own=own), open('analysis_tables.json', 'w'))
