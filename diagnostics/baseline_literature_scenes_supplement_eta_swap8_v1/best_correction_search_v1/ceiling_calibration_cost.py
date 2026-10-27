"""Cheap per-scene calibration: draw K shared candidates, try them on M training states x 7 controllers, keep the best,
score it on the other 16-M states (honest). Existing rows only; 400 random draws per setting."""
import sys, json
sys.path.insert(0, '/home/zhihan/research/Basin_C1/diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1/best_correction_search_v1')
from ceiling_common import *
from ceiling_driver2 import training_rows
import ceiling_analyze as ca
import numpy as np
shared = ['safety'] + [f'eta{i}' for i in range(16)] + [f'sE{i:02d}' for i in range(64)]
wide = [f'sE{i:02d}' for i in range(64)]
rng = np.random.default_rng(20261024); out = {}
for track, scene in [('swap8', s) for s in SCENES] + [('circle', 'lit_circle_five')]:
    parent_raw_index.cache_clear()
    rows = training_rows(('global' if track == 'swap8' else 'circle_global',), scene)
    S, states, ctls = ca.cube(rows, shared); ns = S.shape[0]; idx = {k: i for i, k in enumerate(shared)}
    def honest(pool, K, M, draws=400):
        vals = []
        for _ in range(draws):
            cand = rng.choice([idx[k] for k in pool], size=min(K, len(pool)), replace=False); st = rng.choice(ns, size=M, replace=False); rest = np.setdiff1d(np.arange(ns), st)
            sc = S[st][:, :, cand].mean(axis=(0, 1)); a = cand[int(np.flatnonzero(np.isclose(sc, sc.max()))[0])]
            vals.append(S[rest][:, :, a].mean())
        return float(np.mean(vals)), float(np.percentile(vals, 10))
    res = {}
    for K in (8, 16, 32, 64):
        for M in (4, 8):
            res[f'K{K}_M{M}'] = honest(wide, K, M)
    res['eta13_all_states'] = float(S[:, :, idx['eta13']].mean()); res['best_shared_all_states'] = float(S.mean(axis=(0, 1)).max())
    out[scene] = res
    print(scene.replace('lit_', ''), '| eta13 %.1f | best of 81 (in-sample) %.1f' % (100 * res['eta13_all_states'], 100 * res['best_shared_all_states']))
    for K in (8, 16, 32, 64):
        print('   K=%2d wide-box points:' % K, '  '.join('M=%d states -> %.1f%% (10th pct %.1f), cost %d rollouts' % (M, 100 * res[f'K{K}_M{M}'][0], 100 * res[f'K{K}_M{M}'][1], K * M * 7) for M in (4, 8)))
json.dump(out, open('/home/zhihan/.claude/jobs/be9710ae/tmp/preview/calibration_cost.json', 'w'), indent=1)
