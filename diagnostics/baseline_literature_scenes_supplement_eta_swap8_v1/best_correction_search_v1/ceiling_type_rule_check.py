import sys
sys.path.insert(0, '/home/zhihan/research/Basin_C1/diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1/best_correction_search_v1')
from ceiling_common import *
from ceiling_driver2 import training_rows, fixed_keys, schedule_keys, seed0_stages, actions_store, rank
import ceiling_analyze as ca
import numpy as np
for scene in SCENES:
    parent_raw_index.cache_clear()
    rows = training_rows(seed0_stages('swap8'), scene); fixed = fixed_keys(scene, True)
    order = [r['key'] for r in rank(rows, fixed)]
    for m in (5, 10, 20):
        menu = order[:m]
        S, states, ctls = ca.cube(rows, menu)   # states x ctls x menu, columns in rank order
        ns, nc, _ = S.shape
        # honest: per controller choose on 15 states (ties -> overall rank order = lowest index), score held-out state
        vals = []; fixedv = []
        for c in range(nc):
            for s in range(ns):
                sc = np.delete(S[:, c], s, axis=0).mean(axis=0); a = int(np.flatnonzero(np.isclose(sc, sc.max()))[0])
                vals.append(S[s, c, a])
                f = np.delete(S, s, axis=0).mean(axis=(0, 1)); fixedv.append(S[s, c, int(np.flatnonzero(np.isclose(f, f.max()))[0])])
        full = [menu[int(np.flatnonzero(np.isclose(S[:, c].mean(0), S[:, c].mean(0).max()))[0])] for c in range(nc)]
        print(scene, 'menu', m, 'honest per-controller %.1f vs fixed %.1f' % (100 * np.mean(vals), 100 * np.mean(fixedv)), '| per-type picks', dict(zip(ctls, full)))
