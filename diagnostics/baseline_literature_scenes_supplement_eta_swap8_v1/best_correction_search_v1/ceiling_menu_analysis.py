"""How much of the per-state headroom a small shared menu captures (honest: menu and per-state choice fitted on the
other controllers, scored on the held-out controller). Existing rows only."""
import sys
sys.path.insert(0, '/home/zhihan/research/Basin_C1/diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1/best_correction_search_v1')
from ceiling_common import *
from ceiling_driver2 import training_rows, fixed_keys, schedule_keys, seed0_stages, actions_store
import ceiling_analyze as ca
import numpy as np, json

def greedy_menu(S, k):
    # S: states x controllers x actions; maximize mean over (s, c) of max over menu of S[s, c, a] chosen per state by mean over c
    menu = []
    for _ in range(k):
        best, val = None, -1
        for a in range(S.shape[2]):
            if a in menu: continue
            m = menu + [a]; P = S[:, :, m].mean(axis=1)  # state x menu (mean over controllers)
            pick = P.argmax(axis=1); v = S[np.arange(S.shape[0]), :, np.array(m)[pick]].mean()
            if v > val + 1e-12: best, val = a, v
        menu.append(best)
    return menu

def honest(S, k):
    ns, nc, na = S.shape; vals = []
    for c in range(nc):
        T = np.delete(S, c, axis=1); menu = greedy_menu(T, k)
        P = T[:, :, menu].mean(axis=1)
        for s in range(ns):
            sc = P[s]; best = np.flatnonzero(np.isclose(sc, sc.max()))
            vals.append(S[s, c, np.array(menu)[best]].mean())
    return float(np.mean(vals))

out = {}
for track, scenes in (('swap8', SCENES), ('circle', ('lit_circle_five',))):
    for scene in scenes:
        parent_raw_index.cache_clear()
        rows = training_rows(seed0_stages(track), scene); keys = fixed_keys(scene, True) + schedule_keys(scene)
        S, states, ctls = ca.cube(rows, keys)
        res = {k: honest(S, k) for k in (1, 2, 3, 4, 6)}
        full = greedy_menu(S, 4); store = actions_store()[scene]
        P = S[:, :, full].mean(axis=1); pick = P.argmax(axis=1)
        out[scene] = dict(honest_menu=res, menu4=[keys[a] for a in full], menu4_eta=[[store[keys[a]]['eta_a'], store[keys[a]]['eta_b'], store[keys[a]]['theta']] for a in full],
                          per_state_choice=[keys[full[i]] for i in pick], states=states)
        print(scene, 'honest state-conditioned menu policy:', {k: round(100 * v, 1) for k, v in res.items()})
        print('   in-sample 4-menu:', [keys[a] for a in full])
        print('   per-state choice:', dict(zip(states, [keys[full[i]] for i in pick])))
json.dump(out, open('/home/zhihan/.claude/jobs/be9710ae/tmp/preview/menu_analysis.json', 'w'), indent=1)
