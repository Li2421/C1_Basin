"""Is one fixed correction good in all four scenes? Uses the shared candidate set (safety, history 16, 64 extended-box points)
that every scene ran on its training cohort (seed 0). Existing rows only."""
import sys, json
sys.path.insert(0, '/home/zhihan/research/Basin_C1/diagnostics/baseline_literature_scenes_supplement_eta_swap8_v1/best_correction_search_v1')
from ceiling_common import *
from ceiling_driver2 import training_rows, seed0_stages, rank, actions_store
import numpy as np
shared = ['safety'] + [f'eta{i}' for i in range(16)] + [f'sE{i:02d}' for i in range(64)]
scenes = [('swap8', s) for s in SCENES] + [('circle', 'lit_circle_five')]
table = {}
for track, scene in scenes:
    parent_raw_index.cache_clear()
    rows = [r for r in training_rows((track and ('global' if track == 'swap8' else 'circle_global'),), scene)]
    table[scene] = {r['key']: r['success'] for r in rank(rows, shared)}
best = {s: max(v.values()) for s, v in table.items()}
regret = {k: {s: best[s] - table[s][k] for s in table} for k in shared}
order = sorted(shared, key=lambda k: max(regret[k].values()))
store = actions_store()['lit_swap8_free']
print('per-scene best among shared candidates:', {s: (max(v, key=v.get), round(100 * best[s], 1)) for s, v in table.items()})
print('best minimax-regret shared corrections:')
for k in order[:6]:
    print('  %-6s eta=%s  max regret %.1f pp  per scene %s' % (k, [round(x, 2) for x in store[k]['eta_a']], 100 * max(regret[k].values()), {s.replace('lit_', ''): round(100 * table[s][k], 1) for s in table}))
for k in ('eta13', 'eta9', 'sE41'):
    print('  ref %-5s max regret %.1f pp per scene %s' % (k, 100 * max(regret[k].values()), {s.replace('lit_', ''): round(100 * table[s][k], 1) for s in table}))
json.dump(dict(table=table, best=best, minimax=order[:10]), open('/home/zhihan/.claude/jobs/be9710ae/tmp/preview/shared_across_scenes.json', 'w'), indent=1)
