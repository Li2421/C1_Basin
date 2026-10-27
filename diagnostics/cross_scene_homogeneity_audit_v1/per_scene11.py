import sys, numpy as np
sys.path.insert(0, '.')
from homogeneity import load, loso, KEYS
names, fam, f, y, q, etas = load()
keep = ['lit_narrow_passage','lit_crowded_random','lit_circle_five','lit_swap8_obstacles','xsc_swap_n4','xsc_swap_n12','xsc_swap_n8_dense','xsc_gap_n4','xsc_gap_n6','xsc_double_n4','xsc_offset_n4']
idx = [names.index(n) for n in keep]
ch = loso(idx, names, f, q, etas)
for i in idx:
    c = ch[names[i]]
    print(f"{names[i]:22s} k={c['k']} pen={c['penalty']:5} | global {KEYS[c['global_fixed']]:5s} {100*q[i,c['global_fixed']]:5.1f} | knn {KEYS[c['knn']]:5s} {100*q[i,c['knn']]:5.1f} ({100*(q[i,c['knn']]-q[i,c['global_fixed']]):+5.1f}) | ridge {KEYS[c['ridge']]:5s} {100*q[i,c['ridge']]:5.1f} ({100*(q[i,c['ridge']]-q[i,c['global_fixed']]):+5.1f}) | best {100*q[i].max():5.1f}")
# frozen choices for a prospective test: models fitted on all 11 (dedup pool) and on all 18
from homogeneity import select
for label, pool in (('dedup11', idx), ('all18', list(range(len(names))))):
    s = select(f[pool], q[pool], f[pool][0], etas)   # target irrelevant for global/k/penalty
    print(label, 'global', KEYS[s['global_fixed']], 'k', s['k'], 'penalty', s['penalty'])
