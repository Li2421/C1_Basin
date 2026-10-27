import json, itertools, numpy as np, sys
import os; sys_path_fix = __import__("sys").path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load, splits, own_honest, single_honest, cross_loss, KEYS, ETA
M=load(sys.argv[1:]); S=splits(M); own={n:own_honest(M,S,n) for n in M}
inf=[n for n in M if 0.20<=M[n].mean(0).max()<=0.95 or own[n]>=0.95]
pool=[n for n in inf if n not in ('lit_crowded_random','lit_narrow_passage')]  # these two have ~100% for many etas
pool=[n for n in pool if own[n]>=0.25]
print('pool:',pool)
best={}
for k in range(3,9):
    res=[]
    for sub in itertools.combinations(pool,k):
        g=np.mean([own[n] for n in sub])-single_honest(M,S,list(sub)); res.append((g,sub))
    res.sort(reverse=True); best[k]=res[:3]
    for g,sub in res[:3]: print(f'k={k}: one global eta loses {100*g:+5.1f} pp vs own  {sub}')
key=['lit_circle_five','lit_swap8_obstacles','xsc_swap_n10','xsc_x_plus_w016','lit_swap8_free','xsc_swap_n6','xsc_swap_n4','xsc_x_plus_w025','xsc_x_plus_w040']
print('\ncross-use loss (row scene\'s own best eta applied to column scene), pp')
print(' '*22+''.join(f'{c[-10:]:>11s}' for c in key))
for i in key: print(f'{i[-22:]:22s}'+''.join(f'{100*cross_loss(M,S,own,i,j):11.1f}' if i!=j else f"{'-':>11s}" for j in key))
