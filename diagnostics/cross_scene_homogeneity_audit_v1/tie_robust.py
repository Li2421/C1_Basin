"""Tie-robust redundancy: J[i,j] = min over candidates of the larger of the two scenes' regrets."""
import sys, json, numpy as np
sys.path.insert(0, '.')
from homogeneity import load, loso, spearman, boot_diff, KEYS
names, fam, f, y, q, etas = load(); N = len(names)
mx = q.max(1)
reg = mx[:, None] - q                                 # regret of each candidate in each scene
rho = np.array([[spearman(q[i], q[j]) for j in range(N)] for i in range(N)])
J = np.array([[np.min(np.maximum(reg[i], reg[j])) for j in range(N)] for i in range(N)])
g = int(q.mean(0).argmax())
print('one correction (all-18 global best %s) within delta of each scene best:' % KEYS[g])
for d in (0.0, 0.02, 0.05):
    print('  delta %.0f pp: %d/18' % (100*d, int((reg[:, g] <= d + 1e-12).sum())))
print('scenes needing something else (>5 pp):', [(names[i], round(100*reg[i, g], 1)) for i in range(N) if reg[i, g] > 0.05 + 1e-12])
print('\npairs with a shared candidate within 2 pp of both bests and rho>=0.9:')
for i in range(N):
    for j in range(i+1, N):
        if J[i, j] <= 0.02 + 1e-12 and rho[i, j] >= 0.9: print(f'   {names[i]:20s} {names[j]:20s} rho {rho[i,j]:.2f} J {100*J[i,j]:.1f}')
res = {}
for rmin, dmax in ((0.9, 0.02), (0.9, 0.05), (0.85, 0.02)):
    keep = set(range(N)); dropped = []
    while True:
        pairs = [(rho[i, j], i, j) for i in keep for j in keep if i < j and rho[i, j] >= rmin and J[i, j] <= dmax + 1e-12]
        if not pairs: break
        _, i, j = max(pairs)
        mi = np.mean([rho[i, k] for k in keep if k != i]); mj = np.mean([rho[j, k] for k in keep if k != j])
        d = i if mi >= mj else j; keep.discard(d); dropped.append(names[d])
    idx = sorted(keep); ch = loso(idx, names, f, q, etas)
    pr, cr = boot_diff(idx, names, y, ch, 'ridge', reps=3000); pk, ck = boot_diff(idx, names, y, ch, 'knn', reps=3000)
    gl = np.mean([q[i, ch[names[i]]['global_fixed']] for i in idx]); ins = np.mean([q[i].max() for i in idx])
    dr = np.array([q[i, ch[names[i]]['ridge']] - q[i, ch[names[i]]['global_fixed']] for i in idx])
    print(f'\nrho>={rmin} J<={100*dmax:.0f}pp: n={len(idx)} dropped={dropped}')
    print(f'   kept={[names[i] for i in idx]}')
    print(f'   global {100*gl:.1f} in-sample {100*ins:.1f} | ridge-global {100*pr:+.2f} [{100*cr[0]:+.2f},{100*cr[1]:+.2f}] wins {int((dr>0).sum())} losses {int((dr<0).sum())} | knn-global {100*pk:+.2f} [{100*ck[0]:+.2f},{100*ck[1]:+.2f}]')
    res[f'rho{rmin}_J{dmax}'] = dict(dropped=dropped, kept=[names[i] for i in idx], global_rate=gl, in_sample=ins,
        ridge_minus_global=pr, ridge_ci=cr, knn_minus_global=pk, knn_ci=ck)
json.dump(dict(results=res, J=J.tolist(), rho=rho.tolist(), names=names), open('TIE_ROBUST_AUDIT.json', 'w'), indent=1)
