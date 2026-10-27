import sys, numpy as np
sys.path.insert(0, '.')
from homogeneity import load, loso, spearman, boot_diff, KEYS
names, fam, f, y, q, etas = load(); N = len(names)
best = q.argmax(1)
rho = np.array([[spearman(q[i], q[j]) for j in range(N)] for i in range(N)])
reg = np.array([[q[i, best[i]] - q[i, best[j]] for j in range(N)] for i in range(N)])
for rmin, rmax in ((0.9, 0.05), (0.85, 0.02), (0.85, 0.05)):
    keep = set(range(N)); dropped = []
    while True:
        pairs = [(rho[i, j], i, j) for i in keep for j in keep if i < j and rho[i, j] >= rmin and reg[i, j] <= rmax + 1e-12 and reg[j, i] <= rmax + 1e-12]
        if not pairs: break
        _, i, j = max(pairs)
        mi = np.mean([rho[i, k] for k in keep if k != i]); mj = np.mean([rho[j, k] for k in keep if k != j])
        d = i if mi >= mj else j; keep.discard(d); dropped.append(names[d])
    idx = sorted(keep); ch = loso(idx, names, f, q, etas)
    pr, cr = boot_diff(idx, names, y, ch, 'ridge', reps=3000); pk, ck = boot_diff(idx, names, y, ch, 'knn', reps=3000)
    g = np.mean([q[i, ch[names[i]]['global_fixed']] for i in idx]); ins = np.mean([q[i].max() for i in idx])
    print(f'rho>={rmin} regret<={100*rmax:.0f}pp: n={len(idx)} dropped={dropped}')
    print(f'   global {100*g:.1f} in-sample {100*ins:.1f} | ridge-global {100*pr:+.2f} [{100*cr[0]:+.2f},{100*cr[1]:+.2f}] | knn-global {100*pk:+.2f} [{100*ck[0]:+.2f},{100*ck[1]:+.2f}]')
