from load import *
names=sorted(POOL)
Q={n:curve(n) for n in names}
cov={n:np.isfinite(Q[n]).sum() for n in names}
print('scene                                  grp  n49  mean  best  bestkey  best-sE29  best-median')
i29=KEYS.index('sE29')
for n in names:
 q=Q[n]
 if cov[n]<5:print(f"{n:42s} {POOL[n]['group']:4s} {cov[n]:3d}  (too few menu candidates)");continue
 print(f"{n:42s} {POOL[n]['group']:4s} {cov[n]:3d} {np.nanmean(q):.2f} {np.nanmax(q):.2f} {KEYS[int(np.nanargmax(q))]:6s} {np.nanmax(q)-q[i29] if np.isfinite(q[i29]) else float('nan'):+.2f} {np.nanmax(q)-np.nanmedian(q):+.2f}")
