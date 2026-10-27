from inv import *
T=tables()
ks=['safety','eta1','eta9','eta13','eta15','sE01','sE03','sE05','sE13','sE15','sE17','sE19','sE22','sE27','sE29']
for k in ks:print(k,np.round(MENU[KEYS.index(k)],2))
good={n:t for n,(s,t) in T.items() if 0.15<np.nanmean(t)<0.85}
print()
print(f"{'scene':40s}"+''.join(f"{k[-4:]:>6s}" for k in ks))
for n,t in good.items():
 q=np.nanmean(t,(0,1));print(f"{n[-40:]:40s}"+''.join(f"{q[KEYS.index(k)]:6.2f}" for k in ks))
