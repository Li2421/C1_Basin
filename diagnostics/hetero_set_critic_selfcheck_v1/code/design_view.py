from tab import *
T=load(['design.jsonl']);S={s['name']:s for s in json.load(open('scenes_design.json'))};D=list(range(8))
i29=KEYS.index('sE29')
for n,s in S.items():
 if n not in T:continue
 t=T[n][:,D];done=np.isfinite(t).mean()
 q=np.nanmean(t,(0,1));o=np.argsort(-np.nan_to_num(q,nan=-1))
 loso=np.mean([np.nanmean(t[:,k,int(np.nanargmax(np.nanmean(np.delete(t,k,1),(0,1))))]) for k in range(8)])
 print(f"{n:28s} {s['cluster']:6s} done {done:.2f} mean {np.nanmean(q):.2f} | {KEYS[o[0]]}:{q[o[0]]:.2f} {KEYS[o[1]]}:{q[o[1]]:.2f} {KEYS[o[2]]}:{q[o[2]]:.2f} | sE29 {q[i29]:.2f} sE05 {q[KEYS.index('sE05')]:.2f} sE22 {q[KEYS.index('sE22')]:.2f} sE03 {q[KEYS.index('sE03')]:.2f} sE27 {q[KEYS.index('sE27')]:.2f} | LOSO-own {loso:.2f}")
