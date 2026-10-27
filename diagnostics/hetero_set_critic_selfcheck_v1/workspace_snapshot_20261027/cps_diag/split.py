from load import *
names=sorted(POOL);i29=KEYS.index('sE29');rng=np.random.default_rng(0)
print('honest own-best minus sE29 (choose on half the states, score on other half; 200 splits)')
res={}
for n in names:
 t=cell_table(n)  # 3,S,49
 ok=np.isfinite(t).mean((0,1))>=.9
 if ok.sum()<40:continue
 S=t.shape[1];g=[]
 for _ in range(200):
  p=rng.permutation(S);A,B=p[:S//2],p[S//2:]
  qa=np.nanmean(t[:,A],(0,1));qa[~ok]=-1;a=int(qa.argmax())
  qb=np.nanmean(t[:,B],(0,1));g.append(qb[a]-qb[i29])
 g=np.array(g);res[n]=g.mean()
 print(f"{n:48s} {POOL[n]['group']:4s} gain {g.mean():+.3f}  P(gain>0) {np.mean(g>0):.2f} P(<0) {np.mean(g<0):.2f}")
print('validation scenes (16 eval states):')
for v in VAL:
 t=val_y(v,False);S=t.shape[1];g=[]
 for _ in range(200):
  p=rng.permutation(S);A,B=p[:S//2],p[S//2:]
  a=int(t[:,A].mean((0,1)).argmax());qb=t[:,B].mean((0,1));g.append(qb[a]-qb[i29])
 g=np.array(g);print(f"{v:48s} gain {g.mean():+.3f}  P(gain>0) {np.mean(g>0):.2f}")
