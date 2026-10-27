"""Honest heterogeneity check: choices on design states (0-7), scoring on fresh states."""
from tab import *
import sys
sp=json.load(open('SPLIT.json'));T=load(['design.jsonl','extra.jsonl','fresh.jsonl'])
D=list(range(8))
def check(names,fresh,label,rng):
 G=np.mean([T[n][:,D].mean((0,1)) for n in names],0);g=int(G.argmax())
 rows=[];A=[];B=[]
 for n in names:
  o=int(T[n][:,D].mean((0,1)).argmax());f=T[n][:,fresh]
  if not np.isfinite(f[:,:,[o,g]]).all():print('missing',n);continue
  a=f[:,:,o].mean(0);b=f[:,:,g].mean(0);A.append(a);B.append(b);rows.append((n,KEYS[o],a.mean(),b.mean()))
 A=np.array(A);B=np.array(B);d=A-B;M,Sn=d.shape
 bs=[d[np.arange(M)[:,None],rng.integers(Sn,size=(M,Sn))].mean() for _ in range(5000)]
 hb=[]
 for _ in range(5000):
  sc=rng.integers(M,size=M);hb.append(d[sc[:,None],rng.integers(Sn,size=(M,Sn))].mean())
 print(f'== {label}: {M} scenes, global eta {KEYS[g]} {np.round(MENU[g],2).tolist() if "MENU" in globals() else ""}')
 for n,o,a,b in rows:print(f'  {n:28s} own {o:6s} {a*100:5.1f}  global {b*100:5.1f}  diff {(a-b)*100:+5.1f}')
 print(f'  own-best minus global: {d.mean()*100:+.2f}  state CI [{np.percentile(bs,2.5)*100:+.2f},{np.percentile(bs,97.5)*100:+.2f}]  scene->state CI [{np.percentile(hb,2.5)*100:+.2f},{np.percentile(hb,97.5)*100:+.2f}]')
 return dict(global_eta=KEYS[g],gap=float(d.mean()),ci_state=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],ci_scene_state=[float(np.percentile(hb,2.5)),float(np.percentile(hb,97.5))],rows=rows)
if __name__=='__main__':
 rng=np.random.default_rng(2026102693)
 r=dict(train=check(sp['train'],list(range(8,16)),'TRAIN (fresh states 8-15)',rng),val=check(sp['val'],list(range(8,24)),'VALIDATION (fresh states 8-23)',rng))
 json.dump(r,open('VERIFY.json','w'),indent=1,default=str)
