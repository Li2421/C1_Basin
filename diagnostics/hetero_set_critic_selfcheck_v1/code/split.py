"""Pre-registered eligibility + random split (written before extra-sweep results were viewed)."""
from tab import *
S={s['name']:s for f in ['scenes_design.json','scenes_extra.json'] for s in json.load(open(f))}
T=load(['design.jsonl','extra.jsonl']);D=list(range(8))
elig={}
for n,s in S.items():
 if s['cluster']=='circle' or n not in T:continue
 t=T[n][:,D]
 if not np.isfinite(t).all():continue
 q=t.mean((0,1))
 if .10<=q.mean()<=.90 and q.max()>=.30:elig[n]=s['cluster']
rng=np.random.default_rng(2026102692);val=[];train=[]
for c in ['plus','box','obst','swap']:
 ns=sorted(n for n,k in elig.items() if k==c);pick=set(rng.choice(ns,2,replace=False).tolist());val+=sorted(pick);train+=[n for n in ns if n not in pick]
json.dump(dict(train=train,val=val,eligible=elig,excluded=[n for n in S if n not in elig]),open('SPLIT.json','w'),indent=1)
print('train',len(train),train);print('val',len(val),val);print('excluded',[n for n in S if n not in elig])
