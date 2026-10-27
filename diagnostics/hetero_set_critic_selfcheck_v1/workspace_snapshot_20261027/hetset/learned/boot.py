"""Paired scene-then-state bootstrap of method differences for the learned planners (20 validation+test scenes)."""
import json,collections,numpy as np,os
HERE=os.path.dirname(os.path.abspath(__file__));ROOT=os.path.dirname(HERE)
Z=json.load(open(f'{ROOT}/FINAL_FROZEN.json'));V=Z['val']+Z['test']
L=json.load(open(f'{HERE}/LEARNED.json'));idx=list(range(8,8+len(L)))
R={}
for d in map(json.loads,open(f'{HERE}/eval.jsonl')):
  k=d['key'];k='pipe' if k.startswith('p') else (k[5:] if k.startswith('menu:') else k);R[(d['scene'],k,d['controller'],d['state'])]=d['success']
def arr(m,cs):return np.array([[np.mean([R[(v,m,c,s)] for c in cs]) for s in range(8,24)] for v in V])
rng=np.random.default_rng(0)
def boot(a,b,B=4000):
  d=a-b;out=[]
  for _ in range(B):
    sv=rng.integers(0,len(V),len(V));ss=rng.integers(0,16,(len(V),16))
    out.append(d[sv[:,None],ss].mean())
  return 100*d.mean(),100*np.percentile(out,2.5),100*np.percentile(out,97.5)
groups=[(x['name'],[8+i]) for i,x in enumerate(L)]+[('ALL_LEARNED',idx)]
for name,cs in groups:
  P=arr('pipe',cs)
  print(name,' | '.join(f"pipe-{m}: {a:+.1f} [{lo:+.1f},{hi:+.1f}]" for m,(a,lo,hi) in [(m,boot(P,arr(m,cs))) for m in ['rhr','rhr_goal','sE22','safety']]))
