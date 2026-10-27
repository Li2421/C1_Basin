from tab import *
import critic as CR
sp=json.load(open('SPLIT.json'));VA=sp['val'];G=json.load(open('GEN_FROZEN.json'));CV=json.load(open('CRITIC_VALIDATION.json'))
T=load(['design.jsonl','extra.jsonl','fresh.jsonl'])
Y={v:np.full((3,24,16),np.nan) for v in VA}
for l in open('gen.jsonl'):d=json.loads(l);Y[d['scene']][d['controller'],d['state'],d['proposal']]=d['success']
A=list(range(8,16));B=list(range(16,24));F=list(range(8,24))
# 1) near-duplicate noise
diffs=[]
for v in VA:
 P=(np.array(G[v]['proposals'])-CR.CENTER)/CR.RADIUS;q=Y[v][:,F].mean((0,1))
 for i in range(16):
  for j in range(i):
   if np.linalg.norm(P[i]-P[j])<.06:diffs.append(abs(q[i]-q[j]))
print('near-identical proposal pairs (normalized dist<.06):',len(diffs),'mean |rate diff|',round(np.mean(diffs)*100,1),'max',round(np.max(diffs)*100,1))
print('binomial SE of one proposal rate (48 cells, p=.6):',round(np.sqrt(.6*.4/48)*100,1),'pp')
# 2) honest oracle: pick on half A, score on half B (and vice versa)
def honest(tab):
 r=[]
 for X,Z in [(A,B),(B,A)]:r.append(tab[:,Z,int(tab[:,X].mean((0,1)).argmax())].mean())
 return np.mean(r)
rows=[]
for v in VA:
 ins=Y[v][:,F].mean((0,1)).max();hon=honest(Y[v][:,:,:]);mins=T[v][:,F].mean((0,1)).max();mhon=honest(T[v])
 rows.append((ins,hon,mins,mhon));print(f"{v:26s} proposals: in-sample best {ins*100:5.1f} honest best {hon*100:5.1f} | menu: in-sample best {mins*100:5.1f} honest best {mhon*100:5.1f}")
R=np.array(rows).mean(0)*100;print('MEAN proposals in-sample %.2f honest %.2f | menu in-sample %.2f honest %.2f'%tuple(R))
