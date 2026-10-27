"""Score methods on validation fresh states 8-23 (labels opened only here)."""
import json,pickle,glob,numpy as np
from tab import *
import critic as CR
sp=json.load(open('SPLIT.json'));data=pickle.load(open('critic_data.pkl','rb'));T=load(['design.jsonl','extra.jsonl','fresh.jsonl'])
TR=sp['train'];VA=sp['val'];F=list(range(8,24));TRS=list(range(16))
curve={n:np.nanmean(T[n][:,TRS],(0,1)) for n in TR}
C=np.array([data[n]['c'] for n in TR]);sig=C.std(0);sig[sig<1e-9]=1
def knn(c,k):
 d=np.sqrt((((C-c)/sig)**2).sum(1));near=[TR[i] for i in np.argsort(d,kind='stable')[:k]];return int(np.mean([curve[m] for m in near],0).argmax()),near
G=int(np.mean([curve[n] for n in TR],0).argmax())
choice={m:{} for m in ['global','knn1','knn2','knn3','critic_bce','critic_list']}
for v in VA:
 choice['global'][v]=G
 for k in (1,2,3):choice[f'knn{k}'][v]=knn(data[v]['c'],k)[0]
 for kind in ['bce','list']:
  f=np.mean([np.load(x)[v] for x in glob.glob(f'out/full_{kind}_*.npz')],0);choice['critic_'+kind][v]=int(CR.score(f,kind).argmax())
Y={v:T[v][:,F] for v in VA}
arr={m:np.array([Y[v][:,:,choice[m][v]].mean(0) for v in VA]) for m in choice}
# references (use validation labels)
own=[];copy=[]
for v in VA:
 y=Y[v];cnt=y.sum(0);a=(cnt.sum(0)[None]-cnt).argmax(-1);own.append(y[:,np.arange(16),a].mean(0))
 cands=[int(curve[n].argmax()) for n in TR];q=y.mean((0,1));copy.append(y[:,:,max(cands,key=lambda i:q[i])].mean(0))
arr['scene_own_best_ref']=np.array(own);arr['copy_right_train_scene_ref']=np.array(copy)
rng=np.random.default_rng(2026102694);M,Sn=8,16
si=rng.integers(Sn,size=(5000,M,Sn));sc=rng.integers(M,size=(5000,M));sh=rng.integers(Sn,size=(5000,M,Sn))
def boot(d):
 a=d[np.arange(M)[None,:,None],si].mean((1,2));b=d[sc[:,:,None],sh].mean((1,2));return np.percentile(a,[2.5,97.5]),np.percentile(b,[2.5,97.5])
res={}
print(f"global eta = {KEYS[G]} {np.round(CR.MENU[G],2).tolist()}")
print(f"{'method':28s} {'rate':>7s} {'minus global':>13s} {'state CI':>18s} {'scene->state CI':>20s}")
for m,a in arr.items():
 d=a-arr['global'];c1,c2=boot(d);res[m]=dict(rate=float(a.mean()),minus_global=float(d.mean()),ci_state=c1.tolist(),ci_scene_state=c2.tolist(),per_scene={v:float(x) for v,x in zip(VA,a.mean(1))},choices={v:KEYS[choice[m][v]] for v in VA} if m in choice else None)
 print(f"{m:28s} {a.mean()*100:7.2f} {d.mean()*100:+13.2f} [{c1[0]*100:+6.2f},{c1[1]*100:+6.2f}] [{c2[0]*100:+6.2f},{c2[1]*100:+6.2f}]")
for m in ['critic_list','critic_bce','knn3']:
 d=arr[m]-arr['knn3' if m!='knn3' else 'knn1'];c1,c2=boot(d);print(f"{m} minus {'knn3' if m!='knn3' else 'knn1'}: {d.mean()*100:+.2f} state [{c1[0]*100:+.2f},{c1[1]*100:+.2f}] scene->state [{c2[0]*100:+.2f},{c2[1]*100:+.2f}]")
print('\nper scene (fresh states 8-23):')
print(f"{'scene':28s}"+''.join(f"{m[:12]:>14s}" for m in arr))
for i,v in enumerate(VA):print(f"{v:28s}"+''.join(f"{(KEYS[choice[m][v]] if m in choice else ''):>7s}{arr[m][i].mean()*100:7.1f}" for m in arr))
json.dump(res,open('CRITIC_VALIDATION.json','w'),indent=1)
