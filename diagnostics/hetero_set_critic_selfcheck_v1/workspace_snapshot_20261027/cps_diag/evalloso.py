from load import *;import json,glob
P=json.load(open('pools.json'));names=P['all'];T=json.load(open('loso_targets.json'))
C=np.array([cprime(n) for n in names]);sig=C.std(0);sig[sig<1e-9]=1
def base(src,c,k=None):
 Cs=np.array([cprime(n) for n in src]);d=np.sqrt((((Cs-c)/sig)**2).sum(1));near=[src[i] for i in np.argsort(d,kind='stable')[:k]] if k else src
 cv=np.array([curve(n) for n in near]);f=np.isfinite(cv);q=np.where(f,cv,0).sum(0)/np.maximum(f.sum(0),1);q[f.sum(0)==0]=-1;return int(q.argmax())
rows_=[]
for t in T:
 src=[n for n in names if n!=t];q=curve(t);q0=np.nan_to_num(q,nan=np.nanmean(q))
 r=dict(scene=t,oracle=np.nanmax(q))
 for m,k in [('global',None),('knn1',1),('knn2',2),('knn3',3)]:r[m]=q0[base(src,cprime(t),k)]
 for kind in ['bce','list']:
  f=np.mean([np.load(x)[t] for x in glob.glob(f'out/loso_{t}_{kind}_*.npz')],0)
  sc=(1/(1+np.exp(-f))).mean((0,1)) if kind=='bce' else (lambda p:(p/p.sum(-1,keepdims=True)).mean((0,1)))(np.exp(f-f.max(-1,keepdims=True)))
  r[kind]=q0[int(sc.argmax())]
 rows_.append(r)
ms=['global','knn1','knn2','knn3','bce','list','oracle']
print(f"{'scene':44s}"+''.join(f"{m:>8s}" for m in ms))
for r in rows_:print(f"{r['scene'][:44]:44s}"+''.join(f"{r[m]*100:8.1f}" for m in ms))
A=np.array([[r[m] for m in ms] for r in rows_])
print(f"{'MEAN':44s}"+''.join(f"{x*100:8.2f}" for x in A.mean(0)))
rng=np.random.default_rng(1)
for m in ['knn1','knn2','knn3','bce','list']:
 d=A[:,ms.index(m)]-A[:,0];bs=[d[rng.integers(len(d),size=len(d))].mean() for _ in range(5000)]
 print(f"{m} - global {d.mean()*100:+.2f} [{np.percentile(bs,2.5)*100:+.2f},{np.percentile(bs,97.5)*100:+.2f}]  W/L {np.sum(d>0)}/{np.sum(d<0)}")
