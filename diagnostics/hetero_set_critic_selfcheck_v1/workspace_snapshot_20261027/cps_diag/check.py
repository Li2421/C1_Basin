from load import *
names=sorted(POOL)
C=np.array([POOL[n]['c_prime'] for n in names]);sig=C.std(0);sig[sig<1e-9]=1
def pick(srcs,c,k=None):
 Cs=np.array([POOL[n]['c_prime'] for n in srcs]);d=np.sqrt((((Cs-c)/sig)**2).sum(1));near=[srcs[i] for i in np.argsort(d,kind='stable')[:k]] if k else srcs
 cv=np.array([curve(n) for n in near]);f=np.isfinite(cv);q=np.where(f,cv,0).sum(0)/np.maximum(f.sum(0),1);q[f.sum(0)==0]=np.nan;return int(np.nanargmax(q)),near
tot={}
for k in [None,1,2,3]:
 r=[]
 for v in VAL:
  a,near=pick(names,cprime(v),k);r.append(val_y(v)[:,:,a].mean())
  if k==1:print(v,KEYS[a],near)
 print(k,np.mean(r))
