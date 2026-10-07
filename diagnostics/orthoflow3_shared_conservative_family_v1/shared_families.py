#!/usr/bin/env python3
"""Deterministic low-complexity analytic conservative-set fitters."""
import json, math
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist

HERE=Path(__file__).resolve().parent; D=HERE.parent
AFF=np.array([.875,0,.375]);SCALE=np.array([.75,1,.75])
HS=np.unique(np.array(json.load(open(D/'orthoflow3_t0_multiball_basin_learning_v1/geometry_constants.json'))['halfspaces']),axis=0)

def fix_frame(R):
 R=np.asarray(R,float)
 # Columns are orthonormal axes; deterministic signs and right handedness.
 u,_,vt=np.linalg.svd(R,full_matrices=False);R=u@vt
 for j in range(3):
  k=np.argmax(np.abs(R[:,j]))
  if R[k,j]<0:R[:,j]*=-1
 if np.linalg.det(R)<0:R[:,-1]*=-1
 return R
def pca_frame(X):
 if len(X)<2:return np.eye(3)
 cov=(X-X.mean(0)).T@(X-X.mean(0))/max(len(X)-1,1)
 val,R=np.linalg.eigh(cov);return fix_frame(R[:,np.argsort(val)[::-1]])
def pca2(X):
 if len(X)<2:return np.eye(2)
 cov=(X-X.mean(0)).T@(X-X.mean(0))/max(len(X)-1,1);val,R=np.linalg.eigh(cov);R=R[:,np.argsort(val)[::-1]]
 for j in range(2):
  k=np.argmax(np.abs(R[:,j]));
  if R[k,j]<0:R[:,j]*=-1
 if np.linalg.det(R)<0:R[:,-1]*=-1
 return R
def domain_contains(Z):return np.all(np.atleast_2d(Z)@HS[:,:3].T+HS[:,3]<=1e-10,axis=1)

def preimage(m,Z,ret):
 Z=np.atleast_2d(Z)
 if not ret:return Z
 a=np.array(m['anchor']);g=float(m['gamma'])
 return a+(Z-a)/g
def canonical(m,Z):
 Z=np.atleast_2d(Z);kind=m['family']
 if kind=='two_superbody_union':raise ValueError('Union has component-local canonical coordinates')
 if kind=='domain_minus_boundary_caps':return Z
 if kind in ('affine_superbody','superbody_with_cuts'):
  return ((Z-np.array(m['center']))@np.array(m['R']))/np.array(m['axes'])
 if kind=='affine_capsule_with_cuts':
  y=(Z-np.array(m['center']))@np.array(m['R']);scale=np.array([m['half_length']+m['radii'][0],m['radii'][1],m['radii'][2]])
  return y/np.maximum(scale,1e-8)
 if kind in ('native_conditional_band','rotated_asymmetric_slab'):
  t=(Z-np.array(m['center']))@np.array(m['R']);u=t[:,:2]/np.array(m['axes_tan'])
  X=np.column_stack([np.ones(len(u)),u]);lo=X@np.array(m['lower']);hi=X@np.array(m['upper']);mid=(lo+hi)/2;rad=np.maximum((hi-lo)/2,1e-8)
  return np.column_stack([u,(t[:,2]-mid)/rad])
 raise KeyError(kind)
def contains(m,Z,ret=False,domain=True):
 Z0=np.atleast_2d(Z);Z=preimage(m,Z0,ret);kind=m['family'];p=float(m['p'])
 if kind=='two_superbody_union':
  return np.any(np.column_stack([contains(c,Z0,ret,domain) for c in m['components']]),axis=1)
 ok=domain_contains(Z) if domain else np.ones(len(Z),bool)
 if kind=='domain_minus_boundary_caps':
  for cap in m['caps']:ok &= np.linalg.norm(Z-np.array(cap['center']),axis=1)>=float(cap['radius'])-1e-10
  # The base-minus-caps set need not be star-convex about the anchor.  Retained
  # membership is therefore the homothetic image intersected with the full set,
  # so erosion can never introduce a point excluded by the claimed full set.
  if ret:
   ok &= domain_contains(Z0)
   for cap in m['caps']:ok &= np.linalg.norm(Z0-np.array(cap['center']),axis=1)>=float(cap['radius'])-1e-10
  return ok
 if kind in ('affine_superbody','superbody_with_cuts'):
  u=canonical(m,Z);ok &= np.sum(np.abs(u)**p,axis=1)<=1+1e-10
 elif kind=='affine_capsule_with_cuts':
  y=(Z-np.array(m['center']))@np.array(m['R']);r=np.array(m['radii']);v=np.column_stack([np.maximum(np.abs(y[:,0])-float(m['half_length']),0)/r[0],y[:,1]/r[1],y[:,2]/r[2]])
  ok &= np.sum(np.abs(v)**p,axis=1)<=1+1e-10
 else:
  t=(Z-np.array(m['center']))@np.array(m['R']);u=t[:,:2]/np.array(m['axes_tan']);X=np.column_stack([np.ones(len(u)),u]);lo=X@np.array(m['lower']);hi=X@np.array(m['upper'])
  ok &= np.sum(np.abs(u)**p,axis=1)<=1+1e-10
  ok &= (t[:,2]>=lo-1e-10)&(t[:,2]<=hi+1e-10)&(hi>=lo)
 if m.get('cuts'):
  u=canonical(m,Z)
  for c in m['cuts']:ok &= u@np.array(c['q'])<=float(c['b'])+1e-10
 return ok
def violation(m,Z,ret=True):
 if m['family']=='two_superbody_union':return np.min(np.column_stack([violation(c,Z,ret) for c in m['components']]),axis=1)
 Z0=np.atleast_2d(Z);Z=preimage(m,Z0,ret);kind=m['family'];p=float(m['p']);vs=[]
 vs.extend(np.maximum(Z@HS[:,:3].T+HS[:,3],0).T)
 if kind=='domain_minus_boundary_caps':
  for cap in m['caps']:vs.append(np.maximum(float(cap['radius'])-np.linalg.norm(Z-np.array(cap['center']),axis=1),0))
  if ret:
   vs.extend(np.maximum(Z0@HS[:,:3].T+HS[:,3],0).T)
   for cap in m['caps']:vs.append(np.maximum(float(cap['radius'])-np.linalg.norm(Z0-np.array(cap['center']),axis=1),0))
  return np.sqrt(np.sum(np.asarray(vs)**2,axis=0))
 if kind in ('affine_superbody','superbody_with_cuts'):
  u=canonical(m,Z);vs.append(np.maximum(np.sum(np.abs(u)**p,axis=1)-1,0))
 elif kind=='affine_capsule_with_cuts':
  y=(Z-np.array(m['center']))@np.array(m['R']);r=np.array(m['radii']);u=np.column_stack([np.maximum(np.abs(y[:,0])-float(m['half_length']),0)/r[0],y[:,1]/r[1],y[:,2]/r[2]]);vs.append(np.maximum(np.sum(np.abs(u)**p,axis=1)-1,0))
 else:
  t=(Z-np.array(m['center']))@np.array(m['R']);u=t[:,:2]/np.array(m['axes_tan']);X=np.column_stack([np.ones(len(u)),u]);lo=X@np.array(m['lower']);hi=X@np.array(m['upper'])
  vs += [np.maximum(np.sum(np.abs(u)**p,axis=1)-1,0),np.maximum(lo-t[:,2],0),np.maximum(t[:,2]-hi,0)]
 if m.get('cuts'):
  u=canonical(m,Z)
  for c in m['cuts']:vs.append(np.maximum(u@np.array(c['q'])-float(c['b']),0))
 return np.sqrt(np.sum(np.asarray(vs)**2,axis=0))

def cluster_sizes(n):return sorted(set(max(2,min(n,int(round(f*n)))) for f in (.3,.45,.6,.8,1.0)))
def frame_profiles(P,anchor,native=False):
 order=np.argsort(np.linalg.norm(P-anchor,axis=1));out=[]
 for k in cluster_sizes(len(P)):
  C=P[order[:k]]
  centers=[anchor,C.mean(0)]
  frames=[np.eye(3)] if native else [np.eye(3),pca_frame(C),pca_frame(P)]
  for c in centers:
   for R in frames:
    A=np.abs((C-c)@R)
    for v in (np.max(A,axis=0),np.quantile(A,.8,axis=0),2*np.std((C-c)@R,axis=0)):
     v=np.maximum(v,.025);v=v/max(np.exp(np.mean(np.log(v))),1e-12)
     out.append((c,R,v,k))
 # exact float dedup
 seen=set();ans=[]
 for c,R,v,k in out:
  sig=tuple(np.round(np.r_[c,R.ravel(),v],10))
  if sig not in seen:seen.add(sig);ans.append((c,R,v,k))
 return ans
def weighted_score(m,P,N,W):
 pr=contains(m,P,True);nr=contains(m,N,True) if len(N) else np.zeros(0,bool)
 if nr.any() or not contains(m,np.array(m['anchor'])[None],True)[0]:return None
 pf=contains(m,P,False)
 if not pr.any():return None
 if m['family']=='domain_minus_boundary_caps':extent=1.-sum(float(c['radius'])**3 for c in m['caps'])
 elif m['family']=='two_superbody_union':extent=sum(np.prod(c['axes']) for c in m['components'])
 elif m['family'] in ('affine_superbody','superbody_with_cuts'):extent=float(np.prod(m['axes']))
 elif m['family']=='affine_capsule_with_cuts':extent=float((2*m['half_length']+2*m['radii'][0])*m['radii'][1]*m['radii'][2])
 else:extent=float(np.prod(m['axes_tan'])*max(np.array(m['upper'])[0]-np.array(m['lower'])[0],1e-6))
 return (float(W[pr].sum()),int(pr.sum()),float(np.log(max(extent,1e-15))),int(pf.sum()),-len(m.get('cuts',[])))

def fit_superbody(P,N,W,p,gamma,safety,family='affine_superbody',max_cuts=0):
 anchor=P[0];best=None
 for c,R,ratio,k in frame_profiles(P,anchor):
  if family=='affine_superbody':
   PP=anchor+(P-anchor)/gamma;NN=anchor+(N-anchor)/gamma if len(N) else N
   rp=np.sum(np.abs(((PP-c)@R)/ratio)**p,axis=1)**(1/p)
   rn=np.sum(np.abs(((NN-c)@R)/ratio)**p,axis=1)**(1/p) if len(N) else np.array([max(rp)*1.3])
   lam=max(.02,min(float(safety*np.min(rn)),2.5))
   m=dict(family=family,p=p,center=c.tolist(),R=R.tolist(),axes=(ratio*lam).tolist(),anchor=anchor.tolist(),gamma=gamma,cuts=[],parameter_count=9)
   sc=weighted_score(m,P,N,W)
   if sc is not None and (best is None or sc>best[0]):best=(sc,m)
   continue
  # Cut families deliberately use a broad positive body; exclusions handle negatives.
  PP=anchor+(P-anchor)/gamma
  req=np.sum(np.abs(((PP-c)@R)/ratio)**p,axis=1)**(1/p)
  for expand in (1.001,1.10,1.25):
   lam=max(.03,min(float(np.max(req)*expand),2.5));base=dict(family=family,p=p,center=c.tolist(),R=R.tolist(),axes=(ratio*lam).tolist(),anchor=anchor.tolist(),gamma=gamma,cuts=[],parameter_count=9+4*max_cuts)
   candidates=[[]]
   if max_cuts:
    preN=preimage(base,N,True);insideN=contains(base,N,True);ua=canonical(base,anchor[None])[0];UN=canonical(base,preN)
    cuts=[]
    for un,use in zip(UN,insideN):
     if not use:continue
     q=un-ua;d=np.linalg.norm(q)
     if d<1e-9:continue
     q=q/d
     for frac in (.55,.70,.85):cuts.append({'q':q.tolist(),'b':float(q@ua+frac*(q@un-q@ua))})
    # Unique and pre-rank by negatives removed then positives retained.
    uniq=[];seen=set()
    for cut in cuts:
     sig=tuple(np.round(np.r_[cut['q'],cut['b']],8))
     if sig in seen:continue
     seen.add(sig);mm=dict(base,cuts=[cut]);pn=int(contains(mm,N,True).sum());pp=int(contains(mm,P,True).sum());uniq.append((pn,-pp,sig,cut))
    uniq=[x[3] for x in sorted(uniq)[:24]]
    candidates += [[u] for u in uniq]
    if max_cuts>=2:
     candidates += [[uniq[i],uniq[j]] for i in range(len(uniq)) for j in range(i)][:276]
   for cuts in candidates:
    m=dict(base,cuts=cuts);sc=weighted_score(m,P,N,W)
    if sc is not None and (best is None or sc>best[0]):best=(sc,m)
 return None if best is None else best[1]

def fit_band(P,N,W,p,gamma,family,max_cuts):
 anchor=P[0];PP=anchor+(P-anchor)/gamma;order=np.argsort(np.linalg.norm(P-anchor,axis=1));best=None
 for k in cluster_sizes(len(P)):
  C=PP[order[:k]]
  if family=='native_conditional_band':
   frames=[np.eye(3)];centers=[np.array([C[:,0].mean(),C[:,1].mean(),0.])]
  else:
   frames=[pca_frame(C),pca_frame(PP)];centers=[C.mean(0),anchor]
  for R in frames:
   for c in centers:
    T=(C-c)@R;raw=np.maximum(np.max(np.abs(T[:,:2]),axis=0),.025);raw/=max(np.exp(np.mean(np.log(raw))),1e-12)
    r=np.sum(np.abs(T[:,:2]/raw)**p,axis=1)**(1/p)
    for sf in (1.001,1.15,1.35):
     axes=raw*max(np.max(r)*sf,.03);u=T[:,:2]/axes;X=np.column_stack([np.ones(len(u)),u]);beta=np.linalg.lstsq(X,T[:,2],rcond=1e-10)[0];res=T[:,2]-X@beta
     for wf in (1.02,1.25,1.55):
      lo=beta.copy();hi=beta.copy();lo[0]+=min(res)*wf-.002;hi[0]+=max(res)*wf+.002
      base=dict(family=family,p=p,center=c.tolist(),R=R.tolist(),axes_tan=axes.tolist(),lower=lo.tolist(),upper=hi.tolist(),anchor=anchor.tolist(),gamma=gamma,cuts=[],parameter_count=14+4*max_cuts)
      candidates=[[]]
      if max_cuts:
       insideN=contains(base,N,True);ua=canonical(base,anchor[None])[0];UN=canonical(base,preimage(base,N,True));cuts=[]
       for un,use in zip(UN,insideN):
        if not use:continue
        q=un-ua;d=np.linalg.norm(q)
        if d<1e-9:continue
        q=q/d
        for frac in (.55,.70,.85):cuts.append({'q':q.tolist(),'b':float(q@ua+frac*(q@un-q@ua))})
       uniq=[];seen=set()
       for cut in cuts:
        sig=tuple(np.round(np.r_[cut['q'],cut['b']],8))
        if sig in seen:continue
        seen.add(sig);mm=dict(base,cuts=[cut]);uniq.append((int(contains(mm,N,True).sum()),-int(contains(mm,P,True).sum()),sig,cut))
       uniq=[x[3] for x in sorted(uniq)[:24]];candidates += [[u] for u in uniq]
      for cuts in candidates:
       m=dict(base,cuts=cuts);sc=weighted_score(m,P,N,W)
       if sc is not None and (best is None or sc>best[0]):best=(sc,m)
 return None if best is None else best[1]

def capsule_frame(a,b,P):
 e1=b-a;d=np.linalg.norm(e1)
 if d<1e-8:return None
 e1=e1/d;cov=(P-P.mean(0)).T@(P-P.mean(0));val,V=np.linalg.eigh(cov);order=np.argsort(val)[::-1]
 e2=None
 for j in order:
  v=V[:,j]-e1*(e1@V[:,j]);n=np.linalg.norm(v)
  if n>1e-8:e2=v/n;break
 if e2 is None:
  v=np.eye(3)[np.argmin(np.abs(e1))];e2=v-e1*(e1@v);e2/=np.linalg.norm(e2)
 e3=np.cross(e1,e2);return fix_frame(np.column_stack([e1,e2,e3]))
def fit_capsule(P,N,W,p,gamma,max_cuts):
 anchor=P[0];PP=anchor+(P-anchor)/gamma;NN=anchor+(N-anchor)/gamma if len(N) else N;best=None
 # Robust endpoint pairs: anchor-farthest and global farthest positive pairs.
 dist=np.linalg.norm(P[:,None]-P[None,:],axis=2);pairs=[(0,int(np.argmax(dist[0]))),np.unravel_index(np.argmax(dist),dist.shape)]
 for i in np.argsort(-W)[:min(5,len(P))]:pairs.append((0,int(i)))
 seen=set()
 for ia,ib in pairs:
  if ia==ib:continue
  sig=tuple(sorted((int(ia),int(ib))))
  if sig in seen:continue
  seen.add(sig);a=PP[ia];b=PP[ib];R=capsule_frame(a,b,PP)
  if R is None:continue
  c=(a+b)/2;raw=(PP-c)@R
  for lf in (.75,1.,1.25):
   L=.5*np.linalg.norm(b-a)*lf
   resid=np.column_stack([np.maximum(np.abs(raw[:,0])-L,0),raw[:,1],raw[:,2]])
   for q in (.6,.8,1.0):
    ratio=np.maximum(np.quantile(np.abs(resid),q,axis=0),.02);ratio/=max(np.exp(np.mean(np.log(ratio))),1e-12)
    def req(X):
     y=(X-c)@R;u=np.column_stack([np.maximum(np.abs(y[:,0])-L,0)/ratio[0],y[:,1]/ratio[1],y[:,2]/ratio[2]])
     return np.sum(np.abs(u)**p,axis=1)**(1/p)
    rp=req(PP);rn=req(NN) if len(NN) else np.array([max(rp)*1.3])
    if max_cuts:scales=(max(rp)*1.001,max(rp)*1.15)
    else:scales=(max(.02,.9*np.min(rn)),)
    for lam in scales:
     base=dict(family='affine_capsule_with_cuts',p=p,center=c.tolist(),R=R.tolist(),half_length=float(L),radii=(ratio*lam).tolist(),anchor=anchor.tolist(),gamma=gamma,cuts=[],parameter_count=10+4*max_cuts)
     candidates=[[]]
     if max_cuts:
      insideN=contains(base,N,True);ua=canonical(base,anchor[None])[0];UN=canonical(base,preimage(base,N,True));cuts=[]
      for un,use in zip(UN,insideN):
       if not use:continue
       qv=un-ua;dn=np.linalg.norm(qv)
       if dn<1e-9:continue
       qv/=dn
       for frac in (.55,.70,.85):cuts.append({'q':qv.tolist(),'b':float(qv@ua+frac*(qv@un-qv@ua))})
      uniq=[];ss=set()
      for cut in cuts:
       sg=tuple(np.round(np.r_[cut['q'],cut['b']],8))
       if sg in ss:continue
       ss.add(sg);mm=dict(base,cuts=[cut]);uniq.append((int(contains(mm,N,True).sum()),-int(contains(mm,P,True).sum()),sg,cut))
      uniq=[x[3] for x in sorted(uniq)[:24]];candidates += [[u] for u in uniq]
     for cuts in candidates:
      m=dict(base,cuts=cuts);sc=weighted_score(m,P,N,W)
      if sc is not None and (best is None or sc>best[0]):best=(sc,m)
 return None if best is None else best[1]

def fit_two_union(P,N,W,p,gamma,safety):
 if len(P)<4:return None
 anchor=P[0];dist=np.linalg.norm(P[:,None]-P[None,:],axis=2);pairs=[(0,int(np.argmax(dist[0]))),np.unravel_index(np.argmax(dist),dist.shape)];best=None
 for ia,ib in pairs:
  if ia==ib:continue
  centers=P[[ia,ib]];lab=np.argmin(np.linalg.norm(P[:,None]-centers[None],axis=2),axis=1)
  if len(set(lab))<2:continue
  comps=[]
  for j in range(2):
   idx=np.flatnonzero(lab==j);idx=np.r_[idx[idx!=[ia,ib][j]],[ia,ib][j]]
   idx=np.r_[[idx[-1]],idx[:-1]] # cluster anchor first
   m=fit_superbody(P[idx],N,W[idx],p,gamma,safety,'affine_superbody',0)
   if m is None:break
   comps.append(m)
  if len(comps)!=2:continue
  m=dict(family='two_superbody_union',p=p,gamma=gamma,anchor=anchor.tolist(),components=comps,cuts=[],parameter_count=18)
  sc=weighted_score(m,P,N,W)
  if sc is not None and (best is None or sc>best[0]):best=(sc,m)
 return None if best is None else best[1]

def deterministic_clusters(X,k):
 centers=[0]
 while len(centers)<min(k,len(X)):
  d=np.min(np.linalg.norm(X[:,None]-X[centers][None],axis=2),axis=1);centers.append(int(np.argmax(d)))
 C=X[centers].copy()
 for _ in range(40):
  lab=np.argmin(np.linalg.norm(X[:,None]-C[None],axis=2),axis=1);new=np.array([X[lab==j].mean(0) if np.any(lab==j) else C[j] for j in range(len(C))])
  if np.max(np.abs(new-C))<1e-12:break
  C=new
 return [X[lab==j] for j in range(len(C)) if np.any(lab==j)]
def fit_domain_caps(P,N,W,p,gamma,max_cuts):
 anchor=P[0];PP=anchor+(P-anchor)/gamma;NN=anchor+(N-anchor)/gamma if len(N) else N
 active=NN[domain_contains(NN)]
 if not len(active):return dict(family='domain_minus_boundary_caps',p=2,anchor=anchor.tolist(),gamma=gamma,caps=[],parameter_count=3)
 k=min(max_cuts,len(active));groups=deterministic_clusters(active,k);options=[]
 for G in groups:
  mean=G.mean(0);cand=[]
  for face in HS:
   n=face[:3];nn=n@n;base=mean-(mean@n+face[3])*n/nn
   for shift in (0.,.05,.10,.20,.35,.50):
    c=base+shift*n/np.sqrt(nn);r=float(np.max(np.linalg.norm(G-c,axis=1))+.003)
    if np.linalg.norm(anchor-c)<r+1e-8:continue
    keep=np.linalg.norm(PP-c,axis=1)>=r
    cand.append((float(W[keep].sum()),int(keep.sum()),-r,c,r))
  if not cand:return None
  cand.sort(reverse=True,key=lambda x:x[:3]);options.append(cand[:4])
 best=None
 import itertools
 for combo in itertools.product(*options):
  caps=[{'center':x[3].tolist(),'radius':x[4]} for x in combo];m=dict(family='domain_minus_boundary_caps',p=2,anchor=anchor.tolist(),gamma=gamma,caps=caps,parameter_count=3+4*max_cuts)
  sc=weighted_score(m,P,N,W)
  if sc is not None and (best is None or sc>best[0]):best=(sc,m)
 return None if best is None else best[1]

def fit(family,fit_rows,p,gamma,max_cuts=0,safety=.9):
 P=[];N=[];W=[]
 for r in fit_rows:
  z=(np.array([float(r[f'eta{i}']) for i in (1,2,3)])-AFF)/SCALE
  if r['B63']=='True':
   P.append(z);phase=(r.get('phases','')+';'+r.get('sources','')).lower();W.append(2.0 if ('cross_transfer' in phase or 'common_core' in phase or 'val_neighbor' in phase) else 1.0)
  else:N.append(z)
 if not P:return None
 P=np.array(P);N=np.array(N).reshape(-1,3);W=np.array(W)
 # The first selected row is the frozen robust anchor; reorder positives accordingly.
 first=fit_rows[0]
 if first['B63']=='True':
  a=(np.array([float(first[f'eta{i}']) for i in (1,2,3)])-AFF)/SCALE;j=np.argmin(np.linalg.norm(P-a,axis=1));P[[0,j]]=P[[j,0]];W[[0,j]]=W[[j,0]]
 if family in ('affine_superbody','superbody_with_cuts'):return fit_superbody(P,N,W,p,gamma,safety,family,max_cuts)
 if family=='domain_minus_boundary_caps':return fit_domain_caps(P,N,W,p,gamma,max_cuts)
 if family=='affine_capsule_with_cuts':return fit_capsule(P,N,W,p,gamma,max_cuts)
 if family=='two_superbody_union':return fit_two_union(P,N,W,p,gamma,safety)
 return fit_band(P,N,W,p,gamma,family,max_cuts)

def approx_extent(m,cloud,known=None):
 mask=contains(m,cloud,True);X=cloud[mask]
 if known is not None:
  K=np.atleast_2d(known);X=np.vstack([X,K[contains(m,K,True)]]) if len(K) else X
 if len(X)<2:return dict(diameter=0.,volume_fraction=float(mask.mean()),points=int(mask.sum()))
 if len(X)>512:
  # Deterministic farthest-point compression before exact pairwise diameter.
  idx=[0]
  d=np.linalg.norm(X-X[0],axis=1)
  for _ in range(1,512):
   j=int(np.argmax(d));idx.append(j);d=np.minimum(d,np.linalg.norm(X-X[j],axis=1))
  X=X[idx]
 return dict(diameter=float(np.max(pdist(X))),volume_fraction=float(mask.mean()),points=int(mask.sum()))

def model_to_json(m):return m
