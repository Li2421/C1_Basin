#!/usr/bin/env python3
from audit import *
from scipy.spatial import ConvexHull
def line_bounds(z,d):
 a=HS[:,:3]@d;b=-(HS[:,:3]@z+HS[:,3]);lo=-np.inf;hi=np.inf
 for aa,bb in zip(a,b):
  if aa>1e-12:hi=min(hi,bb/aa)
  elif aa<-1e-12:lo=max(lo,bb/aa)
 return lo,hi
def design():
 inv=read(HERE/'exact_q64_inventory.csv');targets={r['state_id']:np.array([float(r['target_eta'+str(i)]) for i in (1,2,3)]) for r in read(POINT/'selected_eta_targets.csv')}
 rows=[];paths=[]
 for sid in sorted(s for s in STATES if s.startswith('T0_WIDE')):
  rr=[r for r in inv if r['state_id']==sid];pp=[r for r in rr if r['B63']=='True'];nn=[r for r in rr if r['B63']=='False']
  P=np.array([(eta(r)-AFF)/SCALE for r in pp]);N=np.array([(eta(r)-AFF)/SCALE for r in nn]);c=(targets[sid]-AFF)/SCALE
  # Two observed robust kernel candidates: archived target and positive medoid.
  medoid_idx=int(np.argmin(cdist(P,P).sum(axis=1)));medoid=P[medoid_idx]
  for ci,center in enumerate((c,medoid)):
   first=int(np.argmax(np.linalg.norm(P-center,axis=1)));second=int(np.argmax(np.minimum(np.linalg.norm(P-center,axis=1),np.linalg.norm(P-P[first],axis=1))))
   for ti,idx in enumerate((first,second)):
    path=f'{sid}_star_{ci}_{ti}';p=P[idx];rawcenter=targets[sid] if ci==0 else eta(pp[medoid_idx]);rawend=eta(pp[idx]);paths.append(dict(state_id=sid,path_id=path,kind='star',kernel=ci,axis='',start=key(rawcenter),end=pp[idx]['eta_key']))
    for alpha in (.25,.5,.75):rows.append(dict(state_id=sid,eta=((1-alpha)*rawcenter+alpha*rawend).tolist(),phase='geometry_round1',kind='star',path_id=path,coordinate=alpha))
  # Native conditional lines through an already robust target, no near-line binning.
  for axis in range(3):
   d=np.eye(3)[axis];lo,hi=line_bounds(c,d);path=f'{sid}_axis{axis}'
   paths.append(dict(state_id=sid,path_id=path,kind='conditional',kernel=0,axis=axis,start=key(AFF+SCALE*(c+lo*d)),end=key(AFF+SCALE*(c+hi*d))))
   for t in sorted(set([lo,lo/2,0.,hi/2,hi])):rows.append(dict(state_id=sid,eta=(targets[sid]+SCALE*t*d).tolist(),phase='geometry_round1',kind='conditional',path_id=path,coordinate=float(t)))
  # Deepest two observed negatives in positive convex hull; test shortest facet-normal corridor.
  hull=ConvexHull(P);depth=-(N@hull.equations[:,:3].T+hull.equations[:,3]).max(axis=1)
  for ni in np.argsort(-depth)[:2]:
   if depth[ni]<=1e-9:continue
   neg=N[ni];clear=-(HS[:,:3]@neg+HS[:,3]);face=int(np.argmin(clear));d=HS[face,:3];_,lim=line_bounds(neg,d)
   path=f'{sid}_failure{int(ni)}';paths.append(dict(state_id=sid,path_id=path,kind='failure_corridor',kernel='',axis='',start=nn[ni]['eta_key'],end=key(AFF+SCALE*(neg+lim*d))))
   for alpha in (.33,.67,1.):rows.append(dict(state_id=sid,eta=(AFF+SCALE*(neg+alpha*lim*d)).tolist(),phase='geometry_round1',kind='failure_corridor',path_id=path,coordinate=alpha))
 write('targeted_probe_rounds/geometry_round1/paths.csv',paths)
 plan('geometry_round1',rows)
 dump('state_panels.json',{'current_scenario':list(STATES),'dense_panel':[s for s in STATES if s.startswith('T0_WIDE')],'point_manifest_sha256':sha(POINT/'final_source_split.json'),
  'selection':'unchanged40; unchanged8dense; no outcome-based new state replacement','cross_scenario':'pending compatibility and exact-conditioning audit'})
if __name__=='__main__':design()
