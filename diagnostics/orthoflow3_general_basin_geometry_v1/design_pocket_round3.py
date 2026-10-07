#!/usr/bin/env python3
from audit import *
from design_geometry import line_bounds
from scipy.sparse.csgraph import dijkstra
def main():
 inv=read(HERE/'exact_q64_inventory.csv');diag=read(HERE/'semialgebraic/fresh_failure_property_diagnosis.csv');rows=[];paths=[];notes=[]
 for pocket in diag:
  if pocket['B63']!='False' or pocket['inside_prior_positive_hull']!='True':continue
  sid=pocket['state_id'];rr=[r for r in inv if r['state_id']==sid];root=next(r for r in rr if r['eta_key']==pocket['eta_key']);c=(eta(root)-AFF)/SCALE
  for axis in range(3):
   path=f'{sid}_pocket_axis{axis}';paths.append(dict(state_id=sid,path_id=path,kind='conditional',axis=axis,kernel='',start=root['eta_key'],end=''))
   rows.append(dict(state_id=sid,eta=eta(root).tolist(),phase='pocket_round3',kind='conditional',path_id=path,coordinate=0.))
   for sign in (-1,1):
    d=np.eye(3)[axis]*sign;_,lim=line_bounds(c,d);corridor=f'{sid}_pocket_ray{axis}_{sign}'
    paths.append(dict(state_id=sid,path_id=corridor,kind='failure_corridor',axis=axis,kernel='',start=root['eta_key'],end=key(AFF+SCALE*(c+lim*d))))
    for a in (.33,.67,1.):
     v=(AFF+SCALE*(c+a*lim*d)).tolist()
     for pid,kind,coordinate in ((path,'conditional',sign*a*lim),(corridor,'failure_corridor',a)):
      rows.append(dict(state_id=sid,eta=v,phase='pocket_round3',kind=kind,path_id=pid,coordinate=coordinate))
  negatives=[r for r in rr if r['B63']=='False'];N=np.array([(eta(r)-AFF)/SCALE for r in negatives]);rootidx=next(i for i,r in enumerate(negatives) if r['eta_key']==root['eta_key']);dist=cdist(N,N)
  clearance=-(N@HS[:,:3].T+HS[:,3]).max(axis=1);goals=np.flatnonzero(clearance<=.025+1e-9);route=None
  for scale in (.35,.50):
   graph=np.where((dist<=scale)&(dist>0),dist,0);ds,pred=dijkstra(graph,directed=False,indices=rootidx,return_predecessors=True)
   reachable=[i for i in goals if np.isfinite(ds[i])]
   if not reachable:continue
   target=min(reachable,key=lambda i:(ds[i],negatives[i]['eta_key']));route=[int(target)]
   while route[-1]!=rootidx:route.append(int(pred[route[-1]]))
   route.reverse();break
  notes.append(dict(state_id=sid,geometric_failure_route=route,search_scales=[.35,.50],route_edges_unvalidated=True))
  if route is None:continue
  waypoints=[eta(negatives[i]) for i in route];end=N[route[-1]];face=int(np.argmin(-(HS[:,:3]@end+HS[:,3])));d=HS[face,:3];_,lim=line_bounds(end,d)
  if lim>1e-9:waypoints.append(AFF+SCALE*(end+lim*d))
  path=f'{sid}_failure_graph_detour';paths.append(dict(state_id=sid,path_id=path,kind='failure_corridor',axis='',kernel='',start=root['eta_key'],end=key(waypoints[-1])))
  for j,(a,b) in enumerate(zip(waypoints,waypoints[1:])):
   for alpha in (.25,.5,.75,1.):
    v=b if alpha==1 else (1-alpha)*a+alpha*b
    rows.append(dict(state_id=sid,eta=v.tolist(),phase='pocket_round3',kind='failure_corridor',path_id=path,coordinate=j+alpha))
 write('targeted_probe_rounds/pocket_round3/paths.csv',paths);plan('pocket_round3',rows);dump('targeted_probe_rounds/pocket_round3/route_diagnosis.json',notes)
if __name__=='__main__':main()
