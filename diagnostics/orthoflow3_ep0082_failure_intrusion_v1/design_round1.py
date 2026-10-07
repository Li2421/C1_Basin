#!/usr/bin/env python3
from core import *
from scipy.spatial import ConvexHull
from scipy.sparse.csgraph import connected_components,dijkstra

def main():
 rows=read(H/'exact_q64_ep0082.csv');Z=np.array([norm(r) for r in rows]);good=np.array([r['B63']=='True' for r in rows]);P=Z[good];N=Z[~good];nr=[r for r in rows if r['B63']=='False'];h=ConvexHull(P);hv=N@h.equations[:,:3].T+h.equations[:,3]
 pd=cdist(P,P);np.fill_diagonal(pd,np.inf);support=2*float(np.quantile(np.sort(pd,axis=1)[:,4],.9));dp=cdist(N,P);dn=cdist(N,N);np.fill_diagonal(dn,np.inf);internal=(hv.max(1)<=1e-9)&(dp.min(1)<=support);ii=np.flatnonzero(internal);I=N[internal]
 c,R=frame();stats=[]
 for scale in [.05,.1,.15,.2,.3]:
  nc,la=connected_components(cdist(I,I)<=scale,directed=False);stats.append(dict(radius=scale,components=nc,sizes=np.bincount(la).tolist()))
 _,lab=connected_components(cdist(I,I)<=.2,directed=False);regions=[];candidates=[]
 for j,n in enumerate(ii):
  candidates.append(dict(state_id=SID,eta_key=nr[n]['eta_key'],region_id=int(lab[j]),eta=json.dumps(eta(nr[n]).tolist()),Q64=nr[n]['Q64'],nearest_B63=float(dp[n].min()),nearest_non_B63=float(dn[n].min()),B63_within_010=int((dp[n]<=.1).sum()),B63_within_020=int((dp[n]<=.2).sum()),domain_clearance=float(-(N[n]@HS[:,:3].T+HS[:,3]).max()),positive_hull_clearance=float(-hv[n].max()),support_limit=support))
 for k in sorted(set(lab)):
  idx=ii[lab==k];s=cdist(N[idx],N[idx]).sum(1);rep=sorted(zip(s,[nr[j]['eta_key'] for j in idx],idx))[0][2];regions.append(dict(region_id=int(k),candidate_count=len(idx),representative_key=nr[rep]['eta_key'],representative_index=int(rep),eta=eta(nr[rep]).tolist()))
 write('internal_failure_candidates.csv',candidates);dump('region_definitions.json',dict(primary_radius=.2,support_limit=support,scale_sensitivity=stats,regions=regions,hull_only_count=int((hv.max(1)<=1e-9).sum()),candidate_count=len(ii)))
 requests=[];lines=[];views=[];line_defs=[]
 for cand in candidates:
  z=(np.array(json.loads(cand['eta']))-AFF)/SCALE;q=(z-c)@R
  for kind,axes,fixed in [('tangential',[0,1],2),('mixed_s1_n',[0,2],1),('mixed_s2_n',[1,2],0)]:
   for tol in [.02,.05]:views.append(dict(candidate_key=cand['eta_key'],kind=kind,axes=axes,fixed_axis=fixed,fixed_coordinate=float(q[fixed]),tolerance=tol,nearby_exact_points=int((abs(((Z-c)@R)[:,fixed]-q[fixed])<=tol).sum()),interpretation='Approximate slice; not an exact conditional line'))
 for region in regions:
  idx=region['representative_index'];a=N[idx];origin=eta(nr[idx]);rid=region['region_id']
  for ax in range(3):
   d=R[:,ax];line=f'region{rid}_e{ax+1}';line_defs.append(dict(line_id=line,region_id=rid,origin=origin.tolist(),direction=d.tolist(),kind='R0_conditional'))
   for t in [-.1,-.05,0.,.05,.1]:
    if t:
     u=np.sign(t)*d;v=HS[:,:3]@u;sel=v>1e-12;lim=float(np.min(-(HS[sel,:3]@a+HS[sel,3])/v[sel]));actual=np.sign(t)*min(abs(t),.95*lim);e=AFF+SCALE*(a+actual*d)
    else:actual=0.;e=origin
    k=key(e);lines.append(dict(line_id=line,region_id=rid,axis=ax+1,t=float(actual),eta_key=k,eta=json.dumps(e.tolist()),round='round1'))
    requests.append(dict(eta=e,role=f'line:{line}:t={actual:.17g}'))
 write('conditional_slice_manifest.csv',lines);dump('line_definitions.json',line_defs);write('cached_slice_views.csv',views)
 # Cached negative graph and minimum-new-probe route to a feasible domain facet.
 known={r['eta_key']:r for r in rows}
 def segment(a,b,ea,eb):
  d=b-a;length=float(np.linalg.norm(d))
  if length<1e-12:return [],0.
  tt=((P-a)@d)/(d@d);perp=np.linalg.norm(P-a-tt[:,None]*d,axis=1)
  if np.any((tt>1e-8)&(tt<1-1e-8)&(perp<1e-9)):return None,length
  pieces=max(2,int(np.ceil(length/.05)));e=[ea]+[AFF+SCALE*(a+d*j/pieces) for j in range(1,pieces)]+[eb]
  if any(known.get(key(x),{}).get('B63')=='True' for x in e):return None,length
  return e,length
 count=len(N);cost=np.full((count+1,count+1),np.inf);segments={};graphs=[]
 for i in range(count):
  for j in range(i+1,count):
   dist=float(np.linalg.norm(N[i]-N[j]))
   if dist>.3:continue
   scales=[v for v in [.05,.1,.15,.2,.25,.3] if dist<=v];graphs.append(dict(a=nr[i]['eta_key'],b=nr[j]['eta_key'],distance=dist,geometric_scales=json.dumps(scales),validated_continuous=False))
   if dist>.25:continue
   pp,length=segment(N[i],N[j],eta(nr[i]),eta(nr[j]))
   if pp is None:continue
   w=sum(key(x) not in known for x in pp)+.001*length+1e-9*(i*count+j);cost[i,j]=cost[j,i]=w;segments[i,j]=pp;segments[j,i]=list(reversed(pp))
  options=[]
  for face in HS:
   b=N[i]-(N[i]@face[:3]+face[3])*face[:3]
   if not inside(b)[0]:continue
   eb=AFF+SCALE*b;pp,length=segment(N[i],b,eta(nr[i]),eb)
   if pp is None:continue
   w=sum(key(x) not in known for x in pp)+.001*length;options.append((w,length,pp,face.tolist()))
  if options:
   best=min(options,key=lambda x:(x[0],x[1]));cost[i,count]=best[0]+1e-8;segments[i,count]=best[2]
 np.fill_diagonal(cost,0);write('failure_connectivity_graph.csv',graphs)
 corridors=[];corridor_defs=[]
 for reg in regions:
  start=reg['representative_index'];dist,pred=dijkstra(cost,directed=True,indices=start,return_predecessors=True)
  if not np.isfinite(dist[count]):corridor_defs.append(dict(region_id=reg['region_id'],status='NO_CACHED_NEGATIVE_ROUTE'));continue
  nodes=[count]
  while nodes[-1]!=start:nodes.append(int(pred[nodes[-1]]))
  nodes=nodes[::-1];cid=f'region{reg["region_id"]}_exit1';allpoints=[]
  for edge,(u,v) in enumerate(zip(nodes,nodes[1:])):
   pp=segments[u,v]
   for j,e in enumerate(pp):
    allpoints.append(key(e));corridors.append(dict(corridor_id=cid,region_id=reg['region_id'],edge=edge,position=j,eta_key=key(e),eta=json.dumps(np.asarray(e).tolist()),outer_endpoint=(v==count and j==len(pp)-1),round='round1'))
    requests.append(dict(eta=e,role=f'corridor:{cid}:edge={edge}:position={j}'))
  corridor_defs.append(dict(corridor_id=cid,region_id=reg['region_id'],status='FROZEN',cached_waypoints=[nr[n]['eta_key'] for n in nodes if n<count],ordered_keys=allpoints,endpoint_on_domain=True,estimated_new_cost=float(dist[count]),maximum_segment_sample_gap=.05))
 write('candidate_failure_corridors.csv',corridors);dump('corridor_definitions.json',corridor_defs)
 cost=plan('round1',requests,'Are four internal-negative working regions connected through sampled failure corridors to E_bridge boundary, and do their six R0 rays show enclosure or repeated transitions?')
 assert cost['new_exact_Q64']<=80,'Reassess design before execution; preserve adaptive headroom'
 (H/'logs').mkdir(exist_ok=True)
 s=json.load(open(H/'working_state.json'));s.update(completed=s['completed']+['internal candidates and scale sensitivity','round1 freeze'],next_action='Submit frozen round1 on2GPU shards after resource check; then exact transition/corridor diagnosis',candidate_regions=len(regions),internal_candidates=len(candidates),round1_cost=cost);dump('working_state.json',s)
 print(json.dumps(dict(internal_candidates=len(candidates),regions=len(regions),round1=cost)))
if __name__=='__main__':main()
