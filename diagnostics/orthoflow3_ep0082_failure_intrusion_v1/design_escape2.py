from core import *
from scipy.sparse.csgraph import dijkstra

def main():
 name='escape3' if '--third-round' in sys.argv else 'escape2'
 rows=read(H/'exact_q64_ep0082.csv');known={r['eta_key']:r for r in rows};nr=[r for r in rows if r['B63']=='False'];N=np.array([norm(r) for r in nr]);P=np.array([norm(r) for r in rows if r['B63']=='True']);idx={r['eta_key']:i for i,r in enumerate(nr)}
 targets=[]
 for r in json.load(open(H/'region_definitions.json'))['regions']:
  if r['region_id'] in (0,1):targets.append(dict(key=r['representative_key'],region_id=r['region_id'],purpose='unresolved_region_representative'))
 history=read(D/'orthoflow3_t0_basin_shape_v1/targeted_probe_manifest.csv')
 for r in history:
  k=r.get('eta_key_float64',r.get('eta_key'))
  if r['state_id']==SID and r['category']=='SUCCESS_SUCCESS_INTERPOLATION' and k in idx and not any(t['key']==k for t in targets):
   targets.append(dict(key=k,region_id=0,purpose='earlier_success_success_failure'))
 if '--third-round' in sys.argv:
  linked_regions={int(r['region_id']) for r in read(H/'region_connectivity_summary.csv') if r['representative_has_sampled_exterior_path']=='True'}
  linked_old={r['eta_key'] for r in read(H/'historical_interpolation_explanation.csv') if r['failure_sampled_exterior_path']=='True'}
  targets=[t for t in targets if not (t['purpose']=='unresolved_region_representative' and t['region_id'] in linked_regions) and t['key'] not in linked_old]
 count=len(N);cost=np.full((count+1,count+1),np.inf);segments={}
 def segment(a,b,ea,eb):
  d=b-a;l=float(np.linalg.norm(d))
  if l<1e-10:return [ea],l
  tp=((P-a)@d)/(d@d);res=np.linalg.norm(P-a-tp[:,None]*d,axis=1)
  if np.any((tp>=-1e-9)&(tp<=1+1e-9)&(res<1e-9)):return None,l
  # Include every exact collinear negative already observed on this segment.
  tn=((N-a)@d)/(d@d);rn=np.linalg.norm(N-a-tn[:,None]*d,axis=1);inter=[(float(tn[j]),eta(nr[j])) for j in range(count) if 1e-9<tn[j]<1-1e-9 and rn[j]<1e-9]
  vals=[(0.,ea)]+sorted(inter,key=lambda x:x[0])+[(1.,eb)];points=[ea]
  for (ta,va),(tb,vb) in zip(vals,vals[1:]):
   pieces=max(1,int(np.ceil((tb-ta)*l/.05)))
   # At least one interior sample if no archived interior point is available.
   if len(vals)==2:pieces=max(2,pieces)
   points += [AFF+SCALE*(a+(ta+(tb-ta)*j/pieces)*d) for j in range(1,pieces)]+[vb]
  if any(known.get(key(e),{}).get('B63')=='True' for e in points):return None,l
  return points,l
 for i in range(count):
  for j in range(i+1,count):
   if np.linalg.norm(N[i]-N[j])>.30:continue
   pp,l=segment(N[i],N[j],eta(nr[i]),eta(nr[j]))
   if pp is None:continue
   w=sum(key(e) not in known for e in pp)+.001*l+1e-9
   cost[i,j]=cost[j,i]=w;segments[i,j]=pp;segments[j,i]=list(reversed(pp))
  options=[]
  for face in HS:
   v=face[:3];b=N[i]-(N[i]@v+face[3])*v/(v@v)
   if not inside(b)[0]:continue
   pp,l=segment(N[i],b,eta(nr[i]),AFF+SCALE*b)
   if pp is not None:options.append((sum(key(e) not in known for e in pp)+.001*l+1e-9,l,pp))
  if options:
   w,l,pp=min(options,key=lambda x:x[:2]);cost[i,count]=w;segments[i,count]=pp
 np.fill_diagonal(cost,0);requests=[];defs=[];man=[]
 for it,t in enumerate(targets):
  a=idx[t['key']];dist,pred=dijkstra(cost,directed=True,indices=a,return_predecessors=True)
  if not np.isfinite(dist[count]):defs.append(dict(target=t,status='NO_GRAPH_ROUTE'));continue
  nodes=[count]
  while nodes[-1]!=a:nodes.append(int(pred[nodes[-1]]))
  nodes=nodes[::-1];cid=f'{name}_{it}_region{t["region_id"]}';kk=[]
  for edge,(u,v) in enumerate(zip(nodes,nodes[1:])):
   pp=segments[u,v]
   for j,e in enumerate(pp):
    k=key(e);kk.append(k);man.append(dict(corridor_id=cid,region_id=t['region_id'],edge=edge,position=j,eta_key=k,eta=json.dumps(e.tolist()),outer_endpoint=v==count and j==len(pp)-1,round=name))
    requests.append(dict(eta=e,role=f'corridor:{cid}:edge={edge}:position={j}'))
  defs.append(dict(corridor_id=cid,region_id=t['region_id'],purpose=t['purpose'],start_key=t['key'],status='FROZEN',cached_waypoints=[nr[n]['eta_key'] for n in nodes if n<count],ordered_keys=kk,endpoint_on_domain=True,estimated_new_cost=float(dist[count]),maximum_segment_sample_gap=.05))
 new={key(r['eta']) for r in requests if key(r['eta']) not in known};oldnew={r['eta_key'] for r in read(H/'adaptive_probe_manifest.csv') if r['cached_Q64']=='False'}
 dump(f'{name}_preview.json',dict(targets=targets,corridors=defs,new_eta=len(new),projected_total_new=len(oldnew|new),question='Do remaining interrupted representative escape routes have cache-informed alternatives, and do old B63-B63 interpolation failures join boundary-connected failure intrusions?'))
 print(json.dumps(dict(targets=len(targets),new_eta=len(new),projected_total_new=len(oldnew|new),per_route_cost=[r.get('estimated_new_cost') for r in defs])))
 if '--freeze' not in sys.argv:return
 assert len(oldnew|new)<=150
 write('candidate_failure_corridors.csv',read(H/'candidate_failure_corridors.csv')+man)
 dump('corridor_definitions.json',json.load(open(H/'corridor_definitions.json'))+defs)
 cost=plan(name,requests,'Distinguish interrupted direct escapes from a connected bent intrusion; connect historical interpolation failures to the exterior if supported.')
 if len(oldnew|new)>100:dump(f'rounds/{name}/above100_diagnostic.json',dict(current_planned=len(oldnew|new),question='Remaining old interpolation-failure explanation and failed representative escapes are required to distinguish notches from unprobed pockets.',scope='Targeted cache-informed paths plus already-registered boundary bisections; no grid, new state, or family fitting.',within150=True))
if __name__=='__main__':main()
