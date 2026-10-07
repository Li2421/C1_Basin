from core import *
from scipy.sparse.csgraph import connected_components

def main():
 rows=read(H/'exact_q64_ep0082.csv');lookup={r['eta_key']:r for r in rows};neg=[r for r in rows if r['B63']=='False'];index={r['eta_key']:i for i,r in enumerate(neg)};P=np.array([norm(r) for r in rows if r['B63']=='True']);adj=np.eye(len(neg),dtype=bool);edges=[]
 def add(a,b,source):
  if a==b or a not in index or b not in index:return
  x=norm(lookup[a]);y=norm(lookup[b]);d=y-x;l=np.linalg.norm(d)
  if l>.050000001 or l<1e-12:return
  t=(P-x)@d/(d@d);r=np.linalg.norm(P-x-t[:,None]*d,axis=1)
  if np.any((t>1e-9)&(t<1-1e-9)&(r<1e-9)):return
  i,j=index[a],index[b];adj[i,j]=adj[j,i]=True;edges.append(dict(a=a,b=b,normalized_distance=l,evidence=source,status='EXACT_NONB63_ENDPOINTS_WITH_MAX005_SAMPLING_NOT_CONTINUUM_PROOF'))
 for d in json.load(open(H/'corridor_definitions.json')):
  if d['status']!='FROZEN':continue
  for a,b in zip(d['ordered_keys'],d['ordered_keys'][1:]):add(a,b,d['corridor_id'])
 lr=read(H/'conditional_slice_results.csv')
 for lid in sorted({r['line_id'] for r in lr}):
  rr=sorted({r['eta_key']:r for r in lr if r['line_id']==lid}.values(),key=lambda x:float(x['t']))
  for a,b in zip(rr,rr[1:]):add(a['eta_key'],b['eta_key'],lid)
 # Previously queried negative chains are also reusable only with exact collinearity,
 # <=0.05 adjacent separation and no known successful interior point.
 for r in read(H/'cached_conditional_lines.csv'):
  keys=json.loads(r['eta_keys'])
  for a,b in zip(keys,keys[1:]):add(a,b,r['line_id'])
 nc,lab=connected_components(adj,directed=False)
 ext=[i for i,r in enumerate(neg) if abs(np.max(norm(r)@HS[:,:3].T+HS[:,3]))<=1e-9]
 exterior=set(lab[ext]);reach={r['eta_key']:int(lab[i]) in exterior for i,r in enumerate(neg)}
 original=read(H/'internal_failure_candidates.csv');regions=[]
 for reg in json.load(open(H/'region_definitions.json'))['regions']:
  cand=[r for r in original if int(r['region_id'])==reg['region_id']];k=reg['representative_key'];regions.append(dict(region_id=reg['region_id'],original_internal_candidates=len(cand),representative_has_sampled_exterior_path=reach.get(k,False),original_candidates_with_sampled_exterior_path=sum(reach.get(r['eta_key'],False) for r in cand),unresolved_original_candidates=sum(not reach.get(r['eta_key'],False) for r in cand),continuous_topology_certified=False))
 write('sampled_failure_links.csv',edges,['a','b','normalized_distance','evidence','status']);write('region_connectivity_summary.csv',regions)
 old=[]
 for r in read(D/'orthoflow3_t0_basin_shape_v1/targeted_probe_manifest.csv'):
  if r['state_id']!=SID or r['category']!='SUCCESS_SUCCESS_INTERPOLATION':continue
  k=r.get('eta_key_float64',r.get('eta_key'));o=lookup.get(k)
  if o is None:continue
  old.append(dict(eta_key=k,Q64=o['Q64'],B63=o['B63'],endpointA_key=r['endpointA_key'],endpointB_key=r['endpointB_key'],failure_sampled_exterior_path=reach.get(k,False) if o['B63']=='False' else '',interpretation='BOUNDARY_CONNECTED_FAILURE_SUPPORT' if reach.get(k,False) else 'FAILURE_ROUTE_UNRESOLVED' if o['B63']=='False' else 'B63_INTERPOLANT'))
 write('historical_interpolation_explanation.csv',old)
 failures=[r for r in old if r['B63']=='False']
 dump('connectivity_audit_summary.json',dict(original_internal_candidates=len(original),working_regions=len(regions),representatives_with_sampled_exterior_paths=sum(r['representative_has_sampled_exterior_path'] for r in regions),original_internal_points_with_paths=sum(r['original_candidates_with_sampled_exterior_path'] for r in regions),old_interpolation_probes=len(old),old_interpolation_failures=len(failures),old_interpolation_failures_with_paths=sum(r['failure_sampled_exterior_path'] for r in failures),sampled_negative_graph_components=nc,edges=len(edges),boundary_observed_negative_vertices=len(ext),continuous_certificate=False))
 print(json.dumps(json.load(open(H/'connectivity_audit_summary.json'))))
if __name__=='__main__':main()
