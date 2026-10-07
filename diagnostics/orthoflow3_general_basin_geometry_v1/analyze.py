#!/usr/bin/env python3
from audit import *
from scipy.sparse.csgraph import connected_components,minimum_spanning_tree
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
def stats(x):
 x=np.asarray(x);return {'min':float(x.min()),'median':float(np.median(x)),'max':float(x.max())} if x.size else {}
def cached_segment_links(P,N):
 """Observed collinear success chains only; no kNN edge certification.

 Require >=3 observed interior B63 points, max unobserved fraction .30,
 and no observed collinear non-B63. Numerical collinearity tolerance1e-9.
 This supplies finite sampled links, NOT continuous segment safety.
 """
 found={}
 for a in range(len(P)):
  for b in range(a):
   d=P[a]-P[b];norm=float(d@d)
   if norm<1e-16:continue
   t=(P-P[b])@d/norm;res=np.linalg.norm(P-P[b]-t[:,None]*d,axis=1)
   idx=np.flatnonzero((t>=-1e-9)&(t<=1+1e-9)&(res<=1e-9))
   if len(idx)<5 or np.max(np.diff(np.sort(t[idx])))>.30+1e-9:continue
   if len(N):
    tn=(N-P[b])@d/norm;rn=np.linalg.norm(N-P[b]-tn[:,None]*d,axis=1)
    if np.any((tn>=-1e-9)&(tn<=1+1e-9)&(rn<=1e-9)):continue
   found.setdefault(tuple(idx),dict(a=a,b=b,indexes=idx.tolist(),max_sample_gap=float(np.max(np.diff(np.sort(t[idx]))))))
 return list(found.values())
def main():
 inv=[r for r in read(HERE/'exact_q64_inventory.csv') if inside((eta(r)-AFF)/SCALE)[0]];lookup={(r['state_id'],r['eta_key']):r for r in inv};succ=[];fail=[];holes=[];sig=[];stars=[];lines=[];paths=[];cached_links=[];transitions=[]
 # Historical success-success path probes provide direct finite segment evidence.
 hist=read(D/'orthoflow3_t0_basin_shape_v1/targeted_probe_manifest.csv');links=defaultdict(list)
 for r in hist:
  if r['category']=='SUCCESS_SUCCESS_INTERPOLATION':links[(r['state_id'],r['endpointA_key'],r['endpointB_key'])].append(r['eta_key_float64'])
 for p in (HERE/'targeted_probe_rounds').glob('*/paths.csv'):
  man=read(p.parent/'manifest.csv')
  for path in read(p):
   sid=path['state_id'];rr=[r for r in man if r.get('path_id')==path['path_id']];labels=[lookup.get((sid,r['eta_key'])) for r in rr]
   complete=bool(rr) and all(r is not None for r in labels)
   b=sum(r is not None and r['B63']=='True' for r in labels)
   paths.append(dict(state_id=sid,path_id=path['path_id'],kind=path['kind'],probes=len(rr),exact_Q64=sum(r is not None for r in labels),B63=b,complete=complete,
     finite_path_all_success=complete and b==len(rr),no_continuum_certificate=True))
   if path['kind']=='star':
    stars.append(dict(state_id=sid,kernel=path['kernel'],path_id=path['path_id'],B63=b,tested=sum(r is not None for r in labels),all_three_B63=complete and b==len(rr)))
    if complete and b==len(rr):links[(sid,path['start'],path['end'])]+=[r['eta_key'] for r in rr]
   elif path['kind']=='success_detour':
    for segment,(a,c) in enumerate(((path['start'],path['waypoint']),(path['waypoint'],path['end']))):
     selected=[r['eta_key'] for r in rr if int(r['segment'])==segment]
     if len(selected)==3 and all(lookup.get((sid,k),{}).get('B63')=='True' for k in selected):links[(sid,a,c)]+=selected
   elif path['kind']=='conditional':
    ordered=sorted(zip(rr,labels),key=lambda t:float(t[0]['coordinate']));seq=[r['B63']=='True' for _,r in ordered if r is not None];runs=sum(v and (i==0 or not seq[i-1]) for i,v in enumerate(seq))
    lines.append(dict(state_id=sid,axis=int(path['axis'])+1,path_id=path['path_id'],tested=len(seq),complete=complete,labels=''.join('S' if v else 'F' for v in seq),sampled_success_runs=runs,
     interpretation='SINGLE_SAMPLED_INTERVAL' if runs==1 else 'NO_OBSERVED_SUCCESS' if runs==0 else 'MULTIPLE_SAMPLED_INTERVALS',unobserved_gaps='UNKNOWN'))
   elif path['kind']=='failure_corridor':
    holes.append(dict(state_id=sid,path_id=path['path_id'],tested=len(labels),complete=complete,non_B63=len(labels)-b if complete else '',
      classification='SAMPLED_FAILURE_CORRIDOR_TO_DOMAIN_BOUNDARY' if complete and b==0 else 'CANDIDATE_POCKET_CORRIDOR_INTERRUPTED' if complete else 'UNRESOLVED',enclosed_hole_confirmed=False))
   elif path['kind']=='success_failure_transition':
    ordered=sorted(zip(rr,labels),key=lambda t:float(t[0]['coordinate']));vals=[lookup.get((sid,path['start']))]+[r for _,r in ordered]+[lookup.get((sid,path['end']))];done=all(r is not None for r in vals)
    seq=''.join('?' if r is None else 'S' if r['B63']=='True' else 'F' for r in vals)
    changes=sum(a!=b for a,b in zip(seq,seq[1:])) if done else ''
    transitions.append(dict(state_id=sid,path_id=path['path_id'],labels=seq,complete=done,observed_transitions=changes,monotonic_sampled_success_to_failure=done and changes==1,continuum_claim=False))
 # DB uses identical eta semantics but separate conditioning; never merge states.
 dbmanifest=HERE/'targeted_probe_rounds/db_geometry1/manifest.csv'
 if dbmanifest.exists():
  import re
  centers=np.array(json.load(open(HERE/'double_bottleneck_cost_estimate.json'))['candidate_centers'])
  dbr=read(dbmanifest)
  for sid in sorted(set(r['state_id'] for r in dbr)):
   for ci,c in enumerate(centers):
    for axis in range(3):
     coords=[(0.,lookup.get((sid,key(c))))]
     for r in dbr:
      m=re.fullmatch(r'center(\d+)_axis(\d+)_(.+)',r['kind'])
      if r['state_id']==sid and m and int(m[1])==ci and int(m[2])==axis:coords.append((float(m[3]),lookup.get((sid,r['eta_key']))))
     coords.sort(key=lambda x:x[0]);seq=[r['B63']=='True' for _,r in coords if r is not None]
     runs=sum(v and (i==0 or not seq[i-1]) for i,v in enumerate(seq))
     lines.append(dict(state_id=sid,axis=axis+1,path_id=f'{sid}_center{ci}_axis{axis}',tested=len(seq),complete=len(seq)==len(coords),labels=''.join('S' if v else 'F' for v in seq),sampled_success_runs=runs,
       interpretation='SINGLE_SAMPLED_INTERVAL' if runs==1 else 'NO_OBSERVED_SUCCESS' if runs==0 else 'MULTIPLE_SAMPLED_INTERVALS',unobserved_gaps='UNKNOWN'))
   for i in range(len(centers)):
    for j in range(i):
     labels=[lookup.get((sid,key(v))) for v in (centers[i],(centers[i]+centers[j])/2,centers[j])]
     complete=all(r is not None for r in labels);b=sum(r is not None and r['B63']=='True' for r in labels)
     paths.append(dict(state_id=sid,path_id=f'{sid}_connector{i}_{j}',kind='DB_CENTER_ENDPOINT_MIDPOINT_ONLY',probes=3,exact_Q64=sum(r is not None for r in labels),B63=b,complete=complete,finite_path_all_success=complete and b==3,no_continuum_certificate=True))
 balls=read(D/'orthoflow3_t0_multiball_basin_learning_v1/component_balls.csv')
 for sid in sorted(set(r['state_id'] for r in inv)):
  rr=[r for r in inv if r['state_id']==sid];pp=[r for r in rr if r['B63']=='True'];nn=[r for r in rr if r['B63']=='False']
  P=np.array([(eta(r)-AFF)/SCALE for r in pp]);N=np.array([(eta(r)-AFF)/SCALE for r in nn]);n=len(P)
  if n<4:continue
  dist=cdist(P,P);keys={r['eta_key']:i for i,r in enumerate(pp)};g=np.eye(n,dtype=bool)
  for link in cached_segment_links(P,N):
   idx=link['indexes'];g[np.ix_(idx,idx)]=True
   cached_links.append(dict(state_id=sid,endpoint_a=pp[link['a']]['eta_key'],endpoint_b=pp[link['b']]['eta_key'],B63_points=len(idx),max_sample_gap=link['max_sample_gap'],point_keys=json.dumps([pp[i]['eta_key'] for i in idx]),continuum_certified=False))
  for (s,a,b),mid in links.items():
   if s!=sid or a not in keys or b not in keys:continue
   if len(mid)<3 or not all((s,k) in lookup and lookup[s,k]['B63']=='True' for k in mid):continue
   indexes=[keys[a],keys[b]]+[keys[k] for k in mid if k in keys];g[np.ix_(indexes,indexes)]=True
  nc,lab=connected_components(g,directed=False);strict=float(np.bincount(lab).max()/n)
  gb=g.copy()
  for ball in balls:
   if ball['state_id']!=sid or not ball['status'].startswith('ACCEPTED'):continue
   c=(np.array([float(ball['c'+str(i)]) for i in (1,2,3)])-AFF)/SCALE;idx=np.flatnonzero(np.linalg.norm(P-c,axis=1)<=float(ball['r_ball'])+1e-10);gb[np.ix_(idx,idx)]=True
  _,blab=connected_components(gb,directed=False);ballfrac=float(np.bincount(blab).max()/n)
  for radius in (.05,.1,.2,.3,.4):
   ng,labels=connected_components(dist<=radius,directed=False);sizes=sorted(np.bincount(labels),reverse=True)
   succ.append(dict(state_id=sid,scale=radius,geometric_components=ng,largest_geometric_fraction=sizes[0]/n,geometric_sizes=json.dumps([int(x) for x in sizes]),
      sampled_link_component_fraction=strict,ball_assisted_fraction=ballfrac,empirical_main_component_dominated=strict>=.9,geometric_edges_validated=False))
  if len(N):
   nd=cdist(N,N);clear=-(N@HS[:,:3].T+HS[:,3]).max(axis=1)
   for radius in (.05,.1,.2,.3):
    ng,labels=connected_components(nd<=radius,directed=False)
    fail.append(dict(state_id=sid,scale=radius,geometric_failure_components=ng,points_near_domain_boundary=int((clear<=.025).sum()),connectivity_to_exterior='NOT_CERTIFIED_BY_DISTANCE_GRAPH'))
  c=P.mean(axis=0);_,sv,V=np.linalg.svd(P-c,full_matrices=False);Z=(P-c)@V.T;span=np.ptp(Z,axis=0)
  signature=dict(state_id=sid,scenario=rr[0]['scenario'],B63=n,non_B63=len(N),sampled_link_component_fraction=strict,ball_assisted_fraction=ballfrac,
    diameter=float(pdist(P).max()),tangent1_span=float(span[0]),tangent2_span=float(span[1]),normal_span=float(span[2]),tangent1_normal_ratio=float(span[0]/max(span[2],1e-12)),
    pca_variance_ratios=json.dumps((sv**2/sum(sv**2)).tolist()),pca_frame=json.dumps(V.tolist()),centroid=json.dumps(c.tolist()),confirmed_enclosed_holes=0,hole_absence_proven=False,
    nearest_opposite_label_distance=float(cdist(P,N).min()) if len(N) else '')
  sig.append(signature)
 write('success_component_analysis.csv',succ);write('failure_component_analysis.csv',fail)
 write('cached_collinear_success_links.csv',cached_links,['state_id','endpoint_a','endpoint_b','B63_points','max_sample_gap','point_keys','continuum_certified'])
 write('success_failure_transition.csv',transitions,['state_id','path_id','labels','complete','observed_transitions','monotonic_sampled_success_to_failure','continuum_claim'])
 write('hole_vs_notch.csv',holes,['state_id','path_id','tested','complete','non_B63','classification','enclosed_hole_confirmed'])
 write('connectivity_paths.csv',paths,['state_id','path_id','kind','probes','exact_Q64','B63','complete','finite_path_all_success','no_continuum_certificate'])
 write('star_convexity.csv',stars,['state_id','kernel','path_id','B63','tested','all_three_B63'])
 write('conditional_interval_structure.csv',lines,['state_id','axis','path_id','tested','complete','labels','sampled_success_runs','interpretation','unobserved_gaps'])
 write('per_state_geometry_signature.csv',sig)
 # Aligned positive-cloud Chamfer is descriptive, ascertainment-biased, not occupancy accuracy.
 aligned=[]
 for a in sig:
  if a['B63']<30:continue
  pa=np.array([(eta(r)-AFF)/SCALE for r in inv if r['state_id']==a['state_id'] and r['B63']=='True']);ca=np.array(json.loads(a['centroid']));va=np.array(json.loads(a['pca_frame']));xa=(pa-ca)@va.T;xa/=np.maximum(np.ptp(xa,axis=0),1e-10)
  for b in sig:
   if b['B63']<30 or b['state_id']<=a['state_id']:continue
   pb=np.array([(eta(r)-AFF)/SCALE for r in inv if r['state_id']==b['state_id'] and r['B63']=='True']);cb=np.array(json.loads(b['centroid']));vb=np.array(json.loads(b['pca_frame']));xb=(pb-cb)@vb.T;xb/=np.maximum(np.ptp(xb,axis=0),1e-10)
   import itertools
   vals=[]
   for sign in itertools.product((-1,1),repeat=3):
    dd=cdist(xa,xb*np.array(sign));vals.append(.5*(dd.min(axis=0).mean()+dd.min(axis=1).mean()))
   aligned.append(dict(state_a=a['state_id'],state_b=b['state_id'],raw_symmetric_chamfer=float(.5*(cdist(pa,pb).min(axis=0).mean()+cdist(pa,pb).min(axis=1).mean())),scaled_aligned_chamfer=float(min(vals)),metric_units='raw normalized eta vs dimensionless aligned extent',occupancy_claim=False))
 write('cross_state_alignment.csv',aligned)
 if (HERE/'common_eta_panel.csv').exists():
  matrix=[];summary=[]
  for p in read(HERE/'common_eta_panel.csv'):
   records=[]
   for sid in sorted(STATES):
    r=lookup.get((sid,key(eta(p))));records.append(r)
    matrix.append(dict(state_id=sid,mode_id=p['mode_id'],Q64=r['Q64'] if r else '',B63=r['B63'] if r else '',deadlock=r['deadlock'] if r else '',timeout=r['timeout'] if r else '',exact_available=r is not None))
   completed=[r for r in records if r];summary.append(dict(mode_id=p['mode_id'],tested=len(completed),B63=sum(r['B63']=='True' for r in completed),Q_ge_90=sum(float(r['Q64'])>=.9 for r in completed),mean_Q64=float(np.mean([float(r['Q64']) for r in completed])) if completed else None))
  write('cross_state_eta_matrix.csv',matrix);dump('common_core_summary.json',summary)
  cross=[]
  for sid in sorted({r['state_id'] for r in inv if r['scenario']=='DoubleBottleneck_4A'}):
   for p in read(HERE/'common_eta_panel.csv'):
    r=lookup.get((sid,key(eta(p))))
    cross.append(dict(state_id=sid,scenario='DoubleBottleneck_4A',mode_id=p['mode_id'],Q64=r['Q64'] if r else '',B63=r['B63'] if r else '',exact_available=r is not None))
  write('cross_scenario_common_eta_matrix.csv',cross,['state_id','scenario','mode_id','Q64','B63','exact_available'])
  (HERE/'common_core_analysis.md').write_text('# Empirical common-core panel\n\n'+json.dumps(summary,indent=2)+'\n\nA finite panel is not a positive-volume common-core certificate. No overlap threshold is treated as degeneracy. Mode selection used historical robust targets, so performance estimates describe these40 states, not new population generalization.\n')
 print('Geometry analysis:',len(sig),'states; finite tested paths',len(paths))
if __name__=='__main__':main()
