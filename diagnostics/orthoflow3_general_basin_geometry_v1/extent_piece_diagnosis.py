#!/usr/bin/env python3
"""Offline minimum facet cover diagnostic; not a fitted membership lookup."""
from audit import *
from scipy.spatial import ConvexHull
from scipy.optimize import milp,Bounds,LinearConstraint
def main():
 inv=read(HERE/'exact_q64_inventory.csv');development=json.load(open(HERE/'development_validation_batches.json'))['batches'];out=[]
 for sid in sorted({r['state_id'] for r in read(HERE/'family_metrics_polyhedral_r4.csv')}):
  rr=[r for r in inv if r['state_id']==sid];positive=[r for r in rr if r['B63']=='True' and (('retained_validation' not in r['phases'] and r['geometry_role']=='fit') or any(b in r['phases'].split(';') for b in development))]
  P=np.array([(eta(r)-AFF)/SCALE for r in positive]);N=np.array([(eta(r)-AFF)/SCALE for r in rr if r['B63']=='False']);facets=np.unique(ConvexHull(P).equations,axis=0);reject=N@facets[:,:3].T+facets[:,3]>1e-9;external=reject.any(axis=1);cover=reject[external].astype(float)
  if not len(cover):out.append(dict(state_id=sid,exterior_negatives=0,interior_negatives=len(N),minimum_facets=0,optimal=True,greedy3_coverage=1.,facets=len(facets)));continue
  sol=milp(np.ones(len(facets)),integrality=np.ones(len(facets)),bounds=Bounds(0,1),constraints=LinearConstraint(cover,1,np.inf),options={'time_limit':10,'mip_rel_gap':0})
  remaining=np.ones(len(cover),bool);selected=[]
  for k in range(3):
   gain=cover[remaining].sum(axis=0);i=int(np.argmax(gain))
   if gain[i]==0:break
   selected.append(i);remaining &= cover[:,i]==0
  out.append(dict(state_id=sid,exterior_negatives=int(external.sum()),interior_negatives=int((~external).sum()),minimum_facets=int(round(sol.fun)) if sol.fun is not None else '',optimal=sol.status==0,greedy3_coverage=float(1-remaining.mean()),facets=len(facets)))
 prefix=sys.argv[1] if len(sys.argv)>1 else 'extent_piece_diagnosis'
 write(prefix+'.csv',out)
 good=[r['minimum_facets'] for r in out if r['optimal']];result=dict(states=len(out),proven_optima=len(good),median_minimum_support_facets=float(np.median(good)) if good else None,max_minimum_support_facets=max(good) if good else None,median_greedy3_exterior_negative_coverage=float(np.median([r['greedy3_coverage'] for r in out])),
  limitation='Covering exterior negatives while retaining ALL fit positives is a diagnostic, not a lower bound for70%recall. Hull used only in offline fitting, never deployed membership.')
 dump(prefix+'.json',result);print(json.dumps(result))
if __name__=='__main__':main()
