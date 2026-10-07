#!/usr/bin/env python3
from audit import *
from scipy.spatial import ConvexHull
def main():
 inv=read(HERE/'exact_q64_inventory.csv');val=read(HERE/'semialgebraic/fresh_retained_validation_r1.csv');out=[]
 for sid in sorted({r['state_id'] for r in val}):
  prior=[r for r in inv if r['state_id']==sid and 'retained_validation' not in r['phases']];P=np.array([(eta(r)-AFF)/SCALE for r in prior if r['B63']=='True']);N=np.array([(eta(r)-AFF)/SCALE for r in prior if r['B63']=='False'])
  hull=ConvexHull(P);C=P.mean(axis=0);A=np.linalg.inv(np.cov(P.T)+1e-5*np.eye(3));maxq=np.max(np.einsum('ni,ij,nj->n',P-C,A,P-C))
  for r in val:
   if r['state_id']!=sid:continue
   z=(np.array(json.loads(r['eta']))-AFF)/SCALE
   out.append(dict(state_id=sid,eta_key=r['eta_key'],B63=r['B63'],Q64=r['Q64'],inside_prior_positive_hull=bool(np.max(hull.equations[:,:3]@z+hull.equations[:,3])<=1e-10),
    prior_hull_max_violation=float(np.max(hull.equations[:,:3]@z+hull.equations[:,3])),inside_positive_enclosing_ellipsoid=bool((z-C)@A@(z-C)<=maxq),nearest_positive=float(np.linalg.norm(P-z,axis=1).min()),nearest_negative=float(np.linalg.norm(N-z,axis=1).min()),
    normalized_domain_clearance=float(-np.max(HS[:,:3]@z+HS[:,3]))))
 write('semialgebraic/fresh_failure_property_diagnosis.csv',out)
 def summarize(rows):return dict(n=len(rows),inside_positive_hull=sum(r['inside_prior_positive_hull'] for r in rows),inside_ellipsoid=sum(r['inside_positive_enclosing_ellipsoid'] for r in rows),median_nearest_positive=float(np.median([r['nearest_positive'] for r in rows])))
 result={'failed':summarize([r for r in out if r['B63']=='False']),'passed':summarize([r for r in out if r['B63']=='True']),
  'per_state':{sid:sum(r['B63']=='False' for r in out if r['state_id']==sid) for sid in sorted({r['state_id'] for r in out})}}
 dump('semialgebraic/fresh_failure_diagnosis.json',result);print(json.dumps(result))
 hs=json.load(open(HERE/'hypothesis_status.json'))
 for h in hs:
  if h.get('family')=='semialgebraic':h.update(current_status='FRESH_RETAINED_VALIDATION_REJECTED',key_failure={'fresh_non_B63':21,'fresh_total':42,'empty_instance':'ep0082'},next_discriminating_test='Diagnose bounded-support extrapolation vs internal excluded regions; current DBgeometry2 remains informative.')
 dump('hypothesis_status.json',hs)
 state=json.load(open(HERE/'working_state.json'));state['candidate_status']['fresh_validation']='REJECTED21/42';state['candidate_status']['cached_gate']='PASS_BUT_PROSPECTIVELY_FALSIFIED';state['next_action']='Complete902; diagnose fresh failure hull/extent relation; design smallest property-discriminating probe. No training and no current accepted family.';state['completed_stages'].append('prospective semialgebraic42Q64 validation: rejected21 false inclusions');dump('working_state.json',state)
if __name__=='__main__':main()
