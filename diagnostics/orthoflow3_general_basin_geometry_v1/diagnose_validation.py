#!/usr/bin/env python3
from audit import *
from scipy.spatial import ConvexHull
def main():
 name=sys.argv[1];inv=read(HERE/'exact_q64_inventory.csv');val=read(HERE/f'targeted_probe_rounds/{name}/results.csv');out=[]
 for sid in sorted({r['state_id'] for r in val}):
  prior=[r for r in inv if r['state_id']==sid and name not in r['phases']];P=np.array([(eta(r)-AFF)/SCALE for r in prior if r['B63']=='True']);N=np.array([(eta(r)-AFF)/SCALE for r in prior if r['B63']=='False']);hull=ConvexHull(P)
  for r in val:
   if r['state_id']!=sid:continue
   z=(np.array(json.loads(r['eta']))-AFF)/SCALE
   out.append(dict(state_id=sid,eta_key=r['eta_key'],B63=r['B63'],Q64=r['Q64'],inside_prior_positive_hull=bool(np.max(hull.equations[:,:3]@z+hull.equations[:,3])<=1e-9),nearest_positive=float(np.linalg.norm(P-z,axis=1).min()),nearest_negative=float(np.linalg.norm(N-z,axis=1).min()),domain_clearance=float(-np.max(HS[:,:3]@z+HS[:,3]))))
 write(f'targeted_probe_rounds/{name}/failure_diagnosis.csv',out)
 result={}
 for label in ('True','False'):
  rr=[r for r in out if r['B63']==label];result[label]=dict(n=len(rr),inside_prior_positive_hull=sum(r['inside_prior_positive_hull'] for r in rr),median_positive_distance=float(np.median([r['nearest_positive'] for r in rr])) if rr else None)
 result['per_state_false']={sid:sum(r['B63']=='False' for r in out if r['state_id']==sid) for sid in sorted({r['state_id'] for r in out})}
 dump(f'targeted_probe_rounds/{name}/failure_diagnosis.json',result);print(json.dumps(result))
if __name__=='__main__':main()
