#!/usr/bin/env python3
from families import *
def main():
 synthetic={'family':'minimal_polyhedral_support_quadratic_exclusion','planes':[[1.,0.,0.,.1]],'w':[1.]+[0.]*9,'caps':[]}
 assert not contains(synthetic,[0,0,0])[0]
 assert contains(synthetic,[-.2,0,0])[0]
 assert contains(synthetic,[-.125,0,0],True)[0]
 assert not contains(synthetic,[-.12,0,0],True)[0]
 tag='minimal_polyhedral_r5_corrected';models=json.load(open(HERE/f'family_models_{tag}.json'));inv=read(HERE/'exact_q64_inventory.csv');rng=np.random.default_rng(713);z=rng.uniform([-7/6,-.5,-.5],[.5,.5,.5],size=(20000,3));checks=0
 for r in models:
  m=r['model'];neg=np.array([(eta(x)-AFF)/SCALE for x in inv if x['state_id']==r['state_id'] and x['B63']=='False']);assert not contains(m,neg).any()
  p=z[contains(m,z,True)];assert contains(m,p).all()
  if len(p):
   d=rng.normal(size=p.shape);d/=np.linalg.norm(d,axis=1)[:,None];assert contains(m,p+.025*d).all();checks+=len(p)
 dump('minimal_family_unit_tests.json',dict(passed=True,synthetic_support_dispatch=True,cached_negative_check=True,erosion_checks=checks,delta=.025))
 # This tag was fitted immediately before the new snapshot guard was introduced.
 write(f'fitting_evidence/{tag}.csv',inv);dump(f'fitting_evidence/{tag}_provenance.json',dict(inventory_sha256=sha(HERE/'exact_q64_inventory.csv'),snapshot=f'fitting_evidence/{tag}.csv',captured_before_any_new_rollout=True))
 print('Minimal-family support, hard-negative and erosion tests passed',checks)
if __name__=='__main__':main()
