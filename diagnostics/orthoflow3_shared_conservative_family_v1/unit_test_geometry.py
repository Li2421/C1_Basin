#!/usr/bin/env python3
import csv,json,glob
import numpy as np
from scipy.stats import qmc
from shared_families import contains,violation,domain_contains
lo=np.array([-7/6,-.5,-.5]);hi=np.array([.5,.5,.5]);X=lo+(hi-lo)*qmc.Sobol(3,scramble=False).random_base2(15);X=X[domain_contains(X)][:4096]
rows=[]
for p in glob.glob('*/fitted_parameters.csv')+glob.glob('synthesized_families/*/fitted_parameters.csv'):
 for r in csv.DictReader(open(p)):
  if not r.get('model') or r['model']=='null':continue
  m=json.loads(r['model']);full=contains(m,X,False);ret=contains(m,X,True);v=violation(m,X,True);zero=v<=1e-9
  anchor=np.array(m['anchor'])[None]
  rows.append(dict(file=p,state_id=r['state_id'],family=m['family'],retained_subset_violations=int(np.sum(ret&~full)),zero_violation_disagreements=int(np.sum(zero!=ret)),anchor_retained=bool(contains(m,anchor,True)[0]),finite_parameters=bool(np.isfinite(np.array([x for x in v])).all())))
assert rows and all(x['retained_subset_violations']==0 and x['zero_violation_disagreements']==0 and x['anchor_retained'] and x['finite_parameters'] for x in rows),rows
with open('geometry_unit_tests.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
open('geometry_unit_tests.json','w').write(json.dumps(dict(models=len(rows),points_per_model=len(X),all_pass=True,tests=['retained subset full','V=0 iff retained membership','anchor retained','finite batched evaluation']),indent=2)+'\n')
print(json.dumps({'models':len(rows),'all_pass':True}))
