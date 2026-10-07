#!/usr/bin/env python3
from audit import *
from families import contains
import argparse
def input_keys(inv,sid,m):
 rr=[r for r in inv if r['state_id']==sid];development=m.get('development_validation_batches',[])
 fit={r['eta_key'] for r in rr if r['B63']=='True' and (('retained_validation' not in r['phases'] and r['geometry_role']=='fit') or any(b in r['phases'].split(';') for b in development))}
 neg={r['eta_key'] for r in rr if r['B63']=='False'}
 held={r['eta_key'] for r in rr if r['B63']=='True' and 'retained_validation' not in r['phases'] and r['geometry_role']=='heldout'}
 return fit,neg,held
def geometry(m):return {k:m[k] for k in ('family','planes','w')}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--old-tag',required=True);ap.add_argument('--new-tag',required=True);ap.add_argument('--batch',required=True);a=ap.parse_args()
 old={r['state_id']:r['model'] for r in json.load(open(HERE/f'family_models_{a.old_tag}.json'))};new={r['state_id']:r['model'] for r in json.load(open(HERE/f'family_models_{a.new_tag}.json'))}
 oldinv=read(HERE/f'fitting_evidence/{a.old_tag}.csv');newinv=read(HERE/f'fitting_evidence/{a.new_tag}.csv');results=read(HERE/f'targeted_probe_rounds/{a.batch}/results.csv');gate=json.load(open(HERE/f'targeted_probe_rounds/{a.batch}/gate.json'));assert gate['complete'] and gate['false_inclusions']==0
 checks=[]
 for sid in sorted({r['state_id'] for r in results}):
  assert geometry(old[sid])==geometry(new[sid]),(sid,'mathematical instance changed')
  assert input_keys(oldinv,sid,old[sid])==input_keys(newinv,sid,new[sid]),(sid,'fit/primary heldout inputs changed')
  fit,neg,held=input_keys(newinv,sid,new[sid]);rr=[r for r in results if r['state_id']==sid]
  assert all(r['eta_key'] not in fit|neg|held and r['B63']=='True' for r in rr)
  assert contains(new[sid],np.array([(np.array(json.loads(r['eta']))-AFF)/SCALE for r in rr]),True).all()
  checks.append(dict(state_id=sid,mathematical_parameters_identical=True,fit_inputs_identical=True,primary_heldout_identical=True,independent_points=len(rr),false_inclusions=0,geometry_sha256=hashlib.sha256(json.dumps(geometry(new[sid]),sort_keys=True).encode()).hexdigest()))
 dump(f'validation_reuse_{a.new_tag}.json',dict(reuse_allowed=True,old_fit=a.old_tag,new_fit=a.new_tag,batch=a.batch,checks=checks,independent_Q64_reused=len(results),new_rollouts=0,normalization='unchanged frozen E_bridge',erosion_delta=.025,conditioning='authoritative exact-Q64 inventory; same state/h/xi0/future namespaces/controller stack'))
 print('Validated exact independent reuse:',len(results),'Q64 points;',len(checks),'unchanged state instances')
if __name__=='__main__':main()
