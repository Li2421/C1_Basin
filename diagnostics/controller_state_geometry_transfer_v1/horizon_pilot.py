from pathlib import Path
import json,datetime
import numpy as np
from .pilot import stats
HERE=Path(__file__).resolve().parent

def main():
 src=HERE/'broad_pilot';cases=json.loads((src/'cases.json').read_text());W=[.41,.59];C=['flow_isotropic','flow_anisotropic'];E=[0,2,4,5,8,10,12,15]
 allres={};selected=None
 for H in [400,200]:
  Y=np.empty((2,16,2,8,2))
  for c in cases:
   r=json.loads((src/'raw'/(c['rollout_uid']+'.json')).read_text());assert not r['numerical_failure']
   Y[W.index(c['width']),c['state_index'],C.index(c['controller_name']),E.index(c['eta_index']),c['seed']]=int(r['success'] and r['summary']['first_success_step']<=H)
  res=stats(Y);rng=np.random.default_rng(20261011);bs=[stats(Y[:,rng.integers(0,16,16)]) for _ in range(2000)]
  for k in ['state_fraction','state_eta_energy','oracle_adaptive_gain']:res[k+'_ci95']=np.percentile([b[k] for b in bs],[2.5,97.5]).tolist()
  res['passed']=bool(res['state_fraction']>=.2 and res['state_eta_energy']>=.005 and res['oracle_adaptive_gain']>=.05 and res['state_eta_energy_ci95'][0]>0 and res['oracle_adaptive_gain_ci95'][0]>0)
  res['n_G_S_C_with_success_witness']=int(Y.max((3,4)).sum());res['total_G_S_C']=64;allres[str(H)]=res
  if res['passed'] and selected is None:selected=H
  # H200 is also retained as independently preregistered factor analysis, never preferred over passing400.
 out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'results':allres,'selected_horizon':selected,'passed':selected is not None,'prefix_equivalence_audit_required':True}
 (HERE/'horizon_pilot_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
