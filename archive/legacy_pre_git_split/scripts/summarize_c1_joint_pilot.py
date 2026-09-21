"""Report paired pilot outcomes; no checkpoint or scorer selection."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/c1_jax_training_audit/pilot'
rows={name:json.loads((OUT/f'{name}_execution.json').read_text()) for name in ['baseline','P','full']}
assert all(len(r)==32 for r in rows.values())
summary={}
for name,rr in rows.items():
    summary[name]=dict(success=sum(r['success'] for r in rr),deadlock=sum(not r['success'] and r['deadlock'] for r in rr),
        timeout=sum(r['timeout'] for r in rr),controller_errors=sum(r['controller_error'] is not None for r in rr),
        mean_stagnation=float(np.mean([r['stagnation'] for r in rr])),total_stagnation=float(sum(r['stagnation'] for r in rr)),
        by_seed={str(seed):dict(success=sum(r['success'] for r in rr if r['seed']==seed),mean_stagnation=float(np.mean([r['stagnation'] for r in rr if r['seed']==seed]))) for seed in [20260916,20261001]})
    if name!='baseline':
        logs=json.loads((OUT/f'{name}_history.json').read_text());summary[name]['accepted_updates']=sum(r['accepted'] for r in logs)
        summary[name]['backtracks']=sum(r['backtracks'] for r in logs)
        summary[name]['final_training_terms_mean']=np.mean(json.loads((OUT/f'{name}_summary.json').read_text())['final_training_terms'],axis=0).tolist()
all_rows=sum(rows.values(),[])
safety=dict(counts={k:sum(r['safety'][k] for r in all_rows) for k in ['steps','agent_collision_steps','wall_collision_steps','outside_endpoints','cbf_violations','speed_violations']},
    minima={k:min(r['safety'][k] for r in all_rows) for k in ['min_center_separation','min_agent_surface_clearance','min_wall_surface_clearance','min_pair_cbf_residual','min_wall_cbf_residual']})
pairs=[]
for a,b in zip(rows['P'],rows['full']):
    assert (a['rid'],a['seed'])==(b['rid'],b['seed'])
    pairs.append(dict(rid=a['rid'],seed=a['seed'],P_success=a['success'],full_success=b['success'],stagnation_change=b['stagnation']-a['stagnation']))
result=dict(summary=summary,safety=safety,paired_cases=pairs,
    full_rescues_P=sum(not r['P_success'] and r['full_success'] for r in pairs),
    full_loses_P=sum(r['P_success'] and not r['full_success'] for r in pairs))
(OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='paired_cases'},indent=2))
