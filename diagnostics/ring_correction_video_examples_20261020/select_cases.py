from pathlib import Path
import json
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];P=Path(__file__).resolve().parent
selected=[];stats={}
for dn,types in [('crosstype1c_two_eta_schedule_v1',['Flow','gaussian','diffusion']),('crosstype3_unseen_controller_schedule_v1',['transformer_bc'])]:
 d=ROOT/'diagnostics'/dn;index=json.loads((d/'NEW_EVAL_METHOD_UIDS.json').read_text());counts=Counter()
 for kind in types:
  candidates=[]
  for r in index:
   if r['type']!=kind:continue
   for q in (0,1):
    uid=r['method_uids']['self_schedule'][q];f=d/'new_evaluation/raw'/f'{uid}.json'
    if not f.exists():continue
    success=json.loads(f.read_text())
    if not success['success'] or success['numerical_failure'] or success['collision']:continue
    for method in ['all_safety_baseline','nominal_only']:
     if method not in r['method_uids']:continue
     bid=r['method_uids'][method][q];bf=d/'new_evaluation/raw'/f'{bid}.json'
     if not bf.exists():continue
     failure=json.loads(bf.read_text());counts[(kind,method,failure['outcome'])]+=1
     if failure['outcome']=='timeout' or failure['numerical_failure'] or failure['success']:continue
     if not failure['collision']:continue
     candidates.append(dict(type=kind,controller=r['controller'],state_uid=r['state_uid'],seed=q,baseline_method=method,baseline_record=str(bf),correction_record=str(f),failure_steps=failure['episode_length'],success_steps=success['episode_length']))
  if candidates:
   # Prefer protected baseline when it exists; otherwise explicit raw-controller collision.
   candidates.sort(key=lambda x:(x['baseline_method']!='all_safety_baseline',-x['failure_steps'],x['controller'],x['state_uid'],x['seed']))
   selected.append(candidates[0])
 stats[dn]={' | '.join(k):v for k,v in counts.items()}
(P/'CASE_INVENTORY.json').write_text(json.dumps(dict(selected=selected,eligible_outcomes=stats),indent=2)+'\n');print(json.dumps(dict(selected=selected,eligible_outcomes=stats),indent=2))
