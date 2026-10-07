#!/usr/bin/env python3
"""Freeze three DB-panel-common eta and transfer them to eight Toy states."""
import csv,json,hashlib
from pathlib import Path
import numpy as np
from prepare import HERE,D,AFF,SCALE,HS,SHA,sha,key,eta,read,write,dump,TOY
def main():
 if (HERE/'fresh_common_core_manifest.csv').exists():raise FileExistsError('common-core panel already frozen')
 inv=read(HERE/'exact_q64_inventory.csv');by={}
 for r in inv:by.setdefault((r['state_id'],r['eta_key']),r)
 audit=read(HERE/'common_core_audit.csv');cand=[]
 for r in audit:
  if int(r['db_tested'])!=4 or int(r['db_B63'])!=4:continue
  vals=[float(by[s,r['eta_key']]['Q64']) for s in ('DB_T0_0','DB_T0_1','DB_T0_2','DB_T0_3')]
  cand.append((min(vals),float(np.mean(vals)),r))
 assert cand
 cand.sort(key=lambda x:(x[0],x[1],x[2]['eta_key']),reverse=True);selected=[cand[0]]
 while len(selected)<3:
  def score(x):
   z=(eta(x[2])-AFF)/SCALE;dist=min(np.linalg.norm(z-(eta(y[2])-AFF)/SCALE) for y in selected)
   return (dist+.02*x[0]+.01*x[1],x[0],x[1],x[2]['eta_key'])
  selected.append(max([x for x in cand if x not in selected],key=score))
 rows=[];tasks=[];have={(r['state_id'],r['eta_key']) for r in inv}
 for mode,(mn,mean,r) in enumerate(selected):
  v=eta(r);assert np.all(((v-AFF)/SCALE)@HS[:,:3].T+HS[:,3]<=1e-10)
  for sid in TOY:
   cached=(sid,r['eta_key']) in have;rows.append(dict(state_id=sid,scenario='ToyGiveWay_2A',mode_id=f'DB_CORE_{mode}',eta_key=r['eta_key'],eta1=v[0],eta2=v[1],eta3=v[2],DB_min_Q64=mn,DB_mean_Q64=mean,cached_Q64=cached,phase='cross_scenario_common_core_validation'))
   if not cached:
    for fi in range(64):tasks.append(dict(state_id=sid,eta=v.tolist(),mode_id=f'DB_CORE_{mode}',phase='cross_scenario_common_core_validation',eta_key=r['eta_key'],future_index=fi))
 write('fresh_common_core_manifest.csv',rows);(HERE/'plans/common_core').mkdir(parents=True,exist_ok=True)
 for sh in range(2):
  with open(HERE/f'plans/common_core/shard{sh}.jsonl','w') as f:
   for i,r in enumerate(tasks):
    if (i//64)%2==sh:f.write(json.dumps(r)+'\n')
 # Reuse compatible partial futures only for exact planned tuples.
 wanted={(r['state_id'],r['eta_key']) for r in rows};src=D/'orthoflow3_general_basin_geometry_v1/partial_continuation_cache.jsonl';reused=0
 with open(HERE/'partial_continuation_cache.jsonl','w') as o:
  if src.exists():
   for line in open(src):
    q=json.loads(line);k=(q['state_id'],key(q['eta']))
    if k in wanted:o.write(line);reused+=1
 dump('targeted_probe_rounds/common_core/cost_estimate.json',dict(scientific_question='Does any DB-panel-common robust eta satisfy the >=90% overall and >=80% per-scenario shared-core criterion?',candidate_eta=3,state_eta=len(rows),new_state_eta=sum(r['cached_Q64']=='False' for r in rows),maximum_new_continuations=len(tasks),reused_partial_records=reused,GPU_shards=2,estimated_wall_seconds=60+42*(len(tasks)/64)/2,created_before_Toy_outcomes=True))
 print(json.dumps(dict(selected_eta=3,state_eta=len(rows),new_continuations=len(tasks),partial_reuse=reused)))
if __name__=='__main__':main()
