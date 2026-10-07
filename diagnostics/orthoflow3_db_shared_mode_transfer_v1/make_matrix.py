#!/usr/bin/env python3
import csv,json
from pathlib import Path
import numpy as np
H=Path(__file__).parent;SH=H.parent/'orthoflow3_shared_eta_codebook_v1'
def read(p):return list(csv.DictReader(open(p)))
def main():
 sp=json.load(open(H/'db_state_split.json'));states=sp['states'];sel=json.load(open(H/'selected_transform.json'));E=np.array(sel['transform']['eta']);raw=np.array([[float(r[f'eta{i}']) for i in (1,2,3)] for r in read(SH/'codebook_eta.csv')])
 tasks=[]
 for s in states:
  n={'train':16,'val':32,'test':64}[s['split']]
  for m,e in enumerate(E):
   for fi in range(n):tasks.append(dict(state_id=s['state_id'],eta=e.tolist(),future_index=fi,phase='transformed_matrix',probe_id=f'{s["state_id"]}_transformed_{m}_{fi}',controller='transformed',mode_id=m,split=s['split']))
  if s['split']=='test':
   for m,e in enumerate(raw):
    for fi in range(64):tasks.append(dict(state_id=s['state_id'],eta=e.tolist(),future_index=fi,phase='raw_toy_test_matrix',probe_id=f'{s["state_id"]}_raw_{m}_{fi}',controller='raw_toy',mode_id=m,split='test'))
  if s['split'] in ('val','test'):
   for fi in range(n):tasks.append(dict(state_id=s['state_id'],eta=[0.,0.,0.],future_index=fi,phase='safety_matrix',probe_id=f'{s["state_id"]}_safety_{fi}',controller='safety',mode_id=-1,split=s['split']))
 controllers=['transformed','raw_toy','safety'];groups={}
 for t in tasks:
  g=(t['controller'],t['mode_id']);groups.setdefault(g,[]).append(t)
 # Greedy balance whole eta groups; no eta is compiled on multiple shards.
 load=[0]*6;owner={}
 for g,z in sorted(groups.items(),key=lambda q:(-len(q[1]),q[0])):
  sh=int(np.argmin(load));owner[g]=sh;load[sh]+=len(z)
 p=H/'plans'/'matrix';p.mkdir(parents=True,exist_ok=True)
 for sh in range(6):
  with open(p/f'shard{sh}.jsonl','w') as f:
   for t in tasks:
    if owner[(t['controller'],t['mode_id'])]==sh:f.write(json.dumps(t)+'\n')
 transformed=sum(t['controller']=='transformed' for t in tasks);raw_n=sum(t['controller']=='raw_toy' for t in tasks);safe=sum(t['controller']=='safety' for t in tasks)
 cost={'selected_transform':sel['selected'],'transformed_matrix_continuations':transformed,'raw_toy_test_diagnostic_continuations':raw_n,'safety_continuations':safe,'total_new_planned_before_cache':len(tasks),'shard_loads':load,'soft_20000_checkpoint':{'exceeded':len(tasks)>20000,'reason_to_continue':'The central hypothesis requires paired binomial labels for 12 frozen transformed modes on source-isolated true-t0 states. This is a codebook matrix, not Basin mapping; raw Toy TEST is required to isolate deformation benefit. No new eta search is performed.'},'expected_wall_hours_6_cpu_shards':'2.5--5 after eta-grouped JIT reuse'}
 (H/'matrix_cost_estimate.json').write_text(json.dumps(cost,indent=2,sort_keys=True)+'\n');print(json.dumps(cost,indent=2))
if __name__=='__main__':main()
