#!/usr/bin/env python3
import csv,json
from pathlib import Path
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_analytic_basin_margin_learning_v1';POINT=ROOT/'diagnostics/orthoflow3_true_t0_point_learning_v1'
inv=list(csv.DictReader(open(HERE/'exact_q64_inventory.csv')));targets=list(csv.DictReader(open(POINT/'selected_eta_targets.csv')))
vals=sorted({r['state_id'] for r in targets if r['split']=='val'})
modes={}
for r in targets:
 v=(float(r['target_eta1']),float(r['target_eta2']),float(r['target_eta3']));modes.setdefault(v,[]).append(r['state_id'])
have={(r['state_id'],float(r['eta1']),float(r['eta2']),float(r['eta3'])) for r in inv}
rows=[]
for sid in vals:
 for mid,(v,srcs) in enumerate(sorted(modes.items())):
  if (sid,*v) in have:continue
  rows.append({'probe_id':len(rows),'state_id':sid,'eta1':v[0],'eta2':v[1],'eta3':v[2],'eta':json.dumps(v),'mode_index':mid,'eta_source_states':';'.join(srcs),'phase':'val_neighbor_target_q64'})
with open(HERE/'val_neighbor_probe_manifest.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
p=HERE/'plans/val_neighbor_q64';p.mkdir(parents=True,exist_ok=True)
for sh in range(6):
 with open(p/f'shard{sh}.jsonl','w') as f:
  for n,r in enumerate(rows):
   if n%6!=sh:continue
   for future in range(64):
    z=dict(r);z['eta']=[r['eta1'],r['eta2'],r['eta3']];z['future_index']=future;f.write(json.dumps(z,sort_keys=True)+'\n')
summary={'val_states':len(vals),'unique_robust_target_modes':len(modes),'new_exact_q64_state_eta':len(rows),'new_continuations':len(rows)*64,'test_states_touched':0,
 'frozen_role_rule':'even mode_index -> representation fitting acquisition; odd mode_index -> independent representation validation'}
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
