#!/usr/bin/env python3
"""Freeze cache-aware extension before inspecting any new transfer outcome."""
import csv,json
from pathlib import Path
import numpy as np
HERE=Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_t0_eta_continuity_cross_transfer_v1')
SRC=Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_true_t0_point_learning_v1')
rows=list(csv.DictReader(open(HERE/'new_cross_transfer_manifest.csv')))
(HERE/'new_cross_transfer_manifest_initial.csv').write_text((HERE/'new_cross_transfer_manifest.csv').read_text())
nearest=list(csv.DictReader(open(HERE/'nearest_neighbor_pairs.csv')))
pairs=list(csv.DictReader(open(HERE/'normalized_state_distances.csv')))
cache={(r['destination_state'],r['eta_source_state']) for r in csv.DictReader(open(HERE/'cached_cross_transfer.csv'))}
a=np.load(SRC/'point_learning_arrays.npz',allow_pickle=True);ids=[str(x) for x in a['state_ids']];etas=np.asarray(a['physical_target'],float)
targets={r['state_id']:r for r in csv.DictReader(open(SRC/'selected_eta_targets.csv'))}
pmap={tuple(sorted((r['state_i'],r['state_j']))):r for r in pairs}
selected={tuple(sorted((r['destination_state'],r['eta_source_state']))) for r in rows}
newcost=sum(r['cached_exact_q64'].lower()!='true' for r in rows)
order=[];seen=set()
for rank in (1,2,3,5):
 for r in sorted((z for z in nearest if int(z['neighbor_rank'])==rank),key=lambda z:(float(z['d_h']),z['state_id'],z['neighbor_state_id'])):
  k=tuple(sorted((r['state_id'],r['neighbor_state_id'])))
  if k not in seen:seen.add(k);order.append((k,f'neighbor_rank_{rank}_coverage'))
for r in sorted(pairs,key=lambda z:(float(z['d_h']),z['state_i'],z['state_j'])):
 k=(r['state_i'],r['state_j'])
 if k not in seen:seen.add(k);order.append((k,'distance_order_extension'))
nextpid=1+max(int(r['pair_id']) for r in rows)
added=[]
for k,reason in order:
 if k in selected:continue
 cost=sum((d,s) not in cache for d,s in [(k[0],k[1]),(k[1],k[0])])
 # Freeze at most 80 new directed transfers and at most 80 symmetric pairs total.
 if newcost+cost>80 or len(selected)>=80:continue
 p=pmap[k];pid=nextpid;nextpid+=1;selected.add(k);newcost+=cost
 for dest,src in [(k[0],k[1]),(k[1],k[0])]:
  e=etas[ids.index(src)];hit=(dest,src) in cache
  added.append({'pair_id':pid,'destination_state':dest,'eta_source_state':src,'state_id':dest,'eta':json.dumps(e.tolist()),'eta1':e[0],'eta2':e[1],'eta3':e[2],
   'd_h':p['d_h'],'d_eta':p['d_eta'],'distance_bin':p['distance_bin'],'selection_reason':reason,
   'source_eta_quality_tag':targets[src]['quality_tag'],'cached_exact_q64':hit})
 if newcost==80:break
allrows=rows+added
fields=list(allrows[0])
with open(HERE/'new_cross_transfer_manifest.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(allrows)
out=HERE/'plans/cross_transfer64b';out.mkdir(parents=True,exist_ok=True)
missing=[r for r in added if str(r['cached_exact_q64']).lower()!='true']
for sh in range(6):
 with open(out/f'shard{sh}.jsonl','w') as f:
  groups=[r for n,r in enumerate(missing) if n%6==sh]
  for g in groups:
   for future in range(64):
    z={k:v for k,v in g.items() if k!='cached_exact_q64'};z['eta']=[float(g['eta1']),float(g['eta2']),float(g['eta3'])];z.update(future_index=future,phase='cross_transfer64b',controller='neighbor_target')
    f.write(json.dumps(z,sort_keys=True)+'\n')
summary={'initial_pairs':len(rows)//2,'added_pairs':len(added)//2,'final_pairs':len(allrows)//2,'initial_new_directed':sum(r['cached_exact_q64'].lower()!='true' for r in rows),
 'added_new_directed':len(missing),'final_new_directed':sum(str(r['cached_exact_q64']).lower()!='true' for r in allrows),'final_cached_directed':sum(str(r['cached_exact_q64']).lower()=='true' for r in allrows)}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
