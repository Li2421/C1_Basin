#!/usr/bin/env python3
import csv,json
from pathlib import Path
import numpy as np
H=Path(__file__).parent;GEN=H.parent/'orthoflow3_general_basin_geometry_v1';SH=H.parent/'orthoflow3_shared_eta_codebook_v1'
def read(p):return list(csv.DictReader(open(p)))
def main():
 tr=json.load(open(H/'toy_to_db_transform.json'))['candidates'];cb=read(SH/'codebook_eta.csv');raw=np.array([[float(r[f'eta{i}']) for i in (1,2,3)] for r in cb])
 # Similarity is retained in the offline table but is not rolled out: its
 # residual is dominated by both constrained affine candidates.
 codebooks={'raw_toy':raw,'anisotropic':np.array(tr['anisotropic']['eta']),'regularized_affine':np.array(tr['regularized_affine']['eta']),'affine_plus_mode_residual':np.array(tr['affine_plus_mode_residual']['eta'])}
 states=json.load(open(GEN/'double_bottleneck_state_panel.json'));inv=read(GEN/'exact_q64_inventory.csv');cached=set()
 for r in inv:
  if r['scenario']=='DoubleBottleneck_4A' and int(r['trials'])==64:cached.add((r['state_id'],tuple(np.round([float(r[f'eta{i}']) for i in (1,2,3)],12))))
 jobs=[];reuse=[]
 for s in states:
  for c,E in codebooks.items():
   for m,e in enumerate(E):
    key=(s['state_id'],tuple(np.round(e,12)))
    if key in cached:
     reuse.append({'state_id':s['state_id'],'controller':c,'mode_id':m,'eta1':e[0],'eta2':e[1],'eta3':e[2],'source':'existing_exact_Q64'});continue
    for fi in range(16):jobs.append({'state_id':s['state_id'],'eta':e.tolist(),'future_index':fi,'phase':'transform_dev_pilot','probe_id':f'{s["state_id"]}_{c}_{m}_{fi}','controller':c,'mode_id':m,'split':'transform_dev'})
 p=H/'plans'/'transform_dev_pilot';p.mkdir(parents=True,exist_ok=True)
 controllers=list(codebooks)
 for sh in range(6):
  with open(p/f'shard{sh}.jsonl','w') as f:
   for r in jobs:
    # Keep every continuation for one eta on exactly one shard.  The DB
    # rollout graph specializes on eta; this avoids six redundant JIT
    # compilations per mode without changing any rollout input.
    group=controllers.index(r['controller'])*12+int(r['mode_id'])
    if group%6==sh:f.write(json.dumps(r)+'\n')
 with open(H/'pilot_cached_reuse.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['state_id','controller','mode_id','eta1','eta2','eta3','source']);w.writeheader();w.writerows(reuse)
 (H/'pilot_cost.json').write_text(json.dumps({'jobs':len(jobs),'cached_state_mode_Q64':len(reuse),'controllers':list(codebooks),'states':len(states),'continuations_if_no_cache':len(states)*len(codebooks)*12*16,'decision_resolved':'Choose the lowest-complexity transform with adequate robust union coverage before the full matrix.'},indent=2)+'\n')
 print(json.dumps({'jobs':len(jobs),'cached_Q64_pairs':len(reuse)},indent=2))
if __name__=='__main__':main()
