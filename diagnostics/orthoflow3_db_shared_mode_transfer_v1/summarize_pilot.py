#!/usr/bin/env python3
import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
H=Path(__file__).parent;GEN=H.parent/'orthoflow3_general_basin_geometry_v1';SH=H.parent/'orthoflow3_shared_eta_codebook_v1'
def read(p):return list(csv.DictReader(open(p)))
def write(name,rows):
 with open(H/name,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 cb=read(SH/'codebook_eta.csv');raw_eta=np.array([[float(r[f'eta{i}']) for i in (1,2,3)] for r in cb]);tr=json.load(open(H/'toy_to_db_transform.json'))['candidates']
 eta={'raw_toy':raw_eta,**{k:np.array(tr[k]['eta']) for k in ('anisotropic','regularized_affine','affine_plus_mode_residual')}}
 rr=[]
 for p in (H/'raw').glob('transform_dev_pilot_*.jsonl'):
  rr += [json.loads(x) for x in open(p) if x.strip()]
 by=defaultdict(list)
 for r in rr:by[(r['state_id'],r['controller'],int(r['mode_id']))].append(r)
 inv=read(GEN/'exact_q64_inventory.csv');exact={}
 for r in inv:
  if r['scenario']=='DoubleBottleneck_4A' and int(r['trials'])==64:exact[(r['state_id'],tuple(np.round([float(r[f'eta{i}']) for i in (1,2,3)],12)))]=r
 states=[r['state_id'] for r in json.load(open(GEN/'double_bottleneck_state_panel.json'))];rows=[]
 for sid in states:
  for ctl,E in eta.items():
   for m,e in enumerate(E):
    k=(sid,ctl,m);cached=exact.get((sid,tuple(np.round(e,12))))
    if cached is not None:n=64;s=int(cached['successes']);dead=int(cached['deadlock']);tout=int(cached['timeout']);col=int(cached['collision']);src='cached_Q64'
    else:
     z={int(r['future_index']):r for r in by.get(k,[])}
     if len(z)<16:raise RuntimeError(('pilot incomplete',k,len(z)))
     z=[z[i] for i in range(16)];n=16;s=sum(bool(r['success']) for r in z);dead=sum(r['outcome']=='safe_deadlock' for r in z);tout=sum(r['outcome']=='timeout' for r in z);col=sum('collision' in str(r['outcome']) for r in z);src='new_Q16'
    q=s/n;strong=(s>=63 if n==64 else s>=15)
    rows.append(dict(state_id=sid,controller=ctl,mode_id=m,eta1=e[0],eta2=e[1],eta3=e[2],successes=s,trials=n,Q=q,strong=strong,deadlock=dead,timeout=tout,collision=col,source=src))
 write('transform_pilot_results.csv',rows)
 summ=[]
 for ctl in eta:
  z=[r for r in rows if r['controller']==ctl];best=[];union=0
  for sid in states:
   q=[r for r in z if r['state_id']==sid];best.append(max(r['Q'] for r in q));union+=any(r['strong'] for r in q)
  summ.append(dict(controller=ctl,states=len(states),strong_union_states=union,mean_best_Q=float(np.mean(best)),median_mode_Q=float(np.median([r['Q'] for r in z])),strong_state_mode_pairs=sum(r['strong'] for r in z),state_mode_pairs=len(z)))
 write('transform_pilot_summary.csv',summ)
 rank={'anisotropic':0,'regularized_affine':1,'affine_plus_mode_residual':2}
 candidates=[r for r in summ if r['controller']!='raw_toy']
 adequate=[r for r in candidates if r['strong_union_states']>=3 and r['mean_best_Q']>=.90]
 if adequate:win=sorted(adequate,key=lambda r:(rank[r['controller']],-r['strong_union_states'],-r['mean_best_Q']))[0]
 else:win=sorted(candidates,key=lambda r:(-r['strong_union_states'],-r['mean_best_Q'],rank[r['controller']]))[0]
 name=win['controller'];sel={'selected':name,'selection_scope':'four quarantined transform-development states; no VAL/TEST','rule':'lowest complexity with >=3/4 strong union and mean best Q>=0.90, otherwise best coverage/Q','pilot_summary':summ,'transform':tr[name]}
 (H/'selected_transform.json').write_text(json.dumps(sel,indent=2,sort_keys=True)+'\n')
 w=json.load(open(H/'working_state.json'));w.update(status='TRANSFORM_FROZEN',completed=w['completed']+['transform_dev_pilot','transform_selection'],selected_transform=name,next_action='build 64/16/16 mode matrix');(H/'working_state.json').write_text(json.dumps(w,indent=2,sort_keys=True)+'\n')
 print(json.dumps(sel,indent=2))
if __name__=='__main__':main()
