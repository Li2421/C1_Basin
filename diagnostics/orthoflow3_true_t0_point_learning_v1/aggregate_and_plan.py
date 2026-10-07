#!/usr/bin/env python3
"""Aggregate completed searches, compute deficits, and freeze the next source-order wave."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path('/home/zhihan/research/Basin_C1'); HERE=ROOT/'diagnostics/orthoflow3_true_t0_point_learning_v1'
PACT=ROOT/'diagnostics/orthoflow3_t0_pact_training_readiness_v1'; T0=ROOT/'diagnostics/orthoflow3_t0_basin_structure_v1'; COMP=ROOT/'diagnostics/orthoflow3_t0_basin_completion_v1'
REQ={'train':24,'val':8,'test':8}; EXIST={'train':5,'val':1,'test':2}
def write(p,rows,fields=None):
 fields=fields or (list(rows[0]) if rows else ['state_id'])
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--next-wave',default='');a=ap.parse_args()
 targets=[]; attempted=[]; runtimes=[];screen=[];prom=[];local=[]
 for p in sorted((HERE/'state_runs').glob('*/selected_target.json')):
  t=json.load(open(p));targets.append(t);attempted.append({'state_id':t['state_id'],'source_group':t['source_group'],'split':t['split'],'split_rank':t['split_rank'],'status':t['status'],'reason':t.get('reason','')})
  rd=p.parent
  if (rd/'runtime.json').exists():runtimes.append(json.load(open(rd/'runtime.json')))
  for name,dst in [('eta_screening.csv',screen),('q64_promotions.csv',prom),('local_robustness_probes.csv',local)]:
   q=rd/name
   if q.exists():dst.extend(csv.DictReader(open(q)))
 write(HERE/'attempted_states.csv',attempted);write(HERE/'eta_screening.csv',screen);write(HERE/'q64_promotions.csv',prom);write(HERE/'local_robustness_probes.csv',local)
 usable=[x for x in targets if x['status']=='POINT_USABLE'];write(HERE/'selected_eta_targets_new.csv',usable)
 unresolved=[x for x in attempted if x['status']!='POINT_USABLE'];write(HERE/'unresolved_states.csv',unresolved)
 counts={s:EXIST[s]+sum(x['split']==s for x in usable) for s in REQ};deficit={s:max(0,REQ[s]-counts[s]) for s in REQ}
 attempted_ranks={s:{int(x['split_rank']) for x in targets if x['split']==s} for s in REQ}
 summary={'completed_searches':len(targets),'new_usable':len(usable),'new_unresolved':len(unresolved),'counts_including_existing':counts,'deficit':deficit,'runtime_new_continuations':sum(x['new_continuations'] for x in runtimes),'runtime_physical_steps':sum(x['physical_steps'] for x in runtimes),'runtime_worker_wall_seconds_sum':sum(x['wall_seconds'] for x in runtimes)}
 (HERE/'search_progress.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
 if a.next_wave and any(deficit.values()):
  frozen=json.load(open(HERE/'frozen_state_orders.json'));items=[]
  for s in ('train','val','test'):
   ranks=[int(x['rank']) for x in frozen['orders'][s] if int(x['rank']) not in attempted_ranks[s]][:deficit[s]]
   items.extend({'split':s,'rank':r} for r in ranks)
   if len(ranks)<deficit[s]:raise RuntimeError(f'inventory deficit {s}: need {deficit[s]} have {len(ranks)}')
  (HERE/f'{a.next_wave}.json').write_text(json.dumps({'frozen_after_prior_wave_only_to_fill_predeclared_deficit':True,'items':items},indent=2,sort_keys=True)+'\n')
  summary['next_wave']=a.next_wave;summary['next_wave_items']=len(items)
 print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=='__main__':main()
