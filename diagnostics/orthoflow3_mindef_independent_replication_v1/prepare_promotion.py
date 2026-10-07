#!/usr/bin/env python3
"""Make baseline and low-frontier promotion choices without true J."""
import csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');OUT=ROOT/'diagnostics/orthoflow3_mindef_independent_replication_v1';M=ROOT/'diagnostics/orthoflow3_representation_migration_v1';WIDTH=np.array([.75,1.,.75])
def dig(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
 if (OUT/'promotion_plan.json').exists():raise RuntimeError('frozen')
 states=json.load(open(OUT/'independent_active_state_manifest.json'))['states']; old=list(csv.DictReader(open(M/'subset_oracle_candidates.csv')));base=defaultdict(int)
 for r in old:
  if r['B63']=='True':base[r['state_id']]+=1
 by=defaultdict(list)
 for p in (OUT/'raw'/'screen').glob('*.jsonl'):
  for l in p.read_text().splitlines():
   r=json.loads(l);by[(r['state_id'],tuple(round(float(x),15) for x in r['eta']))].append(r)
 screen=[]; choices={};arms=[]
 for s in states:
  cand=[]
  for (sid,e),rr in by.items():
   if sid!=s['state_id']:continue
   g=[float(np.asarray(x['first_step']['g_raw'])@np.asarray(x['first_step']['g_raw'])) for x in rr];eta=np.array(e)
   cand.append({'state_id':sid,'sobol_index':int(rr[0]['arm_id'].split('__I')[-1]),'eta':json.dumps(list(e)),'successes_8':sum(x['success'] for x in rr),'eligible_8of8':sum(x['success'] for x in rr)==8,'R_raw':float(eta@eta),'R_gram':float(np.mean(g))})
  screen+=cand;good=sorted([x for x in cand if x['eligible_8of8']],key=lambda x:x['sobol_index']);picked=[]
  if base[s['state_id']]<2 and good:
   picked.append((good[0],'BASELINE_first_8of8_frozen_index'))
  remain=[x for x in good if x not in [z[0] for z in picked]]
  if remain:picked.append((min(remain,key=lambda x:(x['R_raw'],x['sobol_index'])),'A_lowest_R_raw'))
  remain=[x for x in remain if x not in [z[0] for z in picked]]
  if remain:picked.append((min(remain,key=lambda x:(x['R_gram'],x['sobol_index'])),'B_lowest_R_gram_remaining'))
  choices[s['state_id']]=picked
 with (OUT/'screening_results.csv').open('w',newline='') as f:k=sorted({z for r in screen for z in r});w=csv.DictWriter(f,k);w.writeheader();w.writerows(screen)
 dec=[]
 for s in states:
  for rank,(c,reason) in enumerate(choices[s['state_id']],1):
   dec.append({**c,'promotion_order':rank,'rule':reason,'true_J_consulted':False})
   arms.append({'arm_id':f"PROMOTE{rank}__{s['state_id']}__I{c['sobol_index']}",'basis_family':'orthoflow3','state_id':s['state_id'],'state_file':s['state_file'],'state_sha256':s['state_sha256'],'absolute_step':s['absolute_step'],'rng_namespace':s['rng_namespace'],'eta':json.loads(c['eta']),'sobol_index':c['sobol_index'],'seeds':[int(x) for x in s['matched_flow_seeds'][8:64]],'role':'baseline_or_lowfrontier_promotion','anchor_rank':s['selection_rank'],'offset_steps':0,'probe_id':reason})
 with (OUT/'promotion_decisions.csv').open('w',newline='') as f:k=sorted({z for r in dec for z in r});w=csv.DictWriter(f,k);w.writeheader();w.writerows(dec)
 unresolved=[s['state_id'] for s in states if len(choices[s['state_id']])<(3 if base[s['state_id']]<2 else 2)]
 plan={'schema':'orthoflow3_mindef_independent_promotion_v1','basis_family':'orthoflow3','stage':'promotion','selection':'for a state with <2 cached B63, first new 8/8 Sobol index is baseline; then A lowest raw and B lowest Gram among remaining. No true J used.','underresolved_after_screen':unresolved,'arms':arms};plan['content_sha256']=dig(plan)
 (OUT/'promotion_plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n');print(json.dumps({'base_counts':dict(base),'promotion_arms':len(arms),'new_continuations':sum(len(a['seeds']) for a in arms),'underresolved':unresolved},indent=2))
if __name__=='__main__':main()
