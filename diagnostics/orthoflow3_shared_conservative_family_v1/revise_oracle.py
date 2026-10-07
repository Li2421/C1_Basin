#!/usr/bin/env python3
"""Correct the frozen oracle simulator after detecting DB positive starvation.

The v0 generic maximin queried 15/16 negatives on every DB state.  This is not
the requested robust-anchor / cross-transfer / nearby-boundary fitting regime.
This one-time revision is frozen before any rollout and preserves every v0 file.
"""
import csv,json,hashlib,shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from prepare import AFF,SCALE,HERE,read,write,dump,eta,key,hval

def eligible(r):
 p=(r.get('phases','')+';'+r.get('retained_validation_batches','')).lower()
 return r['in_E_bridge']=='True' and not any(x in p for x in ('retained_validation','minimal_validation','polyhedral_validation','bounded_validation','ep0082_intrusion'))
def select(rr,budget):
 cand=[r for r in rr if eligible(r)];pos=[r for r in cand if r['B63']=='True'];neg=[r for r in cand if r['B63']=='False'];assert pos and len(cand)>=budget
 def z(r):return (eta(r)-AFF)/SCALE
 N=np.array([z(r) for r in neg]);anchor=max(pos,key=lambda r:(float(np.min(np.linalg.norm(N-z(r),axis=1))) if len(N) else 1.,-hval(r['eta_key'])))
 selected=[anchor]
 if neg:
  d=[np.linalg.norm(z(r)-z(anchor)) for r in neg]
  for j in (int(np.argmin(d)),int(np.argmax(d))):
   if neg[j] not in selected:selected.append(neg[j])
 while len(selected)<budget:
  robust=[r for r in selected if r['B63']=='True'];R=np.array([z(r) for r in robust]);S=np.array([z(r) for r in selected]);opts=[]
  for r in cand:
   if r in selected:continue
   phase=(r.get('phases','')+';'+r.get('sources','')).lower();special=('cross_transfer' in phase or 'independent_interpolation' in phase or 'conditional_extent' in phase)
   # Two local robust-support queries followed by one global maximin query.
   if len(selected)%3!=2:score=-float(np.min(np.linalg.norm(R-z(r),axis=1)))+.08*special
   else:score=float(np.min(np.linalg.norm(S-z(r),axis=1)))+.02*special
   opts.append(((score,-hval(r['eta_key'])),r))
  selected.append(max(opts,key=lambda x:x[0])[1])
 return selected
def main():
 rev=HERE/'oracle_revision.json'
 if rev.exists():raise FileExistsError('oracle revision already frozen')
 # Preserve v0 and its downstream offline outcomes before replacing required final paths.
 for name in ('oracle_budget_subsamples.json','oracle_budget_subsamples.csv','family_comparison.json'):
  p=HERE/name
  if p.exists():shutil.copy2(p,HERE/(p.stem+'_v0'+p.suffix))
 for d in ('affine_superbody','superbody_with_cuts','native_conditional_band','rotated_asymmetric_slab'):
  for n in ('fitted_parameters.csv','cv_results.csv','retained_metrics.csv','gate.json','oracle_budget_performance.csv'):
   p=HERE/d/n
   if p.exists():shutil.copy2(p,p.with_name(p.stem+'_v0'+p.suffix))
 inv=read(HERE/'exact_q64_inventory.csv');states=sorted(set(r['state_id'] for r in read(HERE/'scenario_state_inventory.csv')));by=defaultdict(list)
 for r in inv:
  if r['state_id'] in states:by[r['state_id']].append(r)
 sub={};rows=[];counts={}
 for sid in states:
  s=select(by[sid],32);sub[sid]={};counts[sid]={}
  for b in (16,24,32):
   sub[sid][str(b)]=[r['eta_key'] for r in s[:b]];counts[sid][str(b)]={'B63':sum(r['B63']=='True' for r in s[:b]),'non_B63':sum(r['B63']=='False' for r in s[:b])}
   for rank,r in enumerate(s[:b]):rows.append(dict(state_id=sid,scenario=r['scenario'],budget=b,rank=rank,eta_key=r['eta_key'],B63=r['B63'],Q64=r['Q64'],phases=r['phases']))
 dump('oracle_budget_subsamples.json',dict(selection='V1 adaptive fixed policy: one robust anchor; nearest/farthest known negatives; repeat two nearest-to-observed-robust queries then one global maximin query, with fixed provenance bonus for cross-transfer/independent-interpolation/conditional-extent candidates. Nested16/24/32. No family-specific or scenario-specific rule.',states=sub))
 write('oracle_budget_subsamples.csv',rows)
 dump('oracle_revision.json',dict(revision='V1',reason='V0 label-blind global maximin produced only1 B63 among16 labels on every DB state, so theta fitting failure was an acquisition-policy artifact rather than a family test.',v0_preserved=True,rule_shared_across_scenarios=True,uses_only_observed_labels_sequentially=True,counts=counts))
 print(json.dumps(counts))
if __name__=='__main__':main()
