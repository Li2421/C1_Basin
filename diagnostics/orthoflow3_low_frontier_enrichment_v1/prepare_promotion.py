#!/usr/bin/env python3
"""Freeze B63 promotions using only predeclared screening and local proxies."""
import csv, hashlib, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path('/home/zhihan/research/Basin_C1'); OUT=ROOT/'diagnostics/orthoflow3_low_frontier_enrichment_v1'; WIDTH=np.array([.75,1.,.75])
def dig(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def rows():
  ans=[]
  for p in sorted((OUT/'raw'/'screen').glob('*.jsonl')):
    ans += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
  return ans
def main():
  if (OUT/'promotion_plan.json').exists(): raise RuntimeError('promotion already frozen')
  rr=rows(); by=defaultdict(list)
  for r in rr: by[(r['state_id'],tuple(round(float(x),15) for x in r['eta']))].append(r)
  states=json.loads((OUT/'active_state_manifest.json').read_text())['states']; screening=[]; select={}
  for s in states:
    cand=[]
    for (sid,eta),x in by.items():
      if sid!=s['state_id']: continue
      if len({int(a['seed']) for a in x})!=8: raise RuntimeError((sid,eta,len(x)))
      success=sum(bool(a['success']) for a in x); e=np.array(eta); gram=[]
      for a in x:
        fs=a.get('first_step')
        if fs is not None: g=np.asarray(fs['g_raw']); gram.append(float(g@g))
      rec={'state_id':sid,'pair_rank':s['pair_rank'],'side':s['side'],'sobol_index':next(a['arm_id'].split('__I')[-1] for a in x),'eta':json.dumps(list(eta)),'successes_8':success,'R_raw':float(e@e),'R_gram':float(np.mean(gram)),'eligible_8of8':success==8}
      screening.append(rec); cand.append(rec)
    eligible=[x for x in cand if x['eligible_8of8']]
    rawsort=sorted(eligible,key=lambda x:(x['R_raw'],int(x['sobol_index'])))
    chosen=[]
    if rawsort: chosen.append((rawsort[0],'A_lowest_R_raw'))
    remaining=[x for x in eligible if x not in [z[0] for z in chosen]]
    if remaining: chosen.append((sorted(remaining,key=lambda x:(x['R_gram'],int(x['sobol_index'])))[0],'B_lowest_R_gram_remaining'))
    # This path is only used when the required distinct B cannot exist.
    if len(chosen)==1 and len(eligible)>1:
      a=np.array(json.loads(chosen[0][0]['eta'])); b=max(remaining,key=lambda x:(float(np.linalg.norm((np.array(json.loads(x['eta']))-a)/WIDTH)),-int(x['sobol_index'])))
      chosen.append((b,'B_farthest_normalized_fallback'))
    select[s['state_id']]=chosen
  with (OUT/'screening_results.csv').open('w',newline='') as f:
    keys=sorted({k for r in screening for k in r});w=csv.DictWriter(f,keys);w.writeheader();w.writerows(screening)
  decisions=[]; arms=[]
  for s in states:
    chosen=select[s['state_id']]
    for rank,(c,reason) in enumerate(chosen,1):
      decisions.append({**c,'promotion_rank':rank,'rule':reason,'true_J_consulted':False})
      eta=json.loads(c['eta']); idx=int(c['sobol_index']); arms.append({'arm_id':f"PROMOTE{rank}__{s['state_id']}__I{idx:03d}",'basis_family':'orthoflow3','state_id':s['state_id'],'state_file':s['state_file'],'state_sha256':s['state_sha256'],'absolute_step':s['absolute_step'],'rng_namespace':s['rng_namespace'],'eta':eta,'sobol_index':idx,'seeds':[int(x) for x in s['matched_flow_seeds'][8:64]],'role':'predeclared_low_frontier_promotion','anchor_rank':s['pair_rank'],'offset_steps':4 if s['side']=='neighbor_t+4' else 0,'probe_id':f'PROMOTE{rank}'})
  with (OUT/'promotion_decisions.csv').open('w',newline='') as f:
    keys=sorted({k for r in decisions for k in r}) if decisions else ['state_id'];w=csv.DictWriter(f,keys);w.writeheader();w.writerows(decisions)
  unresolved=[s['state_id'] for s in states if len(select[s['state_id']])<2]
  plan={'schema':'orthoflow3_low_frontier_promotion_v1','basis_family':'orthoflow3','stage':'promote_predeclared','selection':'among new 8/8 candidates: A lowest raw eta norm; B lowest start Gram energy among remaining; no true J inspected. Seeds 9..64 extend screening to B63.','underresolved_after_screen':unresolved,'arms':arms};plan['content_sha256']=dig(plan)
  (OUT/'promotion_plan.json').write_text(json.dumps(plan,indent=2,sort_keys=True)+'\n')
  print(json.dumps({'screen_records':len(rr),'eligible_by_state':{s['state_id']:sum(x['eligible_8of8'] for x in screening if x['state_id']==s['state_id']) for s in states},'promotion_arms':len(arms),'new_continuations':sum(len(a['seeds']) for a in arms),'underresolved':unresolved},indent=2))
if __name__=='__main__': main()
