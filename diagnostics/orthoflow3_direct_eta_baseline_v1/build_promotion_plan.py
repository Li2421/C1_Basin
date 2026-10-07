#!/usr/bin/env python3
"""Resolve zero B63 and freeze the first two active candidate promotions."""
import csv,json,hashlib
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('/home/zhihan/research/Basin_C1'); HERE=ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1'; Q=ROOT/'diagnostics/orthoflow3_q_learnability_v2'
def dump(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def load(directory,stem):
 out=[]
 for p in sorted((directory/'raw'/stem).glob('shard*.jsonl')):
  if p.stem.removeprefix('shard').isdigit(): out += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 return out
states=json.loads((HERE/'eligible_state_manifest.json').read_text())['selected_states']; split={x['state_id']:x['split'] for x in states}
cloud=json.loads((HERE/'eta_cloud_reference.json').read_text())['candidates']; index={r['probe_id']:i for i,r in enumerate(cloud)}; eta={r['probe_id']:r['eta'] for r in cloud}
base=load(Q,'base_rollout_plan'); zrows=[r for r in base if r['probe_id']=='zero']+load(Q,'zero_followup_plan')+load(HERE,'zero_completion_plan')
zb=defaultdict(dict)
for r in zrows: zb[r['state_id']][int(r['future_index'])]=r
zero_status={sid:sum(r['success'] for r in rows.values())>=63 and len(rows)==64 for sid,rows in zb.items()}
if len(zero_status)!=120 or any(len(zb[s['state_id']])!=64 for s in states): raise RuntimeError('zero completion incomplete')
candidates=defaultdict(lambda:defaultdict(list))
for r in base:
 if r['probe_id']!='zero': candidates[r['state_id']][r['probe_id']].append(r)
rankings={}; tasks=[]
for st in states:
 sid=st['state_id']
 if zero_status[sid]: continue
 ranked=[]
 for pid,rr in candidates[sid].items():
  succ=[x for x in rr if x['success']]
  ranked.append({'probe_id':pid,'eta':eta[pid],'screen_trials':len(rr),'screen_success':len(succ),'screen_q':len(succ)/len(rr),'mean_successful_jdef':sum(x['J_def'] for x in succ)/len(succ) if succ else None,'eta_cloud_index':index[pid]})
 ranked.sort(key=lambda r:(-r['screen_q'],float('inf') if r['mean_successful_jdef'] is None else r['mean_successful_jdef'],r['eta_cloud_index']))
 rankings[sid]=ranked
 for r in ranked[:2]:
  existing={int(x['future_index']) for x in candidates[sid][r['probe_id']]}
  for fi in range(64):
   if fi not in existing: tasks.append({'task_id':f"promote1__{sid}__{r['probe_id']}__f{fi:02d}",'candidate_id':f"{sid}__{r['probe_id']}",'state_id':sid,'split':st['split'],'probe_id':r['probe_id'],'eta':r['eta'],'future_index':fi,'promotion_round':1})
plan={'schema':'direct_eta_active_promotion_round1_v1','frozen_utc':datetime.now(timezone.utc).isoformat(),'orthoflow3_sha256':'51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38','future_root_seed':2026092702,'conditioning':'exact Q-v2 h conditioning','zero_b63_states':sum(zero_status.values()),'active_required_states':sum(not v for v in zero_status.values()),'tasks':tasks,'maximum_new_continuations':len(tasks),'rankings':rankings}
dump(HERE/'active_promotion_round1_plan.json',plan)
dump(HERE/'target_preflight.json',{'zero_b63_by_split':{sp:sum(zero_status[s['state_id']] for s in states if s['split']==sp) for sp in ('train','val','test')},'active_by_split':{sp:sum(not zero_status[s['state_id']] for s in states if s['split']==sp) for sp in ('train','val','test')},'zero_reused':1008,'zero_new':len(load(HERE,'zero_completion_plan')),'promotion_round1_new':len(tasks),'budget_after_round1_before_test_and_wide':15000-len(load(HERE,'zero_completion_plan'))-len(tasks)})
print(json.dumps(json.loads((HERE/'target_preflight.json').read_text()),indent=2))
