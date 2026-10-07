#!/usr/bin/env python3
"""Freeze exact Q-v2 split/cloud reuse and eta=0 B63 completion plan."""
import csv, hashlib, json, shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path('/home/zhihan/research/Basin_C1'); HERE=ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1'; Q=ROOT/'diagnostics/orthoflow3_q_learnability_v2'
HERE.mkdir(parents=True,exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def raw(stem):
 out=[]
 for p in sorted((Q/'raw'/stem).glob('shard*.jsonl')):
  if p.stem.removeprefix('shard').isdigit(): out += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 return out

split=json.loads((Q/'state_split_manifest.json').read_text()); eligible=json.loads((Q/'eligible_state_manifest.json').read_text())
dump(HERE/'reused_q_split_manifest.json',{'schema':'direct_eta_reused_q_v2_split','source':str(Q/'state_split_manifest.json'),'source_sha256':sha(Q/'state_split_manifest.json'),'exact_copy':split})
dump(HERE/'source_group_leakage_audit.json',{'passed':not any(split['overlap'].values()),'overlap':split['overlap'],'counts':split['counts'],'split_unit':'leakage_group','exact_q_v2_split_reused':True})
dump(HERE/'eligible_state_manifest.json',eligible); shutil.copy2(Q/'conditioning_features.npz',HERE/'conditioning_features.npz')
eta=[]
with (Q/'eta_probe_cloud.csv').open() as f:
 for r in csv.DictReader(f): eta.append({'probe_id':r['probe_id'],'eta':[float(r['eta1']),float(r['eta2']),float(r['eta3'])],'source':r['source']})
dump(HERE/'eta_cloud_reference.json',{'source':str(Q/'eta_probe_cloud.csv'),'source_sha256':sha(Q/'eta_probe_cloud.csv'),'count':len(eta),'candidates':eta})
records=raw('base_rollout_plan')+raw('zero_followup_plan')
zero=defaultdict(dict)
for r in records:
 if r['probe_id']=='zero': zero[r['state_id']][int(r['future_index'])]=r
tasks=[]
for st in eligible['selected_states']:
 for i in range(64):
  if i not in zero[st['state_id']]:
   tasks.append({'task_id':f"zero64__{st['state_id']}__f{i:02d}",'candidate_id':f"{st['state_id']}__zero",'state_id':st['state_id'],'split':st['split'],'probe_id':'zero','eta':[0.,0.,0.],'future_index':i})
plan={'schema':'direct_eta_zero_completion_plan_v1','frozen_utc':datetime.now(timezone.utc).isoformat(),'orthoflow3_sha256':'51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38','future_root_seed':2026092702,'conditioning':'exact Q-v2 fixed query Flow key; future-only resampling','tasks':tasks,'existing_reused_continuations':sum(len(v) for v in zero.values()),'maximum_new_continuations':len(tasks)}
dump(HERE/'zero_completion_plan.json',plan)
print(json.dumps({'states':len(eligible['selected_states']),'reused_zero':plan['existing_reused_continuations'],'new_zero':len(tasks)},indent=2))
