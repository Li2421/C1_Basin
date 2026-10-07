#!/usr/bin/env python3
"""Deterministic staged continuous-eta search planning and aggregation."""
from __future__ import annotations
import argparse,csv,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
H=Path(__file__).parent
def read(p):return list(csv.DictReader(open(H/p)))
def write(p,rows,fields=None):
 p=H/p;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  if not rows:
   f.write(','.join(fields or ['state_id'])+'\n');return
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]),extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(p,x):(H/p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def loadraw(prefix):
 out=[]
 for p in (H/'raw').glob(prefix+'*.jsonl'):out += [json.loads(x) for x in open(p) if x.strip()]
 bad=[r for r in out if r.get('scientific_outcome_valid') is not True]
 if bad:raise RuntimeError(f'{prefix}: {len(bad)} scientifically invalid records; repair before interpreting')
 return out
def makeplan(name,tasks):
 groups=defaultdict(list)
 for t in tasks:groups[(t['controller'],t['mode_id'])].append(t)
 bins=[[] for _ in range(6)]
 for g,v in sorted(groups.items(),key=lambda x:(-len(x[1]),x[0])):
  # Keep a candidate eta on one shard when practical; many candidate groups
  # make greedy group allocation naturally balanced.
  j=min(range(6),key=lambda z:(len(bins[z]),z));bins[j].extend(v)
 out=H/'plans'/name;out.mkdir(parents=True,exist_ok=True)
 for j,b in enumerate(bins):
  with open(out/f'shard{j}.jsonl','w') as f:
   for t in b:f.write(json.dumps(t)+'\n')
 return [len(b) for b in bins]
def candidates(stage):
 return [r for r in read('continuous_candidate_cloud.csv') if int(r['stage'])==stage]
def eta(r):return [float(r[f'eta{i}']) for i in (1,2,3)]
def z(r):return np.array([float(r[f'z{i}']) for i in (1,2,3)])
def plan_screen(stage):
 if stage==1:states=json.load(open(H/'continuous_search_targets.json'))['states']
 else:states=json.load(open(H/'stage1_decision.json'))['unresolved_states']
 cc=candidates(stage);tasks=[]
 for sid in states:
  for ci,c in enumerate(cc):
   for fi in range(8):tasks.append(dict(state_id=sid,eta=eta(c),future_index=fi,phase=f'continuous_screen{stage}',probe_id=f'{sid}_{c["candidate_id"]}_s{stage}_{fi}',controller=f'screen{stage}',mode_id=ci,candidate_id=c['candidate_id'],candidate_kind=c['kind']))
 loads=makeplan(f'screen{stage}',tasks);dump(f'screen{stage}_cost.json',{'states':len(states),'eta_candidates_per_state':len(cc),'continuations':len(tasks),'shard_loads':loads,'decision':f'test frozen stage-{stage} cloud on all unresolved states'})
 print(json.dumps({'stage':stage,'states':len(states),'candidates':len(cc),'tasks':len(tasks),'loads':loads},indent=2))
def aggregate_screen(stage):
 states=json.load(open(H/'continuous_search_targets.json'))['states'] if stage==1 else json.load(open(H/'stage1_decision.json'))['unresolved_states'];cc=candidates(stage);cm={r['candidate_id']:r for r in cc};raw=loadraw(f'screen{stage}_');by=defaultdict(dict)
 for r in raw:by[(r['state_id'],r['candidate_id'])][int(r['future_index'])]=r
 rows=[];prom=[]
 for sid in states:
  local=[]
  for c in cc:
   v=by[(sid,c['candidate_id'])]
   if len(v)!=8:raise RuntimeError(('incomplete screen',stage,sid,c['candidate_id'],len(v)))
   k=sum(bool(v[i]['success']) for i in range(8));o=Counter(str(v[i]['outcome']) for i in range(8));q=dict(state_id=sid,stage=stage,candidate_id=c['candidate_id'],candidate_kind=c['kind'],frozen_order=int(c['frozen_order']),eta1=c['eta1'],eta2=c['eta2'],eta3=c['eta3'],screen_successes=k,screen_trials=8,screen_Q=k/8,deadlock=sum(o[x] for x in ('strict_deadlock','safe_deadlock','deadlock')),timeout=o['timeout'],collision=sum('collision' in x for x in o for _ in range(o[x])),domain_clearance=float(c['domain_clearance']),nearest_mode_distance=float(c['nearest_mode_distance']));rows.append(q);local.append(q)
  fails=[q for q in local if q['screen_successes']<=5]
  for q in local:q['nearest_screen_failure_distance']=min((float(np.linalg.norm(z(cm[q['candidate_id']])-z(cm[f['candidate_id']]))) for f in fails if f['candidate_id']!=q['candidate_id']),default=math.inf)
  eligible=[q for q in local if q['screen_successes']>=7]
  eligible.sort(key=lambda q:(-q['screen_successes'],-q['nearest_screen_failure_distance'],-q['domain_clearance'],q['frozen_order'],q['candidate_id']))
  for rank,q in enumerate(eligible[:4],1):prom.append({**q,'promotion_rank':rank})
 write(f'screening_stage{stage}.csv',rows);write(f'promotions_stage{stage}.csv',prom,fields=['state_id','stage','candidate_id','candidate_kind','frozen_order','eta1','eta2','eta3','screen_successes','screen_trials','screen_Q','deadlock','timeout','collision','domain_clearance','nearest_mode_distance','nearest_screen_failure_distance','promotion_rank'])
 tasks=[]
 for q in prom:
  e=[float(q[f'eta{i}']) for i in (1,2,3)]
  for fi in range(8,64):tasks.append(dict(state_id=q['state_id'],eta=e,future_index=fi,phase=f'continuous_promote{stage}',probe_id=f"{q['state_id']}_{q['candidate_id']}_p{stage}_{fi}",controller=f'promote{stage}',mode_id=int(q['frozen_order']),candidate_id=q['candidate_id'],candidate_kind=q['candidate_kind']))
 loads=makeplan(f'promote{stage}',tasks);dump(f'promote{stage}_cost.json',{'states':len(states),'promotions':len(prom),'continuations':len(tasks),'shard_loads':loads})
 print(json.dumps({'stage':stage,'screen_rows':len(rows),'promotions':len(prom),'promotion_tasks':len(tasks),'loads':loads},indent=2))
def aggregate_promote(stage):
 screen=read(f'screening_stage{stage}.csv');prom=read(f'promotions_stage{stage}.csv');screenraw=loadraw(f'screen{stage}_');promraw=loadraw(f'promote{stage}_') if prom else [];by=defaultdict(dict)
 for r in screenraw+promraw:by[(r['state_id'],r['candidate_id'])][int(r['future_index'])]=r
 rows=[]
 for q in prom:
  v=by[(q['state_id'],q['candidate_id'])]
  if len(v)!=64:raise RuntimeError(('incomplete promotion',stage,q['state_id'],q['candidate_id'],len(v)))
  k=sum(bool(v[i]['success']) for i in range(64));o=Counter(str(v[i]['outcome']) for i in range(64));rows.append({**q,'Q64_successes':k,'Q64_trials':64,'Q64':k/64,'B63':k>=63,'Q64_deadlock':sum(o[x] for x in ('strict_deadlock','safe_deadlock','deadlock')),'Q64_timeout':o['timeout'],'Q64_collision':sum('collision' in x for x in o for _ in range(o[x])),'Q64_J_def':float(np.mean([v[i]['J_def'] for i in range(64)])),'Q64_episode_length':float(np.mean([v[i]['episode_steps'] for i in range(64)]))})
 write(f'q64_stage{stage}.csv',rows,fields=(list(rows[0]) if rows else ['state_id','candidate_id','B63']))
 target=json.load(open(H/'continuous_search_targets.json'))['states'] if stage==1 else json.load(open(H/'stage1_decision.json'))['unresolved_states'];solved={r['state_id'] for r in rows if str(r['B63']).lower()=='true'};unresolved=[s for s in target if s not in solved]
 decision={'stage':stage,'target_states':target,'B63_states':sorted(solved),'unresolved_states':unresolved,'promoted_candidates':len(rows)};dump(f'stage{stage}_decision.json',decision);print(json.dumps(decision,indent=2))
 if stage==1:plan_screen(2)
def main():
 a=argparse.ArgumentParser();a.add_argument('action',choices=['plan_screen','aggregate_screen','aggregate_promote']);a.add_argument('stage',type=int,choices=[1,2]);q=a.parse_args();globals()[q.action](q.stage)
if __name__=='__main__':main()
