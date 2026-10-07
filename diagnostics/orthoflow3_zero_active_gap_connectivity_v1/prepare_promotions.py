#!/usr/bin/env python3
import ast,csv,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1'); H=ROOT/'diagnostics/orthoflow3_zero_active_gap_connectivity_v1'; M=ROOT/'diagnostics/orthoflow3_representation_migration_v1'
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def h(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def ek(e):return np.asarray(e,dtype=np.float64).tobytes().hex()
def allrows():
 out=[]
 for p in list(M.glob('raw/*/shard*.jsonl'))+list(H.glob('raw/initial_screen/shard*.jsonl')):out += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 return out
def main():
 states=json.load(open(H/'state_manifest.json'))['states'];sm={s['state_id']:s for s in states}; rows=allrows();lookup={(r['state_id'],ek(r['eta']),int(r['seed'])):r for r in rows if r['state_id'] in sm}
 grid=list(csv.DictReader(open(H/'radial_alpha_grid.csv'))); cand=list(csv.DictReader(open(M/'subset_oracle_candidates.csv')))
 zrob={x['state_id'] for x in cand if x['candidate_kind'].startswith('zero') and x['B63']=='True'}
 screen=[]; selected={}; reasons={}
 for sid in sm:
  for ep in sorted({x['endpoint'] for x in grid if x['state_id']==sid}):
   xs=sorted([x for x in grid if x['state_id']==sid and x['endpoint']==ep],key=lambda x:float(x['alpha'])); vals=[]
   for x in xs:
    e=[float(x[f'eta{i}']) for i in (1,2,3)]; rr=[lookup[(sid,ek(e),int(seed))] for seed in sm[sid]['matched_flow_seeds'][:8]]; suc=sum(r['success'] for r in rr); vals.append(suc)
    screen.append({**x,'successes':suc,'trials':8,'deadlock':sum(r['outcome']=='deadlock' for r in rr),'timeout':sum(r['outcome']=='timeout' for r in rr),'collision':sum(r['outcome']=='collision' for r in rr),'execution_error':sum(r['outcome']=='execution_error' for r in rr)})
   choose=set()
   # endpoints only need promotion if cached B63 evidence is absent.
   if sid not in zrob: choose.add(0); reasons[(sid,ep,0)]='alpha0_not_B63_resolved'
   for i in range(len(xs)-1):
    if (vals[i]==8 and vals[i+1]<=6) or (vals[i]<=6 and vals[i+1]==8):
     for j in range(max(0,i-1),min(len(xs),i+3)):choose.add(j);reasons[(sid,ep,j)]='adjacent_8_vs_le6_transition_and_neighbors'
   for i,v in enumerate(vals):
    if v==7:
     for j in range(max(0,i-1),min(len(xs),i+2)):choose.add(j);reasons[(sid,ep,j)]='7of8_ambiguous_transition_and_neighbors'
   if sid in zrob:
    for i,v in enumerate(vals[1:-1],1):
     if v<8:choose.add(i);reasons[(sid,ep,i)]='surprising_interior_both_endpoints_B63'
   for i in choose:selected[(sid,ep,float(xs[i]['alpha']))]=xs[i]
 with open(H/'radial_screening.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(screen[0]));w.writeheader();w.writerows(screen)
 # Neighborhood screening and at most first failing representative per state/alpha.
 dirs=list(csv.DictReader(open(H/'zero_neighborhood_directions.csv')));zn=[]
 for s in states:
  if s['state_id'] not in zrob:continue
  for d in dirs:
   anchor=np.array(ast.literal_eval(d['anchor_eta']))
   for a in (.05,.10,.20):
    e=a*anchor;rr=[lookup[(s['state_id'],ek(e),int(seed))] for seed in s['matched_flow_seeds'][:8]];suc=sum(r['success'] for r in rr)
    zn.append({'state_id':s['state_id'],'direction_index':d['direction_index'],'alpha':a,'eta1':e[0],'eta2':e[1],'eta3':e[2],'successes':suc,'trials':8})
 with open(H/'zero_neighborhood_results.csv','w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(zn[0]));w.writeheader();w.writerows(zn)
 tasks=[];reuse=0
 for (sid,ep,a),x in selected.items():
  s=sm[sid];e=np.array([float(x[f'eta{i}']) for i in (1,2,3)]); missing=[]
  for seed in s['matched_flow_seeds']:
   if (sid,ek(e),int(seed)) in lookup:reuse+=1
   else:missing.append(int(seed))
  if missing:tasks.append({'arm_id':f'P64__{s["selection_rank"]:02d}__{ep}__{a:.2f}','basis_family':'orthoflow3','state_id':sid,'selection_rank':s['selection_rank'],'state_file':s['state_file'],'state_sha256':s['state_sha256'],'absolute_step':s['absolute_step'],'rng_namespace':s['rng_namespace'],'eta_index':'BRIDGE','parameter_id':'BRIDGE','eta':e.tolist(),'seeds':missing,'audit_kind':'radial_promotion','endpoint':ep,'alpha':a,'promotion_reason':reasons[(sid,ep,xs.index(x))] if False else 'frozen_transition_rule'})
 # The frozen protocol requires representative 64-seed resolution of every
 # observed small-radius neighborhood miss.  These coordinates were fixed
 # before outcomes by zero_neighborhood_directions.csv and the alpha grid.
 for x in zn:
  if int(x['successes']) == 8:continue
  sid=x['state_id'];s=sm[sid];e=np.array([float(x[f'eta{i}']) for i in (1,2,3)]);missing=[]
  for seed in s['matched_flow_seeds']:
   if (sid,ek(e),int(seed)) in lookup:reuse+=1
   else:missing.append(int(seed))
  if missing:tasks.append({'arm_id':f'ZN64__{s["selection_rank"]:02d}__d{int(x["direction_index"]):02d}__{float(x["alpha"]):.2f}','basis_family':'orthoflow3','state_id':sid,'selection_rank':s['selection_rank'],'state_file':s['state_file'],'state_sha256':s['state_sha256'],'absolute_step':s['absolute_step'],'rng_namespace':s['rng_namespace'],'eta_index':'BRIDGE','parameter_id':'BRIDGE','eta':e.tolist(),'seeds':missing,'audit_kind':'zero_neighborhood_promotion','direction_index':int(x['direction_index']),'alpha':float(x['alpha']),'promotion_reason':'screening_below_8of8_predeclared_representative'})
 plan={'schema':'orthoflow3_migration_arm_plan_v1','stage':'promotion64','basis_family':'orthoflow3','selection_rule':'endpoints lacking B63 plus adjacent 8-vs<=6 transitions, neighbors, and 7/8 ambiguities','arms':tasks,'maximum_new_continuations':sum(len(x['seeds']) for x in tasks),'maximum_physical_steps':sum((850-x['absolute_step'])*len(x['seeds']) for x in tasks)};plan['content_sha256']=h(plan);dump(H/'promotion64_plan.json',plan)
 print(json.dumps({'promoted_points':len(selected),'arms_missing':len(tasks),'cached':reuse,'new':plan['maximum_new_continuations'],'steps_upper':plan['maximum_physical_steps'],'zero_neighborhood_under8':sum(int(x['successes'])<8 for x in zn)},indent=2))
if __name__=='__main__':main()
