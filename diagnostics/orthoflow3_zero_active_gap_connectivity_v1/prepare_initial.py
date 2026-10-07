#!/usr/bin/env python3
"""Freeze the 12-state bridge-domain audit and its initial 8-seed tasks."""
from __future__ import annotations
import ast,csv,hashlib,json,math,os
from pathlib import Path
import numpy as np
from scipy.stats import qmc

ROOT=Path('/home/zhihan/research/Basin_C1'); M=ROOT/'diagnostics/orthoflow3_representation_migration_v1'; HERE=ROOT/'diagnostics/orthoflow3_zero_active_gap_connectivity_v1'
LOW=np.array([.5,-.5,0.]); HIGH=np.array([1.25,.5,.75]); ALPHAS=[0,.05,.10,.20,.30,.40,.50,.60,.75,.90,1.]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def planhash(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def ekey(x): return np.asarray(x,dtype=np.float64).tobytes().hex()
def rows():
 out=[]
 for p in M.glob('raw/*/shard*.jsonl'):
  out += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 return out
def main():
 HERE.mkdir(parents=True,exist_ok=True)
 subset=json.load(open(M/'migration_subset_manifest.json')); states=subset['selected_states'][:12]; sm={x['state_id']:x for x in states}
 cand=list(csv.DictReader(open(M/'subset_oracle_candidates.csv')))
 old=rows(); oldkey={(r['state_id'],ekey(r['eta']),int(r['seed'])):r for r in old if r['state_id'] in sm}
 # Exact implementation audit: bounds occur only in search-plan sources, not correction or projection code.
 audit=f'''# Intermediate eta implementation-domain audit\n\nAuthoritative basis source: `{ROOT/'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py'}` (SHA256 `{sha(ROOT/'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py')}`).\n\n`BasisFields.correction` in `shared_control/basis_families.py` converts eta to float64 and requires only a finite length-three vector. It applies no lower/upper eta bound, clipping, rejection, or reinterpretation. The authoritative fixed-eta runner passes this correction directly to `u_safe + g`, then to the unchanged second `project_velocity_with_retry` projection. Neither projection receives eta or consults eta bounds.\n\nThe values eta1=[0.5,1.25], eta2=[-0.5,0.5], eta3=[0,0.75] occur in migration `orthoflow3_search_config.json`, `prepare_candidate_screen.py`, and related oracle-plan generation as `LOW/HIGH`; they are not controller constraints. No safety theorem or projection assumption in the inspected implementation depends on eta1>=0.5. Intermediate finite bridge eta are implementation-valid under the unchanged control stack.\n'''
 (HERE/'implementation_domain_audit.md').write_text(audit)
 bridge={'old_active_box':{'low':LOW.tolist(),'high':HIGH.tolist()},'zero':[0.,0.,0.],'definition':'conv({0} union ACTIVE_BOX) = { alpha*x : alpha in [0,1], x in ACTIVE_BOX }','membership':'eta in E_bridge iff exists alpha in [0,1] with eta=alpha*x and x in ACTIVE_BOX; radial and Sobol constructions store alpha,x explicitly','normalization_for_distances':'(eta-[0.875,0,0.375])/[0.75,1,0.75]','basis_sha256':sha(ROOT/'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py')}; dump(HERE/'bridge_domain_definition.json',bridge)
 endpoints=[]; endpoint_map={}
 for s in states:
  sid=s['state_id']; rr=[]
  for x in cand:
   if x['state_id']!=sid or x['B63']!='True': continue
   eta=np.array(ast.literal_eval(x['eta']),dtype=float)
   if np.allclose(eta,0): continue
   rr.append((int(x['eta_index']) if x['eta_index'].isdigit() else 10**9,eta,x))
  rr.sort(key=lambda z:z[0]);
  if not rr: raise RuntimeError(('missing B63 nonzero endpoint',sid))
  a=rr[0]; endpoints.append({'state_id':sid,'endpoint':'A','eta_index':a[0],'eta':a[1].tolist(),'source_kind':a[2]['candidate_kind'],'B63_cached':True})
  endpoint_map[(sid,'A')]=a[1]
  if len(rr)>1:
   scale=np.array([.75,1.,.75]); best=sorted(rr[1:],key=lambda z:(-np.linalg.norm((z[1]-a[1])/scale),z[0]))[0]
   endpoints.append({'state_id':sid,'endpoint':'B','eta_index':best[0],'eta':best[1].tolist(),'source_kind':best[2]['candidate_kind'],'B63_cached':True})
   endpoint_map[(sid,'B')]=best[1]
 with open(HERE/'active_endpoint_manifest.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['state_id','endpoint','eta_index','eta','source_kind','B63_cached']);w.writeheader();w.writerows(endpoints)
 dump(HERE/'state_manifest.json',{'source':str(M/'migration_subset_manifest.json'),'source_sha256':sha(M/'migration_subset_manifest.json'),'selection':'first 12 selected_states in frozen migration order','states':states})
 radial=[]; tasks=[]; reuse=0
 def add(s,eta,seeds,armid,meta):
  nonlocal reuse
  missing=[]
  for seed in seeds:
   if (s['state_id'],ekey(eta),int(seed)) in oldkey: reuse+=1
   else: missing.append(int(seed))
  if missing:
   tasks.append({'arm_id':armid,'basis_family':'orthoflow3','state_id':s['state_id'],'selection_rank':s['selection_rank'],'state_file':s['state_file'],'state_sha256':s['state_sha256'],'absolute_step':s['absolute_step'],'rng_namespace':s['rng_namespace'],'eta_index':meta.get('eta_index','BRIDGE'),'parameter_id':meta.get('parameter_id','BRIDGE'),'eta':np.asarray(eta,dtype=float).tolist(),'seeds':missing,**meta})
 for s in states:
  for ep in ('A','B'):
   if (s['state_id'],ep) not in endpoint_map: continue
   e=endpoint_map[(s['state_id'],ep)]
   for a in ALPHAS:
    eta=a*e; arm=f"R__{s['selection_rank']:02d}__{ep}__{a:.2f}"
    radial.append({'state_id':s['state_id'],'endpoint':ep,'alpha':a,'eta1':eta[0],'eta2':eta[1],'eta3':eta[2],'arm_id':arm})
    add(s,eta,s['matched_flow_seeds'][:8],arm,{'audit_kind':'radial','endpoint':ep,'alpha':a})
 # Fixed Sobol active anchors and alpha fractions; their coordinates are frozen before outcomes.
 sob=qmc.Sobol(d=3,scramble=False).random_base2(4)[:12]; ndirs=[]
 for j,u in enumerate(sob):
  anchor=LOW+(HIGH-LOW)*u
  ndirs.append({'direction_index':j,'anchor_eta':anchor.tolist(),'construction':'unscrambled Sobol d=3 index '+str(j)})
 with open(HERE/'zero_neighborhood_directions.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['direction_index','anchor_eta','construction']);w.writeheader();w.writerows(ndirs)
 # Use existing B63 zero label only to decide which state receives the predeclared neighborhood grid.
 zrob={x['state_id'] for x in cand if x['candidate_kind'].startswith('zero') and x['B63']=='True'}
 for s in states:
  if s['state_id'] not in zrob: continue
  for d in ndirs:
   for a in (.05,.10,.20):
    eta=a*np.array(d['anchor_eta']); arm=f"Z__{s['selection_rank']:02d}__{d['direction_index']:02d}__{a:.2f}"
    add(s,eta,s['matched_flow_seeds'][:8],arm,{'audit_kind':'zero_neighborhood','direction_index':d['direction_index'],'alpha':a})
 plan={'schema':'orthoflow3_migration_arm_plan_v1','stage':'initial_screen','basis_family':'orthoflow3','selection_rule':'frozen radial alpha grid plus frozen Sobol zero-neighborhood directions; exact cache omitted only by state/eta/seed tuple','arms':tasks,'maximum_new_continuations':sum(len(x['seeds']) for x in tasks),'maximum_physical_steps':sum((850-x['absolute_step'])*len(x['seeds']) for x in tasks)}; plan['content_sha256']=planhash(plan); dump(HERE/'initial_screen_plan.json',plan)
 with open(HERE/'radial_alpha_grid.csv','w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(radial[0]));w.writeheader();w.writerows(radial)
 dump(HERE/'cache_reuse_audit.json',{'source_records_scanned':len(old),'exact_cached_screen_tuples':reuse,'initial_new_continuations':plan['maximum_new_continuations'],'initial_worst_physical_steps':plan['maximum_physical_steps'],'key':'state_id,float64 eta,matched Flow seed,implementation/projection/horizon/rng namespace','source_raw':str(M/'raw')})
 (HERE/'protocol.md').write_text('# OrthoFlow3 zero/ACTIVE gap connectivity audit v1\n\nFrozen 12-state migration-prefix, endpoints, alpha grid, bridge domain, and zero-neighborhood Sobol anchors were written before new outcomes. B63 is >=63/64; screen is 8 matched migration Flow seeds.\n')
 print(json.dumps({'states':len(states),'paths':len({(r['state_id'],r['endpoint']) for r in radial}),'zero_neighborhood_states':len(zrob & set(sm)),'cached':reuse,'new':plan['maximum_new_continuations'],'steps_upper':plan['maximum_physical_steps']},indent=2))
if __name__=='__main__': main()
