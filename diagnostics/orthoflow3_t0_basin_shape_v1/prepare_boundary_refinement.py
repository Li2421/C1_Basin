#!/usr/bin/env python3
"""Pre-registered one-step boundary midpoint refinement after initial Q64 results."""
from __future__ import annotations
import csv,json
from pathlib import Path
from itertools import product
import numpy as np

HERE=Path('/home/zhihan/research/Basin_C1/diagnostics/orthoflow3_t0_basin_shape_v1')
def key(e):return np.asarray(e,dtype=np.float64).tobytes().hex()
def read(p):return list(csv.DictReader(open(p)))
def write(p,rows,fields=None):
 fields=fields or (list(rows[0]) if rows else ['state_id'])
 with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def nt(e):return (np.asarray(e,float)-np.array([.875,0.,.375]))/np.array([.75,1.,.75])
def boundary_metadata(sid,cached):
 """Reconstruct the frozen three nearest B63/non-B63 midpoint pairs.

 The first preparation CSV intentionally used a compact common schema,
 which omitted role-specific pair fields.  The exact frozen selection is
 recovered here from the immutable cached-Q64 table and matched by midpoint.
 """
 rows=[r for r in cached if r['state_id']==sid and r['Q64_available']=='True']
 good=[r for r in rows if r['B63']=='True'];bad=[r for r in rows if r['B63']!='True']
 allkeys={r['eta_key_float64'] for r in rows};pairs=[]
 for s,f in product(good,bad):
  ts=nt([float(s['eta1']),float(s['eta2']),float(s['eta3'])]);tf=nt([float(f['eta1']),float(f['eta2']),float(f['eta3'])])
  mid=(ts+tf)/2; raw=np.array([.875,0.,.375])+np.array([.75,1.,.75])*mid; mk=key(raw)
  if mk not in allkeys:pairs.append((float(np.linalg.norm(ts-tf)),s,f,raw,mk))
 pairs.sort(key=lambda z:(z[0],z[1]['eta_key_float64'],z[2]['eta_key_float64']))
 seen=set();out={}
 for d,s,f,raw,mk in pairs:
  if mk in seen:continue
  seen.add(mk);out[mk]={'boundary_id':len(out),'pair_distance':d,'success_key':s['eta_key_float64'],'failure_key':f['eta_key_float64']}
  if len(out)==3:break
 return out
def main():
 states=json.load(open(HERE/'fixed8_manifest.json'))['states'];cached=read(HERE/'cached_q64_manifest.csv');cache={(r['state_id'],r['eta_key_float64']):r for r in cached};roles=read(HERE/'targeted_probe_roles_initial.csv');all_initial=read(HERE/'targeted_probe_manifest.csv');refroles=[];reftargets=[]
 for st in states:
  sid=st['state_id'];res=read(HERE/'state_runs'/sid/f'initial_results_{sid}.csv');rmap={r['probe_id']:r for r in res};exist={r['eta_key_float64'] for r in all_initial if r['state_id']==sid}|{k for s,k in cache if s==sid};chosen=[]
  meta=boundary_metadata(sid,cached)
  boundary_roles=[]
  for r in roles:
   if r['state_id']!=sid or r['category']!='SUCCESS_FAILURE_BOUNDARY':continue
   m=meta.get(r['eta_key_float64'])
   if m is None:raise RuntimeError((sid,'missing frozen boundary metadata',r['probe_id']))
   boundary_roles.append({**r,**m})
  for role in sorted(boundary_roles,key=lambda r:int(r['boundary_id'])):
   mid=rmap[role['probe_id']];d=float(role['pair_distance'])
   if d/2 < .05:continue
   sk,fk=role['success_key'],role['failure_key'];success=cache[(sid,sk)];failure=cache[(sid,fk)]
   m_eta=np.array([float(mid['eta1']),float(mid['eta2']),float(mid['eta3'])]);s_eta=np.array([float(success['eta1']),float(success['eta2']),float(success['eta3'])]);f_eta=np.array([float(failure['eta1']),float(failure['eta2']),float(failure['eta3'])])
   if mid['B63']=='True':a,b=m_eta,f_eta;bracket='MID_B63_TO_FAILURE'
   else:a,b=s_eta,m_eta;bracket='SUCCESS_TO_MID_NONB63'
   eta=(a+b)/2;k=key(eta)
   rec={'state_id':sid,'probe_id':f'{sid}__R{len(chosen):02d}','category':'SUCCESS_FAILURE_BOUNDARY_REFINEMENT','eta1':eta[0],'eta2':eta[1],'eta3':eta[2],'eta_key_float64':k,'boundary_id':role['boundary_id'],'original_pair_distance':d,'refined_bracket_distance':d/2,'bracket_type':bracket,'parent_probe_id':role['probe_id'],'cached_q64':k in exist,'cached_lower_trials':0,'new_continuations_needed':0 if k in exist else 64}
   if k not in {x['eta_key_float64'] for x in chosen}:chosen.append(rec)
  if len(chosen)>3:raise RuntimeError((sid,'refinement cap',len(chosen)))
  if len([x for x in all_initial if x['state_id']==sid and not x['category'].endswith('DOMAIN_EXCLUDED')])+len(chosen)>32:raise RuntimeError((sid,'hard eta cap'))
  reftargets.extend(chosen);refroles.extend(chosen);(HERE/f'refinement_targets_{sid}.json').write_text(json.dumps(chosen,indent=2,sort_keys=True)+'\n')
 write(HERE/'targeted_probe_refinement_manifest.csv',reftargets)
 initial_roles=read(HERE/'targeted_probe_roles_initial.csv');write(HERE/'targeted_probe_roles.csv',initial_roles+refroles)
 (HERE/'refinement_cost.json').write_text(json.dumps({'max_refinements_per_state':3,'new_continuations':sum(int(x['new_continuations_needed']) for x in reftargets),'target_count':len(reftargets),'rule':'one midpoint per initial boundary bracket if initial pair distance/2 >=0.05'},indent=2,sort_keys=True)+'\n')
 print(json.dumps({'targets':len(reftargets),'new_continuations':sum(int(x['new_continuations_needed']) for x in reftargets),'per_state':{s['state_id']:len(json.load(open(HERE/f"refinement_targets_{s['state_id']}.json"))) for s in states}},indent=2))
if __name__=='__main__':main()
