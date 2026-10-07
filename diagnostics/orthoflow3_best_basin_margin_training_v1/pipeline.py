#!/usr/bin/env python3
"""Geometry preparation for best-available set-valued margin supervision.

This script intentionally ranks imperfect analytic labels; it never treats a
geometry gate as an authority to stop neural training.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,shutil,sys,time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path('/home/zhihan/research/Basin_C1');D=ROOT/'diagnostics';H=Path(__file__).parent
POINT=D/'orthoflow3_true_t0_point_learning_v1';XFER=D/'orthoflow3_t0_eta_continuity_cross_transfer_v1';SHARED=D/'orthoflow3_shared_conservative_family_v1';EP82=D/'orthoflow3_ep0082_failure_intrusion_v1'
sys.path.insert(0,str(SHARED));from shared_families import fit_superbody,fit_band,contains,violation,domain_contains,HS
from shared_families import pca_frame,canonical,preimage
AFF=np.array([.875,0.,.375]); SCALE=np.array([.75,1.,.75])
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(open(p)))
def dump(name,x):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def write(name,rows,fields=None):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True)
 fields=fields or (list(rows[0]) if rows else ['state_id'])
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def key(z):return np.asarray(z,dtype='<f8').tobytes().hex()
def z_from(r):return (np.array([float(r['eta1']),float(r['eta2']),float(r['eta3'])])-AFF)/SCALE
def physical(z):return AFF+SCALE*np.asarray(z)

def add(obs,sid,z,b63,source,q64=None):
 k=key(z); old=obs[sid].get(k)
 rec=dict(z=np.asarray(z,float),B63=bool(b63),source=source,Q64=q64)
 if old is not None and old['B63']!=rec['B63']:raise RuntimeError(('conflicting compatible Q64',sid,k,old,rec))
 obs[sid][k]=rec

def aggregate():
 targets=read(POINT/'selected_eta_targets.csv');ids={r['state_id']:r for r in targets};obs=defaultdict(dict)
 # Point search has exact Q64 promotion records under the frozen learning semantics.
 for p in sorted((POINT/'state_runs').glob('*/q64_promotions.csv')):
  for r in read(p):
   sid=r['state_id'];
   if sid not in ids or not r.get('successes64'):continue
   add(obs,sid,z_from(r),r['B63']=='True','point_search_Q64',float(r['Q64']))
 # Directed cross-transfer is exact Q64 at its destination and is valuable
 # multi-solution evidence, not a target selection heuristic.
 for name in ('cached_cross_transfer.csv','cross_transfer_results.csv'):
  for r in read(XFER/name):
   sid=r.get('destination_state',r.get('state_id'))
   if sid in ids and r.get('B63','')!='':add(obs,sid,z_from(r),r['B63']=='True','cross_transfer_Q64',float(r['Q64']))
 # Dense prior geometry is allowed only for the same state IDs and exact Q64.
 for r in read(SHARED/'exact_q64_inventory.csv'):
  if r['state_id'] in ids and r['scenario']=='ToyGiveWay_2A':add(obs,r['state_id'],z_from(r),r['B63']=='True','shared_geometry_Q64',float(r['Q64']))
 # Current ep0082 audit supersedes its earlier subset but is semantically identical.
 for r in read(EP82/'exact_q64_ep0082.csv'):
  if r['state_id'] in ids:add(obs,r['state_id'],z_from(r),r['B63']=='True','ep0082_topology_Q64',float(r['Q64']))
 out=[]
 for sid in ids:
  # target is a required robust witness even if its separately stored exact row
  # was deduplicated from another source above.
  t=ids[sid];z=(np.array([float(t[f'target_eta{i}']) for i in (1,2,3)])-AFF)/SCALE
  add(obs,sid,z,True,'frozen_selected_target',float(t['target_Q64']))
  for rec in obs[sid].values():out.append(dict(state_id=sid,split=ids[sid]['split'],eta1=physical(rec['z'])[0],eta2=physical(rec['z'])[1],eta3=physical(rec['z'])[2],z1=rec['z'][0],z2=rec['z'][1],z3=rec['z'][2],B63=rec['B63'],Q64=rec['Q64'],source=rec['source'],eta_key=key(rec['z'])))
 write('geometry_evidence_inventory.csv',out)
 dump('evidence_index.json',{'point_targets':str(POINT/'selected_eta_targets.csv'),'point_q64_promotions':'state_runs/*/q64_promotions.csv','cross_transfer':[str(XFER/'cached_cross_transfer.csv'),str(XFER/'cross_transfer_results.csv')],'shared_exact':str(SHARED/'exact_q64_inventory.csv'),'ep0082_exact':str(EP82/'exact_q64_ep0082.csv')})
 return ids,obs,out

def force_anchor(P,a):
 i=np.argmin(np.linalg.norm(P-a,axis=1));P=P.copy();P[[0,i]]=P[[i,0]];return P
def model_inner(m):
 q=json.loads(json.dumps(m));q['gamma']=float(q['gamma'])*.80;return q
def model_extent(m):
 if m['family'] in ('affine_superbody','superbody_with_cuts'):return float(2*np.linalg.norm(np.asarray(m['axes']))*m['gamma'])
 return float(2*np.linalg.norm(np.asarray(m['axes_tan']).tolist()+[.5*(m['upper'][0]-m['lower'][0])])*m['gamma'])
def score_model(m,P,N):
 inner=model_inner(m);pi=contains(inner,P,True);nf=contains(inner,N,True) if len(N) else np.zeros(0,bool)
 # The exact retained support, rather than complete-basin recall, is primary.
 sep=0.
 if pi.sum()>1:
  sep=float(np.max(np.linalg.norm(P[pi,None]-P[None,pi],axis=2)))
 return dict(known_false=int(nf.sum()),witnesses=int(pi.sum()),modes_separation=sep,diameter=model_extent(inner),full_recall=float(contains(m,P,False).mean()),retained_recall=float(pi.mean()),volume_proxy=float(np.prod(m.get('axes',m.get('axes_tan',[.01,.01])))*max(m.get('gamma',.1),.01)),complexity=int(m.get('parameter_count',0)))

def fit_local_body(P,N,W):
 """Evidence-capped superbody plus sparse affine exclusions.

 Sparse point-search states cannot justify an unconstrained expansion merely
 because no negative happened to be sampled.  This common family applies the
 same 0.18 normalized-axis evidence cap to every state, then uses <=3 cuts to
 remove known intrusion directions while preserving the selected anchor.
 """
 anchor=P[0];near=P[np.linalg.norm(P-anchor,axis=1)<=.35]
 R=pca_frame(near) if len(near)>=3 else np.eye(3)
 y=(near-anchor)@R
 axes=np.clip(np.maximum(np.quantile(np.abs(y),.85,axis=0)*1.25,.10),.10,.18)
 m=dict(family='superbody_with_cuts',p=4,center=anchor.tolist(),R=R.tolist(),axes=axes.tolist(),anchor=anchor.tolist(),gamma=.75,cuts=[],parameter_count=21,evidence_axis_cap=.18)
 # Greedy canonical half-space cuts: each preserves anchor and all closer
 # positive witnesses when possible; candidates that remove an in-retained
 # negative are considered in a deterministic order.
 for _ in range(3):
  inside=np.flatnonzero(contains(m,N,True)) if len(N) else []
  if len(inside)==0:break
  ua=canonical(m,anchor[None])[0];best=None
  for j in inside:
   un=canonical(m,preimage(m,N[j:j+1],True))[0];q=un-ua;d=np.linalg.norm(q)
   if d<1e-10:continue
   q=q/d
   for f in (.55,.65,.75,.85):
    cut={'q':q.tolist(),'b':float(q@ua+f*(q@un-q@ua))};trial=dict(m,cuts=m['cuts']+[cut])
    neg=int(contains(trial,N,True).sum());pos=int(contains(trial,P,True).sum())
    score=(neg,-pos,tuple(np.round(np.r_[q,cut['b']],10)))
    if best is None or score<best[0]:best=(score,trial)
  if best is None:break
  m=best[1]
 return m

def prepare():
 ids,obs,_=aggregate();candidates=[];models={}
 specs=[('evidence_capped_superbody_cut3_p4',fit_local_body),('rotated_superbody_p4',lambda P,N,W:fit_superbody(P,N,W,4,.50,.90,'affine_superbody',0)),('rotated_superbody_cut2_p4',lambda P,N,W:fit_superbody(P,N,W,4,.50,.90,'superbody_with_cuts',2)),('rotated_superbody_cut2_p6',lambda P,N,W:fit_superbody(P,N,W,6,.50,.90,'superbody_with_cuts',2)),('native_band_cut1_p4',lambda P,N,W:fit_band(P,N,W,4,.50,'native_conditional_band',1))]
 rows=[]
 for name,fn in specs:
  total=dict(false=0,witness=0,diam=[],sep=[],rec=[],usable=0,complexity=[]);models[name]={}
  for sid,t in ids.items():
   # TEST states are quarantined until the selected controller is frozen.
   # Their state-specific diagnostic fit is constructed only afterwards.
   if t['split']=='test':continue
   vv=list(obs[sid].values());P=np.array([x['z'] for x in vv if x['B63']],float).reshape(-1,3);N=np.array([x['z'] for x in vv if not x['B63']],float).reshape(-1,3);a=(np.array([float(t[f'target_eta{i}']) for i in (1,2,3)])-AFF)/SCALE;P=force_anchor(P,a);W=np.ones(len(P));m=fn(P,N,W)
   if m is None:continue
   sc=score_model(m,P,N);models[name][sid]=m;total['false']+=sc['known_false'];total['witness']+=sc['witnesses'];total['diam'].append(sc['diameter']);total['sep'].append(sc['modes_separation']);total['rec'].append(sc['retained_recall']);total['complexity'].append(sc['complexity']);total['usable']+=int(sc['diameter']>=.15 and sc['witnesses']>=1)
   rows.append(dict(candidate=name,state_id=sid,split=t['split'],**sc))
  candidates.append(dict(candidate=name,known_false_inclusions=total['false'],usable_states=total['usable'],median_diameter=float(np.median(total['diam'])),median_modes_separation=float(np.median(total['sep'])),median_retained_recall=float(np.median(total['rec'])),median_complexity=float(np.median(total['complexity'])),pareto_rank=''))
 # Lexicographic Pareto proxy fixed before fresh data: reliability, usable count,
 # extent, mode separation, retained recall, simplicity.
 # An expansion beyond the globally declared 0.18 evidence cap has no
 # negative-evidence support for sparse states.  Treat it as a reliability
 # risk before rewarding additional volume.
 for r in candidates:r['unsupported_expansion_risk']=float(r['median_diameter']>.70)
 ranked=sorted(candidates,key=lambda x:(x['known_false_inclusions'],x['unsupported_expansion_risk'],-x['usable_states'],-x['median_diameter'],-x['median_modes_separation'],-x['median_retained_recall'],x['median_complexity'],x['candidate']))
 for i,r in enumerate(ranked):r['pareto_rank']=i+1
 selected=ranked[0]['candidate'];dump('selected_family.json',dict(family=selected,form='evidence-capped rotated superbody intersected with <=3 affine canonical exclusion cuts' if selected.startswith('evidence_') else ('rotated superellipsoid / superbox body intersected with <=2 affine canonical exclusion cuts' if 'superbody' in selected else 'native conditional band with <=1 affine canonical exclusion cut'),selection='Frozen lexicographic Pareto: known exact false inclusion, unsupported sparse-evidence expansion risk, usable states, diameter, separated witnesses, retained recall, complexity. No neural outcome used.',global_hyperparameters={'p':4 if selected.endswith('p4') else 6,'gamma_retained':.75 if selected.startswith('evidence_') else .50,'gamma_inner_relative':.80,'max_exclusion_cuts':3 if selected.startswith('evidence_') else (2 if 'superbody' in selected else 1),'axis_cap_normalized':.18 if selected.startswith('evidence_') else None},quality_pending_fresh_validation=True))
 dump('fitted_models.json',models[selected]);write('representation_candidates.csv',candidates);write('pareto_frontier.csv',ranked);write('retained_set_statistics.csv',[dict(candidate=selected,state_id=r['state_id'],split=r['split'],**{k:r[k] for k in ('known_false','witnesses','modes_separation','diameter','full_recall','retained_recall','volume_proxy','complexity')}) for r in rows if r['candidate']==selected])
 # labels freeze only after fresh validation/erosion choice, but persist current
 # candidate before any neural optimization.
 dump('working_state.json',dict(status='GEOMETRY_SELECTED_PENDING_FRESH_VAL',selected_family=selected,training_required=True,completed=['evidence_aggregation','candidate_fit','pareto_selection'],next_action='freeze 6 deterministic inner points per VAL state and run exact Q64 fresh validation'))
 dump('hypothesis_status.json',[dict(hypothesis_id='BODY_PLUS_EXCLUSIONS',current_status='SELECTED_FOR_BEST_AVAILABLE_LABEL',support='ep0082 sampled exterior intrusion paths; candidate Pareto calculation',failure='not expected to recover complete Basin',next_discriminating_test='fresh retained validation')])
 dump('experiment_ledger.csv',[]) 
 print(json.dumps({'selected':selected,'ranked':ranked},indent=2))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare']);a=ap.parse_args();prepare()
if __name__=='__main__':main()
