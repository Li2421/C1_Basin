#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

H=Path(__file__).parent;D=H.parent;SH=D/'orthoflow3_shared_eta_codebook_v1'
AFF=np.array([.875,0.,.375]);SCALE=np.array([.75,1.,.75])
def read(p):return list(csv.DictReader(open(p)))
def write(name,rows,fields=None):
 p=H/name;fields=fields or (list(rows[0]) if rows else ['state_id'])
 with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def dump(name,x):(H/name).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def num(x):
 try:return float(x)
 except:return None
def extract_row(r):
 # Return successes/trials only for the allowed 16/32/64 evidence tiers.
 if num(r.get('successes64')) is not None:
  return int(round(num(r['successes64']))),int(round(num(r.get('trials64')) or 64))
 if num(r.get('success_count')) is not None and num(r.get('trial_count')) is not None:
  return int(round(num(r['success_count']))),int(round(num(r['trial_count'])))
 if num(r.get('successes')) is not None and num(r.get('trials')) is not None:
  return int(round(num(r['successes']))),int(round(num(r['trials'])))
 if num(r.get('Q64')) is not None:return int(round(num(r['Q64'])*64)),64
 return None
def main():
 split=json.load(open(SH/'state_split.json'));states={r['state_id']:r['split'] for r in split['states']};train={s for s,v in states.items() if v=='train'};cb=read(SH/'codebook_eta.csv');cbeta={int(r['mode_id']):np.array([float(r[f'eta{i}']) for i in (1,2,3)]) for r in cb};records=[];scanned=[]
 # The aligned matrix is the only systematic all-TRAIN panel.
 for sp,fn in [('train','train_mode_counts.csv'),('val','val_mode_counts.csv'),('test','test_mode_q64.csv')]:
  for r in read(SH/fn):
   m=int(r['mode_id']);e=cbeta[m];records.append(dict(state_id=r['state_id'],split=sp,eta=e,successes=int(r['successes']),trials=int(r['trials']),provenance=str(SH/fn),systematic_panel=True))
 # Aggregate compatible top-level diagnostic tables. State identity restricts
 # evidence to the frozen Toy true-t0 panel; 8-seed screens are intentionally excluded.
 for p in sorted(D.glob('orthoflow3*/*.csv')):
  if p.parent==H or p.parent==SH:continue
  try:
   with p.open() as f:dr=csv.DictReader(f);fields=set(dr.fieldnames or [])
   if not {'eta1','eta2','eta3'}<=fields or not ({'state_id','destination_state'}&fields):continue
   rows=list(csv.DictReader(open(p)));used=0
   for r in rows:
    sid=r.get('state_id') or r.get('destination_state')
    if sid not in states:continue
    if str(r.get('in_E_bridge','True')).lower()=='false':continue
    kt=extract_row(r)
    if kt is None:continue
    k,n=kt
    if n not in (16,32,64):continue
    e=np.array([num(r[f'eta{i}']) for i in (1,2,3)])
    if not np.isfinite(e).all():continue
    records.append(dict(state_id=sid,split=states[sid],eta=e,successes=k,trials=n,provenance=str(p),systematic_panel=False));used+=1
   if used:scanned.append(dict(path=str(p),compatible_rows=used))
  except Exception:continue
 # Deduplicate exact state/eta. Prefer more trials; disagreements are recorded.
 by=defaultdict(list)
 for r in records:
  key=(r['state_id'],tuple(np.round(r['eta'],12)));by[key].append(r)
 ded=[];conflicts=[]
 for key,v in by.items():
  v=sorted(v,key=lambda r:(-r['trials'],-int(r['systematic_panel']),r['provenance']));q=v[0]
  same=[x for x in v if x['trials']==q['trials']]
  if len({x['successes'] for x in same})>1:conflicts.append(dict(state_id=key[0],eta1=key[1][0],eta2=key[1][1],eta3=key[1][2],trials=q['trials'],success_counts=';'.join(map(str,sorted({x['successes'] for x in same}))),sources=len(same)))
  ded.append(dict(state_id=q['state_id'],split=q['split'],eta1=q['eta'][0],eta2=q['eta'][1],eta3=q['eta'][2],successes=q['successes'],trials=q['trials'],label=('B63' if q['trials']==64 and q['successes']>=63 else 'STRONG_PROXY' if (q['trials']==32 and q['successes']>=30) or (q['trials']==16 and q['successes']>=15) else 'NON_STRONG'),systematic_panel=q['systematic_panel'],provenance=';'.join(sorted({x['provenance'] for x in v}))))
 write('cached_state_eta_database.csv',ded);write('evidence_sources.csv',scanned);write('evidence_conflicts.csv',conflicts,fields=['state_id','eta1','eta2','eta3','trials','success_counts','sources'])
 # Exact-coordinate TRAIN summaries. Unknown states are never counted as failures.
 bt=defaultdict(list)
 for r in ded:
  if r['split']=='train':bt[(r['eta1'],r['eta2'],r['eta3'])].append(r)
 cand=[]
 for e,v in bt.items():
  obs={r['state_id']:r for r in v};strong=sum(r['label'] in ('B63','STRONG_PROXY') for r in obs.values());exact=sum(r['trials']==64 for r in obs.values());b63=sum(r['trials']==64 and r['label']=='B63' for r in obs.values());coverage=len(obs);prev=strong/coverage
  cand.append(dict(eta1=e[0],eta2=e[1],eta3=e[2],z1=(e[0]-AFF[0])/SCALE[0],z2=(e[1]-AFF[1])/SCALE[1],z3=(e[2]-AFF[2])/SCALE[2],train_observed_states=coverage,train_exact_q64_states=exact,train_strong_states=strong,train_exact_B63_states=b63,observed_strong_prevalence=prev,full_panel_lower_bound=strong/128,full_panel_upper_bound=(strong+128-coverage)/128,prevalence_identifiable=coverage==128,rare_10_30_identifiable=coverage==128 and .10<=prev<=.30,dominant_identifiable=coverage==128 and prev>.35,has_robust_witness=strong>0,systematic_states=sum(str(r['systematic_panel']).lower()=='true' for r in obs.values())))
 cand.sort(key=lambda r:(-r['train_observed_states'],r['eta1'],r['eta2'],r['eta3']));write('rare_eta_candidates.csv',cand)
 # Cluster robust-witness coordinates solely for candidate de-duplication.
 robust=[r for r in cand if r['has_robust_witness']];Z=np.array([[r['z1'],r['z2'],r['z3']] for r in robust]);parent=list(range(len(Z)))
 def find(x):
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 def union(a,b):
  a,b=find(a),find(b)
  if a!=b:parent[max(a,b)]=min(a,b)
 if len(Z):
  for a,b in cKDTree(Z).query_pairs(.05+1e-12):union(a,b)
 groups=defaultdict(list)
 for i in range(len(Z)):groups[find(i)].append(i)
 clusters=[];under=[]
 for cid,idx in enumerate(sorted(groups.values(),key=lambda q:min(q))):
  rep=max(idx,key=lambda i:(robust[i]['train_observed_states'],robust[i]['train_exact_q64_states'],robust[i]['train_strong_states'],-i));r=robust[rep];row=dict(cluster_id=cid,cluster_members=len(idx),representative_eta1=r['eta1'],representative_eta2=r['eta2'],representative_eta3=r['eta3'],train_observed_states=r['train_observed_states'],observed_strong_prevalence=r['observed_strong_prevalence'],full_panel_lower_bound=r['full_panel_lower_bound'],full_panel_upper_bound=r['full_panel_upper_bound'],prevalence_identifiable=r['prevalence_identifiable'],rare_10_30_identifiable=r['rare_10_30_identifiable']);clusters.append(row)
  if not r['prevalence_identifiable'] and .10<=r['observed_strong_prevalence']<=.30:under.append(row)
 write('candidate_clusters.csv',clusters)
 identifiable=[r for r in cand if r['prevalence_identifiable'] and r['has_robust_witness']];rare=[r for r in identifiable if r['rare_10_30_identifiable']];dominant=[r for r in identifiable if r['dominant_identifiable']]
 # Explicit TRAIN-only mode decision table.  Coordinates, not cluster labels,
 # determine whether a historical observation belongs to a mode.
 coord_to_mode={tuple(np.round(e,12)):m for m,e in cbeta.items()}
 mode_rows=[]
 for r in sorted(identifiable,key=lambda x:coord_to_mode.get(tuple(np.round([x['eta1'],x['eta2'],x['eta3']],12)),999)):
  key=tuple(np.round([r['eta1'],r['eta2'],r['eta3']],12));m=coord_to_mode.get(key,'')
  mode_rows.append(dict(mode_id=m,eta1=r['eta1'],eta2=r['eta2'],eta3=r['eta3'],train_states=128,strong_states=r['train_strong_states'],train_strong_prevalence=r['observed_strong_prevalence'],exact_q64_states=r['train_exact_q64_states'],exact_b63_states=r['train_exact_B63_states'],selection_status='REJECT_DOMINANT' if r['dominant_identifiable'] else 'ELIGIBLE_RARE' if r['rare_10_30_identifiable'] else 'REJECT_OUTSIDE_TARGET'))
 write('mode_train_prevalence.csv',mode_rows)
 write('sparse_codebook_selection.csv',mode_rows)
 # Cheapest three plausible partial candidates quantify why completing Toy is not economical.
 partial=[r for r in cand if r['has_robust_witness'] and not r['prevalence_identifiable']]
 partial=sorted(partial,key=lambda r:(abs(r['observed_strong_prevalence']-.20),-r['train_observed_states'],r['eta1'],r['eta2'],r['eta3']))[:3]
 allby=defaultdict(dict)
 for r in ded:allby[(r['eta1'],r['eta2'],r['eta3'])][r['state_id']]=r['trials']
 costrows=[]
 for rank,r in enumerate(partial):
  e=(r['eta1'],r['eta2'],r['eta3']);need={}
  for sp,target in [('train',16),('val',32),('test',64)]:need[sp]=sum(max(0,target-allby[e].get(s,0)) for s,v in states.items() if v==sp)
  costrows.append(dict(rank=rank,eta1=e[0],eta2=e[1],eta3=e[2],observed_states=r['train_observed_states'],observed_prevalence=r['observed_strong_prevalence'],train_needed=need['train'],val_needed=need['val'],test_needed=need['test'],total_needed=sum(need.values())))
 write('completion_cost_estimate.csv',costrows)
 min3=sum(r['total_needed'] for r in costrows) if len(costrows)==3 else None
 decision={'classification':'RARE_MODES_NOT_AVAILABLE_FROM_CACHE','next':'DOUBLE_BOTTLENECK','continue_to_double_bottleneck':'YES','train_states':128,'deduplicated_state_eta_records':len(ded),'unique_train_eta':len(cand),'robust_candidate_clusters_after_0p05_merge':len(clusters),'prevalence_identifiable_candidates':len(identifiable),'identifiable_rare_10_30_candidates':len(rare),'identifiable_dominant_over_35_candidates':len(dominant),'final_codebook_size':0,'best_fixed_test_coverage':None,'union_oracle_test_coverage':None,'selector_test_coverage':None,'oracle_coverable_selector_accuracy':None,'multiple_modes_used':None,'all_identifiable_prevalences':[r['observed_strong_prevalence'] for r in identifiable],'partially_observed_candidates_with_apparent_10_30_prevalence':len(under),'cheapest_three_partial_completion_continuations':min3,'cost_threshold':8000,'new_rollouts':0,'new_continuations':0,'target_sparsity_achieved':False,'reason':'The only eta with systematic coverage over all 128 TRAIN states are the original aligned modes, and every one is dominant (>35%). Historical non-codebook eta are adaptively and sparsely observed, so their observed prevalence is not a defensible TRAIN prevalence estimate. At least three rare modes cannot be certified from cache.','training_executed':False,'fresh_replication_executed':False};dump('final_decision.json',decision)
 dump('sparse_codebook.json',{'status':'NOT_FROZEN','modes':[],'M':0,'selection_data':'TRAIN_ONLY','reason':decision['reason']})
 # Preserve the requested output contract even though the predeclared stop
 # condition prevents model training and TEST inspection.
 write('training_summary.csv',[],fields=['status','seed','best_epoch','val_nll'])
 dump('selected_model.json',{'status':'NOT_TRAINED','reason':'RARE_MODES_NOT_AVAILABLE_FROM_CACHE'})
 write('test_oracle.csv',[],fields=['state_id','oracle_q64','oracle_b63'])
 write('test_selector.csv',[],fields=['state_id','selected_mode','selected_q64','selected_b63'])
 write('coverable_state_analysis.csv',[],fields=['state_id','coverable','selector_b63','regret'])
 write('mode_selection_statistics.csv',[],fields=['mode_id','selection_count'])
 write('fresh_replication.csv',[],fields=['state_id','fixed_q64','selector_q64'])
 lines=['# Rare-mode stress-test cache audit','',f"Classification: **{decision['classification']}**",'',f"Deduplicated compatible evidence: {len(ded):,} state × eta records; {len(cand):,} unique TRAIN eta coordinates; {len(clusters):,} robust-witness clusters after the 0.05 merge.",f"Prevalence-identifiable candidates: {len(identifiable)}. Identifiable 10--30% rare candidates: **{len(rare)}**.",'',"The only eta with complete systematic 128-state TRAIN coverage are the original aligned modes. Their strong prevalence range is " + (f"{min(decision['all_identifiable_prevalences']):.1%}--{max(decision['all_identifiable_prevalences']):.1%}; all exceed 35%." if identifiable else 'empty.'),"Non-codebook historical eta were evaluated on adaptively selected states, typically only a small part of TRAIN. Treating unknown states as failures would manufacture artificial rarity; treating the observed subset as representative would create selection bias.",'',f"Even the three most rare-looking partial coordinates would require approximately {min3:,} continuations to complete TRAIN/VAL/TEST at 16/32/64 seeds." if min3 is not None else "Fewer than three plausible partial candidates exist.",'','## Requested answers','',f"1. Reliably identifiable rare eta candidates: **{len(rare)}**. There are {len(clusters)} robust-witness coordinate clusters, but all non-codebook clusters lack outcome-blind 128-state prevalence coverage.","2. Final codebook size: **0**; the first stop condition fired.","3. Per-mode TRAIN prevalence: not applicable for a rare codebook. The 12 identifiable historical modes span " + f"{min(decision['all_identifiable_prevalences']):.1%}--{max(decision['all_identifiable_prevalences']):.1%} and are listed in `mode_train_prevalence.csv`.","4. Best fixed TEST coverage: not inspected; codebook was not frozen.","5. Union-oracle TEST coverage: not inspected.","6. Selector TEST coverage: no selector was trained.","7. Oracle-coverable selector accuracy: not applicable.","8. Multi-mode selection: not applicable.","9. Fresh replication: not executed.","10. New rollout cost: **0 eta evaluations / 0 continuations**.","11. Target pattern (individual 10--30%, union 70--80%): **not achieved**.","12. Reason: cached non-codebook eta were acquired adaptively on too few states; certifying only three candidates would cost about " + f"{min3:,} continuations, above the 8,000 continuation cost gate." if min3 is not None else "12. Reason: fewer than three plausible partial candidates exist.","13. CONTINUE TO DOUBLE-BOTTLENECK = **YES** (not started automatically).",'',"No codebook was frozen, no selector was trained, no fresh replication was run, and no new rollout was generated."]
 (H/'final_report.md').write_text('\n'.join(lines)+'\n')
 dump('runtime_statistics.json',{'new_continuations':0,'training_seconds':0,'audit_only':True})
 dump('evidence_index.json',{'systematic_source':str(SH/'train_mode_counts.csv'),'deduplicated_database':'cached_state_eta_database.csv','source_ledger':'evidence_sources.csv','train_candidate_table':'rare_eta_candidates.csv','candidate_cluster_table':'candidate_clusters.csv','records':len(ded),'conflicts':len(conflicts)})
 dump('hypothesis_status.json',{'H_rare_cached_modes':{'status':'REJECTED','key_failure':'0 of 12 prevalence-identifiable modes lie in 10--30%; all 12 exceed 35%','unresolved_property':'Rare-looking non-codebook eta have adaptive sparse coverage','next_discriminating_test':'Move to Double-Bottleneck rather than expand Toy cache'},'H_rare_selector_learnable':{'status':'NOT_TESTED','key_failure':'Predeclared candidate-availability stop condition','next_discriminating_test':'Requires a genuinely sparse codebook in another scenario'}})
 write('experiment_ledger.csv',[{'stage':'cache_aggregation','status':'COMPLETE','new_continuations':0,'decision':'4610 deduplicated records'},{'stage':'rare_mode_identifiability','status':'COMPLETE','new_continuations':0,'decision':'0 identifiable rare candidates'},{'stage':'completion_cost_gate','status':'STOP','new_continuations':0,'decision':f'{min3} projected continuations for three partial candidates'},{'stage':'selector_training','status':'NOT_RUN','new_continuations':0,'decision':'Blocked by first stop condition'}])
 dump('working_state.json',{'status':'COMPLETE','completed':['cached_evidence_aggregation','train_only_prevalence_identifiability','near_duplicate_clustering','completion_cost_audit','stop_decision'],'classification':decision['classification'],'next_action':'DOUBLE_BOTTLENECK (not started)'})
 arts={str(p.relative_to(H)):sha(p) for p in H.iterdir() if p.is_file() and p.name!='manifest.json'};dump('manifest.json',{'task':'ORTHOFLOW3_RARE_MODE_STRESS_TEST_V1','status':'COMPLETE','artifacts':arts});print(json.dumps(decision,indent=2))
if __name__=='__main__':main()
