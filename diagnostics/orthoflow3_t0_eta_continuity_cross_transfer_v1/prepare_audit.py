#!/usr/bin/env python3
from __future__ import annotations
import csv, json, math, struct
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT=Path('/home/zhihan/research/Basin_C1')
HERE=ROOT/'diagnostics/orthoflow3_t0_eta_continuity_cross_transfer_v1'
SRC=ROOT/'diagnostics/orthoflow3_true_t0_point_learning_v1'
PACT=ROOT/'diagnostics/orthoflow3_t0_pact_training_readiness_v1/exact_q64_manifest.csv'
SCHEMA=ROOT/'diagnostics/stable_oracle_feature_audit/feature_schema_expanded.csv'
HERE.mkdir(parents=True,exist_ok=True)

def write_csv(path, rows, fields=None):
    rows=list(rows)
    if fields is None:
        fields=list(rows[0]) if rows else []
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def ekey(v):
    return b''.join(struct.pack('<d',float(x)) for x in v).hex()

arr=np.load(SRC/'point_learning_arrays.npz',allow_pickle=True)
x=np.asarray(arr['x'],float); eta=np.asarray(arr['physical_target'],float); eta_norm=np.asarray(arr['target'],float)
ids=[str(z) for z in arr['state_ids']]; splits=[str(z) for z in arr['splits']]
norm=json.load(open(SRC/'normalization.json'))
manifest=json.load(open(SRC/'final_source_split.json'))
states={r['state_id']:r for r in manifest['states']}
assert x.shape==(40,214) and len(set(ids))==40
assert len({states[s]['source_group'] for s in ids})==40
targets={r['state_id']:r for r in csv.DictReader(open(SRC/'selected_eta_targets.csv'))}

# All state pairs in exact training-normalized h and frozen normalized eta.
pair=[]
for i in range(40):
  for j in range(i+1,40):
    pair.append({'state_i':ids[i],'state_j':ids[j],'source_i':states[ids[i]]['source_group'],'source_j':states[ids[j]]['source_group'],
      'split_i':splits[i],'split_j':splits[j], 'd_h':float(np.linalg.norm(x[i]-x[j])),
      'd_eta':float(np.linalg.norm(eta_norm[i]-eta_norm[j]))})
dh=np.array([r['d_h'] for r in pair]); de=np.array([r['d_eta'] for r in pair])
q=np.quantile(dh,[.25,.5,.75])
for r in pair:
    r['distance_bin']='nearest_very_local' if r['d_h']<=q[0] else ('local' if r['d_h']<=q[1] else ('medium' if r['d_h']<=q[2] else 'far'))
write_csv(HERE/'normalized_state_distances.csv',pair)

nearest=[]; neighbor_pairs=[]
for i,s in enumerate(ids):
    order=sorted((float(np.linalg.norm(x[i]-x[j])),j) for j in range(40) if j!=i and states[ids[j]]['source_group']!=states[s]['source_group'])
    for rank in (1,2,3,5):
      d,j=order[rank-1]
      row={'state_id':s,'neighbor_rank':rank,'neighbor_state_id':ids[j],'d_h':d,'d_eta':float(np.linalg.norm(eta_norm[i]-eta_norm[j])),
           'state_split':splits[i],'neighbor_split':splits[j]}
      nearest.append(row); neighbor_pairs.append(row)
write_csv(HERE/'nearest_neighbor_pairs.csv',nearest)
one=np.array([r['d_h'] for r in nearest if r['neighbor_rank']==1]); h_jump=float(np.quantile(one,.25))
eta_jump=float(np.quantile([r['d_eta'] for r in nearest],.75))
jumpmap={}
for r in nearest:
    if r['d_h']<=h_jump and r['d_eta']>=eta_jump:
      k=tuple(sorted([r['state_id'],r['neighbor_state_id']]))
      jumpmap[k]={'state_i':k[0],'state_j':k[1],'d_h':r['d_h'],'d_eta':r['d_eta'],'h_threshold':h_jump,'eta_threshold':eta_jump,'status':'TARGET_JUMP_CANDIDATE'}
jumps=sorted(jumpmap.values(),key=lambda r:(r['d_h'],-r['d_eta'],r['state_i'],r['state_j']))
write_csv(HERE/'target_jump_candidates.csv',jumps)
rho_all=spearmanr(dh,de)
rho_local=spearmanr([r['d_h'] for r in nearest],[r['d_eta'] for r in nearest])
cons=[]
for s in ids:
    rr=[r for r in nearest if r['state_id']==s]
    vals=np.array([r['d_eta'] for r in rr],float)
    cons.append({'state_id':s,'nearest_h_distance':next(r['d_h'] for r in rr if r['neighbor_rank']==1),
      'nearest_target_displacement':next(r['d_eta'] for r in rr if r['neighbor_rank']==1),
      'local_5nn_target_displacement_variance':float(np.var(vals))})
cons.append({'state_id':'__GLOBAL__','nearest_h_distance':float(rho_all.statistic),'nearest_target_displacement':float(rho_local.statistic),
             'local_5nn_target_displacement_variance':math.nan})
write_csv(HERE/'target_consistency.csv',cons)

# Exact-Q64 cache: authoritative old-eight aggregate plus raw new-state search records.
cache={}
if PACT.exists():
  for r in csv.DictReader(open(PACT)):
    if r['state_id'] in states and r['Q64_available'].lower()=='true' and int(r['trials'])>=64:
      v=[float(r['eta1']),float(r['eta2']),float(r['eta3'])]
      cache[(r['state_id'],ekey(v))]={'state_id':r['state_id'],'eta':v,'successes':int(r['successes']),'trials':64,
        'deadlock':int(r['deadlock']),'timeout':int(r['timeout']),'collision':int(r['collision']),'provenance':'pact_exact_q64_manifest'}
rawgroups=defaultdict(dict)
for p in sorted((SRC/'state_runs').glob('*/raw/pilot_rollouts.jsonl')):
  for line in open(p):
    if not line.strip(): continue
    r=json.loads(line); sid=r['state_id']; v=r['eta']; rawgroups[(sid,ekey(v))][int(r['future_index'])]=r
for (sid,k),g in rawgroups.items():
  if sid in states and all(z in g for z in range(64)):
    rows=[g[z] for z in range(64)]; cache[(sid,k)]={'state_id':sid,'eta':rows[0]['eta'],'successes':sum(bool(r['success']) for r in rows),'trials':64,
      'deadlock':sum(r['outcome']=='deadlock' for r in rows),'timeout':sum(r['outcome']=='timeout' for r in rows),
      'collision':sum(r['outcome']=='collision' for r in rows),'provenance':'point_label_search_raw'}

cached=[]
for dest in ids:
  for src in ids:
    if src==dest: continue
    k=(dest,ekey(eta[ids.index(src)]))
    if k in cache:
      c=cache[k]; cached.append({'destination_state':dest,'eta_source_state':src,'eta1':eta[ids.index(src),0],'eta2':eta[ids.index(src),1],'eta3':eta[ids.index(src),2],
        'successes':c['successes'],'Q64':c['successes']/64,'B63':c['successes']>=63,'deadlock':c['deadlock'],'timeout':c['timeout'],'collision':c['collision'],'provenance':c['provenance']})
write_csv(HERE/'cached_cross_transfer.csv',cached)

# Freeze exactly 40 symmetric pairs. Prioritize target jumps and unique 1-NN edges,
# then controls across all distance scales. No outcomes enter this selection.
pmap={tuple(sorted([r['state_i'],r['state_j']])):r for r in pair}
selected=[]; seen=set()
def add(k,reason):
    k=tuple(sorted(k))
    if k in seen or len(selected)>=40:return
    seen.add(k); z=dict(pmap[k]);z['selection_reason']=reason;selected.append(z)
for r in jumps:add((r['state_i'],r['state_j']),'target_jump_candidate')
for r in sorted((r for r in nearest if r['neighbor_rank']==1),key=lambda z:(z['d_h'],z['state_id'],z['neighbor_state_id'])):
    add((r['state_id'],r['neighbor_state_id']),'unique_1nn_edge')
# Guarantee medium/far controls first.
for b,n in [('far',5),('medium',5),('local',5)]:
    cand=sorted((r for r in pair if r['distance_bin']==b),key=lambda z:(abs(z['d_h']-np.median([y['d_h'] for y in pair if y['distance_bin']==b])),z['d_eta'],z['state_i'],z['state_j']))
    got=0
    for r in cand:
      before=len(selected); add((r['state_i'],r['state_j']),b+'_control')
      if len(selected)>before:got+=1
      if got>=n:break
for rank in (2,3,5):
  for r in sorted((r for r in nearest if r['neighbor_rank']==rank),key=lambda z:(z['d_h'],z['d_eta'],z['state_id'])):
    add((r['state_id'],r['neighbor_state_id']),f'unique_{rank}nn_edge')
for r in sorted(pair,key=lambda z:(z['d_h'],z['d_eta'],z['state_i'])):add((r['state_i'],r['state_j']),'distance_fill')
assert len(selected)==40

plan=[]; cached_keys={(r['destination_state'],r['eta_source_state']) for r in cached}
for pid,p in enumerate(selected):
  for dest,src in [(p['state_i'],p['state_j']),(p['state_j'],p['state_i'])]:
    idx=ids.index(src); hit=(dest,src) in cached_keys
    plan.append({'pair_id':pid,'destination_state':dest,'eta_source_state':src,'state_id':dest,'eta':eta[idx].tolist(),
      'eta1':eta[idx,0],'eta2':eta[idx,1],'eta3':eta[idx,2],'d_h':p['d_h'],'d_eta':p['d_eta'],'distance_bin':p['distance_bin'],
      'selection_reason':p['selection_reason'],'source_eta_quality_tag':targets[src]['quality_tag'],'cached_exact_q64':hit})
write_csv(HERE/'new_cross_transfer_manifest.csv',plan)

plans=HERE/'plans/cross_transfer64';plans.mkdir(parents=True,exist_ok=True)
missing=[r for r in plan if not r['cached_exact_q64']]
for sh in range(6):
  with open(plans/f'shard{sh}.jsonl','w') as f:
    groups=[r for k,r in enumerate(missing) if k%6==sh]
    for g in groups:
      for future in range(64):
        row={k:(bool(v) if isinstance(v,np.bool_) else (float(v) if isinstance(v,np.floating) else v)) for k,v in g.items() if k not in ('cached_exact_q64',)}
        row.update({'future_index':future,'phase':'cross_transfer64','controller':'neighbor_target'})
        f.write(json.dumps(row,sort_keys=True)+'\n')
summary={'states':40,'pairs_total':780,'selected_symmetric_pairs':40,'directed_tests':80,'cached_directed':sum(r['cached_exact_q64'] for r in plan),
 'new_directed':len(missing),'new_continuations':len(missing)*64,'h_pair_quantiles':{'q25':q[0],'q50':q[1],'q75':q[2]},
 'one_nn_quantiles':{str(z):float(np.quantile(one,z)) for z in [0,.25,.5,.75,1]},'target_jump_thresholds':{'d_h':h_jump,'d_eta':eta_jump},
 'target_jump_count':len(jumps),'spearman_all_pairs':float(rho_all.statistic),'spearman_local_neighbors':float(rho_local.statistic)}
(plans/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(HERE/'protocol.md').write_text(f'''# OrthoFlow3 true-t0 eta continuity and cross-transfer protocol\n\nFrozen inputs: 40 source-isolated true-t0 states from `orthoflow3_true_t0_point_learning_v1`; authoritative OrthoFlow3 SHA256 `51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38`.\n\nDistances use the exact TRAIN-frozen 214-D normalization and frozen normalized eta geometry. Target jumps are frozen at `d_h <= {h_jump:.9g}` (25th percentile of 1-NN distances) and `d_eta >= {eta_jump:.9g}` (75th percentile of rank-1/2/3/5 neighbor displacements). Forty unordered pairs are frozen before rollout and tested symmetrically, totaling at most 80 new directed Q64 transfers after exact cache reuse. No model training or basin reconstruction is permitted.\n''')
print(json.dumps(summary,indent=2))
