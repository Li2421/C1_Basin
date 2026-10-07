#!/usr/bin/env python3
"""Freeze the outcome-blind hard-DB cohort and discrete baseline plan."""
from __future__ import annotations
import csv, hashlib, json, math
from pathlib import Path
from collections import defaultdict
import numpy as np

H=Path(__file__).parent
D=H.parent
PREV=D/'orthoflow3_db_shared_mode_transfer_v1'
POOL=D/'double_bottleneck_initial_state_coverage/data'
ROOT=Path('/home/zhihan/research/Basin_C1')

def sha_bytes(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sha_text(s):return hashlib.sha256(str(s).encode()).hexdigest()
def dump(name,x):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def write(name,rows):
 p=H/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def point_segment_distance(p,a,b):
 d=b-a;t=np.clip(np.dot(p-a,d)/max(np.dot(d,d),1e-30),0,1);return float(np.linalg.norm(p-(a+t*d)))
def line_y_at_zero(p,g):
 t=(0-p[0])/(g[0]-p[0]);return float(p[1]+t*(g[1]-p[1]))
def percentile_high(v):
 # Midrank percentile in [0,1], with larger raw value receiving larger rank.
 order=np.argsort(v,kind='mergesort');r=np.empty(len(v),float);i=0
 while i<len(v):
  j=i+1
  while j<len(v) and v[order[j]]==v[order[i]]:j+=1
  r[order[i:j]]=((i+j-1)/2)/(len(v)-1 if len(v)>1 else 1);i=j
 return r

def main():
 H.mkdir(parents=True,exist_ok=True);(H/'raw').mkdir(exist_ok=True);(H/'logs').mkdir(exist_ok=True);(H/'plans').mkdir(exist_ok=True)
 # Exclude every state/source used to fit, select, test, replicate, or develop
 # the prior transform/selector.  This reads identities only, never outcomes.
 prior=json.load(open(PREV/'db_state_split.json'));fresh=json.load(open(PREV/'fresh_state_manifest.json'))
 used={s['source_group'] for s in prior['states']}|{s['source_group'] for s in fresh['states']}|set(prior['transform_development_source_groups'])
 goals=None;walls=None;candidates=[]
 for pool_name in ('train_pool','validation_pool','untouched_test_pool'):
  base=POOL/pool_name;m=json.load(open(base/'manifest.json'));goals=np.asarray(m['environment']['goals'],float);walls=np.asarray(m['environment']['walls'],float)
  fam={}
  for r in m['files']:fam.setdefault(r['family_id'],r['file'])
  for family,file in sorted(fam.items()):
   if family in used:continue
   z=np.load(base/file,allow_pickle=False);P=np.asarray(z['initial_positions'],float);V=np.asarray(z['initial_velocities'],float)
   assert P.shape==(4,2) and V.shape==(4,2)
   entry=np.r_[np.maximum(-1.85-P[:2,0],0),np.maximum(P[2:,0]-1.85,0)]
   arrival=entry/.5
   y0=np.array([line_y_at_zero(P[i],goals[i]) for i in range(4)])
   cross_y=min(abs(y0[i]-y0[j]) for i in (0,1) for j in (2,3))
   initial_cross_y=min(abs(P[i,1]-P[j,1]) for i in (0,1) for j in (2,3))
   same_side=min(np.linalg.norm(P[0]-P[1]),np.linalg.norm(P[2]-P[3]))
   pair_clear=min(np.linalg.norm(P[i]-P[j]) for i in range(4) for j in range(i+1,4))-.32
   wall_clear=min(point_segment_distance(P[i],a,b) for i in range(4) for a,b in walls)-.16666666666666667
   candidates.append(dict(family_id=family,source_group=family,pool=pool_name,dataset=str(base),episode_file=str(base/file),
      regime=('near_tie' if 'near_tie' in family else 'weakly_asymmetric' if 'weakly_asymmetric' in family else 'clearly_asymmetric'),
      arrival_side_gap=abs(float(np.mean(arrival[:2])-np.mean(arrival[2:]))),arrival_spread=float(np.std(arrival)),
      mean_entry_distance=float(np.mean(entry)),cross_route_y_gap=float(cross_y),initial_cross_y_gap=float(initial_cross_y),
      same_side_pair_distance=float(same_side),initial_pair_clearance=float(pair_clear),initial_wall_clearance=float(wall_clear),
      initial_speed_mean=float(np.mean(np.linalg.norm(V,axis=1)))))
 assert len(candidates)>=48,(len(candidates),'insufficient unused source groups')
 # Equal-weight rank aggregation.  Every raw metric has an explicit monotonic
 # physical interpretation and every component is frozen before outcomes.
 fields=['arrival_side_gap','arrival_spread','mean_entry_distance','cross_route_y_gap','initial_cross_y_gap','same_side_pair_distance','initial_pair_clearance','initial_wall_clearance']
 for f in fields:
  hard=1-percentile_high(np.array([r[f] for r in candidates],float))
  for r,v in zip(candidates,hard):r['rank_hard_'+f]=float(v)
 for r in candidates:r['hard_score']=float(np.mean([r['rank_hard_'+f] for f in fields]));r['tie_hash']=sha_text('db_continuous_generator_hard_v1|'+r['source_group'])
 selected=sorted(candidates,key=lambda r:(-r['hard_score'],r['tie_hash']))[:48]
 states=[]
 for i,r in enumerate(selected):
  states.append(dict(state_id=f'DB_HARD_{i:03d}',split='hard',family_id=r['family_id'],source_group=r['source_group'],dataset=r['dataset'],episode_file=r['episode_file'],regime=r['regime'],hard_rank=i+1,hard_score=r['hard_score'],initial_flow_root=2026093001,future_root=2026093002,rng_namespace=60000+i,h_schema='obs72+current_raw_flow8'))
 assert len(states)==48 and len({s['source_group'] for s in states})==48 and not ({s['source_group'] for s in states}&used)
 rule='''# Outcome-blind hard Double-Bottleneck state rule\n\nThe candidate inventory is every source-unique true-t0 family in the pre-existing DB train/validation/untouched-test pools after excluding all source groups used in the prior transform-development, 64/16/16 selector split, or 32-state fresh replication. No controller or eta outcome is read.\n\nFor each remaining state, only initial positions, initial velocities, frozen goals, bottleneck geometry, agent radius, and wall segments are inspected. Eight difficulty components are computed: smaller left/right mean arrival-time gap to the first bottleneck; smaller four-agent arrival-time spread; smaller mean distance to the first bottleneck; smaller opposite-direction straight-route lateral gap at x=0; smaller initial opposite-direction lateral gap; smaller same-side pair distance; smaller initial pair clearance; and smaller initial wall clearance. Each is converted to an empirical hard-percentile rank over the eligible inventory. The hard score is their unweighted mean. Ties use SHA256(`db_continuous_generator_hard_v1|source_group`). The top 48 are frozen.\n\nThe ranking and manifest are frozen before any new Safety, mode, selector, codebook, or eta-search outcome. The rule intentionally does not use prior success, deadlock, timeout, collision, controller score, or eta evidence.\n'''
 (H/'hard_state_rule.md').write_text(rule)
 dump('hard_state_manifest.json',{'schema':'orthoflow3_db_generator_necessity_hard_states_v1','frozen_before_new_outcomes':True,'selection_rule':'hard_state_rule.md','eligible_unused_states':len(candidates),'selected_states':len(states),'excluded_prior_source_groups':len(used),'states':states})
 write('hard_state_geometry.csv',sorted(candidates,key=lambda r:(-r['hard_score'],r['tie_hash'])))
 # Freeze prior model/codebook assets by content hash.
 tf=json.load(open(PREV/'selected_transform.json'));eta=tf['transform']['eta'];summary=json.load(open(PREV/'test_summary.json'));sel=json.load(open(PREV/'selected_selector.json'))
 frozen={'prior_dir':str(PREV),'transform':tf,'transformed_eta':eta,'best_fixed_mode':summary['train_prior_best_fixed_mode'],'selector':sel,'calibration':json.load(open(PREV/'calibration.json')),'threshold':json.load(open(PREV/'selected_threshold.json')),'feature_normalization':json.load(open(PREV/'feature_normalization.json')),'asset_hashes':{x:sha_bytes(PREV/x) for x in ('selected_transform.json','selected_selector.json','calibration.json','selected_threshold.json','feature_normalization.json',sel['checkpoint'])}}
 dump('frozen_assets.json',frozen)
 # Discrete plan: Safety plus every transformed mode, exact Q64.  Fixed,
 # selector, and oracle are all exact readouts of this one matched matrix.
 tasks=[]
 for s in states:
  for fi in range(64):tasks.append(dict(state_id=s['state_id'],eta=[0.,0.,0.],future_index=fi,phase='hard_discrete_q64',probe_id=f"{s['state_id']}_safety_{fi}",controller='safety',mode_id=-1))
  for m,e in enumerate(eta):
   for fi in range(64):tasks.append(dict(state_id=s['state_id'],eta=e,future_index=fi,phase='hard_discrete_q64',probe_id=f"{s['state_id']}_mode{m}_{fi}",controller='transformed',mode_id=m))
 groups=defaultdict(list)
 for t in tasks:groups[(t['controller'],t['mode_id'])].append(t)
 bins=[[] for _ in range(6)]
 # Split large constant-eta groups across shards only as needed for balance.
 for g,v in sorted(groups.items(),key=lambda x:(-len(x[1]),x[0])):
  for t in sorted(v,key=lambda x:(x['state_id'],x['future_index'])):bins[min(range(6),key=lambda j:(len(bins[j]),j))].append(t)
 out=H/'plans'/'discrete';out.mkdir(parents=True,exist_ok=True)
 for j,b in enumerate(bins):
  with open(out/f'shard{j}.jsonl','w') as f:
   for t in b:f.write(json.dumps(t)+'\n')
 dump('discrete_cost_estimate.json',{'states':48,'controllers_or_modes_per_state':13,'trials':64,'new_continuations_before_cache_reuse':len(tasks),'shard_loads':[len(b) for b in bins],'estimated_wall_hours_at_prior_empirical_aggregate_rate':len(tasks)/(6*0.65*3600),'scientific_decision':'establish frozen fixed/selector/codebook upper bound before any continuous search'})
 dump('working_state.json',{'status':'DISCRETE_READY','completed':['hard_rule_freeze','hard_manifest_freeze','frozen_asset_hashes','discrete_plan'],'next_action':'run exact matched discrete matrix'})
 dump('evidence_index.json',{'prior_final_decision':str(PREV/'final_decision.json'),'prior_transformed_modes':str(PREV/'selected_transform.json'),'prior_selector':sel['checkpoint'],'hard_inventory':'hard_state_geometry.csv','hard_manifest':'hard_state_manifest.json'})
 dump('hypothesis_status.json',{'H_continuous_needed':{'status':'UNTESTED','unresolved':'Whether any hard state lacks all 12 transformed B63 modes but has another B63 eta'},'H_fixed_codebook_sufficient':{'status':'UNTESTED','unresolved':'Hard-cohort codebook oracle coverage'},'H_hard_is_sparser':{'status':'UNTESTED','unresolved':'Feasible modes and overlap versus ordinary DB/Toy'}})
 write('experiment_ledger.csv',[{'stage':'freeze_hard_cohort','status':'COMPLETE','new_continuations':0,'decision':f'{len(states)} of {len(candidates)} unused outcome-blind candidates'},{'stage':'freeze_discrete_plan','status':'COMPLETE','new_continuations':0,'decision':f'{len(tasks)} planned before cache reuse'}])
 print(json.dumps({'eligible':len(candidates),'selected':len(states),'regimes':{q:sum(s['regime']==q for s in states) for q in sorted({s['regime'] for s in states})},'score_range':[min(s['hard_score'] for s in states),max(s['hard_score'] for s in states)],'tasks':len(tasks),'loads':[len(b) for b in bins]},indent=2))

if __name__=='__main__':main()
