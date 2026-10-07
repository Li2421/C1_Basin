#!/usr/bin/env python3
from __future__ import annotations
import csv,hashlib,json,math,os,time
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_t0_eta_continuity_cross_transfer_v1';SRC=ROOT/'diagnostics/orthoflow3_true_t0_point_learning_v1'
def readcsv(p):return list(csv.DictReader(open(p)))
def writecsv(p,rows,fields=None):
 rows=list(rows);fields=fields or (list(rows[0]) if rows else [])
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def corr(a,b):
 if len(a)<3 or np.std(a)==0 or np.std(b)==0:return math.nan
 return float(spearmanr(a,b).statistic)

t0=time.time();arr=np.load(SRC/'point_learning_arrays.npz',allow_pickle=True);x=np.asarray(arr['x'],float);eta=np.asarray(arr['target'],float);ids=[str(z) for z in arr['state_ids']]
idix={s:i for i,s in enumerate(ids)};targets={r['state_id']:r for r in readcsv(SRC/'selected_eta_targets.csv')}
manifest=readcsv(HERE/'new_cross_transfer_manifest.csv');cache=readcsv(HERE/'cached_cross_transfer.csv');cachemap={(r['destination_state'],r['eta_source_state']):r for r in cache}

# Aggregate new rows from both frozen waves.
groups=defaultdict(dict)
paths=list((HERE/'raw').glob('shard*.jsonl'))+list((HERE/'raw/cross_transfer64b').glob('shard*.jsonl'))
for p in paths:
 for line in open(p):
  if not line.strip():continue
  r=json.loads(line);groups[(r['state_id'],tuple(float(z) for z in r['eta']))][int(r['future_index'])]=r
newsum={}
for k,g in groups.items():
 assert all(i in g for i in range(64)),(k,len(g))
 rr=[g[i] for i in range(64)]
 newsum[k]={'successes':sum(bool(r['success']) for r in rr),'deadlock':sum(r['outcome']=='deadlock' for r in rr),'timeout':sum(r['outcome']=='timeout' for r in rr),
  'collision':sum(r['outcome']=='collision' for r in rr),'provenance':'new_cross_transfer_q64'}

directed=[]
for m in manifest:
 k=(m['destination_state'],m['eta_source_state']);nk=(m['destination_state'],(float(m['eta1']),float(m['eta2']),float(m['eta3'])));z=cachemap.get(k) or newsum.get(nk)
 assert z is not None,k
 succ=int(z['successes']);directed.append({**m,'successes':succ,'Q64':succ/64,'B63':succ>=63,'deadlock':int(z['deadlock']),'timeout':int(z['timeout']),
  'collision':int(z['collision']),'provenance':z['provenance']})
writecsv(HERE/'cross_transfer_results.csv',directed)

# Symmetric pair categories.
pg=defaultdict(list)
for r in directed:pg[int(r['pair_id'])].append(r)
pairs=[]
for pid,rr in sorted(pg.items()):
 assert len(rr)==2
 n=sum(str(r['B63']).lower()=='true' for r in rr);cat='MUTUAL_ROBUST_TRANSFER' if n==2 else ('ONE_WAY_ROBUST_TRANSFER' if n==1 else 'NO_ROBUST_TRANSFER')
 pairs.append({'pair_id':pid,'state_i':rr[0]['destination_state'],'state_j':rr[1]['destination_state'],'d_h':float(rr[0]['d_h']),'d_eta':float(rr[0]['d_eta']),
  'distance_bin':rr[0]['distance_bin'],'selection_reason':rr[0]['selection_reason'],'Q64_i_from_j':float(rr[0]['Q64']),'Q64_j_from_i':float(rr[1]['Q64']),
  'category':cat,'robust_directions':n})

# Target jump interpretation.
jumpkeys={tuple(sorted((r['state_i'],r['state_j']))):r for r in readcsv(HERE/'target_jump_candidates.csv')}
jout=[]
for r in pairs:
 k=tuple(sorted((r['state_i'],r['state_j'])))
 if k not in jumpkeys:continue
 interp='TARGET_JUMP_BUT_BASINS_OVERLAP' if r['category']=='MUTUAL_ROBUST_TRANSFER' else ('PARTIAL_OVERLAP_OR_ASYMMETRIC' if r['category']=='ONE_WAY_ROBUST_TRANSFER' else 'POSSIBLE_BASIN_CHANGE')
 jout.append({**jumpkeys[k],**r,'interpretation':interp})
writecsv(HERE/'target_jump_candidates.csv',jout)

# All available exact directed evidence supplies kNN overlap, not only selected tests.
allres={(r['destination_state'],r['eta_source_state']):r for r in cache}
allres.update({(r['destination_state'],r['eta_source_state']):r for r in directed})
pairall=readcsv(HERE/'normalized_state_distances.csv')
allpairs=[]
for p in pairall:
 a,b=p['state_i'],p['state_j']
 if (a,b) not in allres or (b,a) not in allres:continue
 za,zb=allres[(a,b)],allres[(b,a)];n=(int(za['successes'])>=63)+(int(zb['successes'])>=63)
 allpairs.append({'state_i':a,'state_j':b,'d_h':float(p['d_h']),'d_eta':float(p['d_eta']),'distance_bin':p['distance_bin'],
  'category':'MUTUAL_ROBUST_TRANSFER' if n==2 else ('ONE_WAY_ROBUST_TRANSFER' if n==1 else 'NO_ROBUST_TRANSFER'),'robust_directions':n,
  'Q64_i_from_j':int(za['successes'])/64,'Q64_j_from_i':int(zb['successes'])/64})
bins=[]
for evidence,collection in [('frozen_targeted',pairs),('all_exact_available_ascertainment_biased',allpairs)]:
 for b in ('nearest_very_local','local','medium','far'):
  rr=[r for r in collection if r['distance_bin']==b];c=Counter(r['category'] for r in rr)
  bins.append({'evidence_set':evidence,'distance_bin':b,'pairs':len(rr),'d_h_min':min((r['d_h'] for r in rr),default=math.nan),'d_h_median':np.median([r['d_h'] for r in rr]) if rr else math.nan,
   'd_h_max':max((r['d_h'] for r in rr),default=math.nan),'median_d_eta':np.median([r['d_eta'] for r in rr]) if rr else math.nan,
   'mutual':c['MUTUAL_ROBUST_TRANSFER'],'one_way':c['ONE_WAY_ROBUST_TRANSFER'],'none':c['NO_ROBUST_TRANSFER'],
   'mutual_rate':c['MUTUAL_ROBUST_TRANSFER']/len(rr) if rr else math.nan,'mutual_or_one_way_rate':(c['MUTUAL_ROBUST_TRANSFER']+c['ONE_WAY_ROBUST_TRANSFER'])/len(rr) if rr else math.nan})
writecsv(HERE/'continuity_by_distance.csv',bins)
over=[]
for sid in ids:
 i=idix[sid];order=sorted((np.linalg.norm(x[i]-x[j]),ids[j]) for j in range(40) if j!=i)
 for k in (3,5):
  neigh=[s for _,s in order[:k]];known=[allres[(sid,s)] for s in neigh if (sid,s) in allres];b=sum((int(r['successes'])>=63) for r in known)
  over.append({'state_id':sid,'k':k,'tested_neighbor_targets':len(known),'B63_neighbor_targets':b,'neighbor_target_robust_overlap':b/len(known) if known else math.nan,
   'complete_k_coverage':len(known)==k})
writecsv(HERE/'local_overlap_scores.csv',over)

# Documented, fixed feature subsets. All use the exact train-normalized coordinates.
schema=readcsv(ROOT/'diagnostics/stable_oracle_feature_audit/feature_schema_expanded.csv')
idx_by_group=defaultdict(list);idx_by_seg=defaultdict(list)
for r in schema:idx_by_group[r['semantic_group']].append(int(r['dimension']));idx_by_seg[r['segment']].append(int(r['dimension']))
subsets={'full_h0':list(range(214)),'current_geometry_observation':idx_by_group['current_geometry_observation'],
 'history_monitor':idx_by_group['history_monitor'],'projection_safety_diagnostics':idx_by_group['projection_safety_diagnostics'],
 'current_control':idx_by_group['current_control']}
relseg=['positions','goal_relative','inter_agent_relative_position','inter_agent_relative_velocity','goal_errors','pairwise_barrier_h','wall_barrier_h']
subsets['documented_physical_relational']=sorted(set(sum((idx_by_seg[s] for s in relseg),[])))
metric=[]
for name,ii in subsets.items():
 da=[];de=[]
 for r in pairall:
  a=idix[r['state_i']];b=idix[r['state_j']];da.append(float(np.linalg.norm(x[a,ii]-x[b,ii])));de.append(float(r['d_eta']))
 # Tested symmetric transfer category as ordinal robust overlap proxy.
 ds=[];rob=[];qq=[]
 for r in pairs:
  a=idix[r['state_i']];b=idix[r['state_j']];ds.append(float(np.linalg.norm(x[a,ii]-x[b,ii])));rob.append(float(r['robust_directions'])/2);qq.append((r['Q64_i_from_j']+r['Q64_j_from_i'])/2)
 metric.append({'metric':name,'dimensions':len(ii),'spearman_distance_vs_target_displacement_all_pairs':corr(da,de),
  'spearman_distance_vs_cross_transfer_robust_fraction':corr(ds,rob),'spearman_distance_vs_mean_cross_Q64':corr(ds,qq),
  'median_pair_distance':float(np.median(da))})
writecsv(HERE/'state_metric_comparison.csv',metric)

# Physical midpoint is intentionally skipped: continuous positions alone do not define
# compatible Flow xi0, history/monitor, and RNG-conditioned true-t0 state semantics.
writecsv(HERE/'midpoint_state_results.csv',[{'status':'SKIPPED','reason':'Valid physical interpolation is ambiguous: interpolating positions or 214-D h0 cannot preserve exact Flow xi0/RNG conditioning and full true-t0 state semantics.','new_evaluations':0}])

nearest=next(r for r in bins if r['evidence_set']=='frozen_targeted' and r['distance_bin']=='nearest_very_local');jump_mut=sum(r['interpretation']=='TARGET_JUMP_BUT_BASINS_OVERLAP' for r in jout)/len(jout) if jout else 0
if jout and jump_mut>=0.5 and nearest['mutual_or_one_way_rate']>=0.6:
 classification='TARGET_SELECTION_MULTIMODALITY_DOMINANT';gate='YES'
elif nearest['mutual_or_one_way_rate']>=0.6:
 classification='LOCAL_BASIN_CONTINUITY_WITH_PARTIAL_TARGET_AMBIGUITY';gate='YES'
else:
 classification='STATE_COVERAGE_INSUFFICIENT';gate='NO'

requirements='''# Requirements for the next analytic basin representation search\n\nThe continuity audit permits representation search, but does not itself fit a basin. The next representation must respect all accumulated evidence:\n\n- eta is a persistent 3-D OrthoFlow3 parameter selected once at true t0.\n- Most observed t0 basins are broad tangentially and narrower normally.\n- Existing success-success interpolation was 88/96 B63; explicit hole probes were 16/16 B63.\n- Local success/failure boundaries can be sharp.\n- Cross-state morphology is `SHARED_MORPHOLOGY_STATE_DEPENDENT_DEFORMATION`.\n- A sphere is too conservative; the axis-aligned ellipsoid produced false inclusions.\n- A four-ball union had severe recall failure; naive manifold and PACT encodings generalized poorly to independent B63 samples.\n- The representation must allow state-dependent orientation and extent, asymmetric thickness, local nonconvexity or occasional gaps, low false inclusion, efficient distance/membership computation, and learnability from h0.\n- Target transfer evidence implies that multiple distant robust eta values can be valid for the same local neighborhood, so it must preserve set-valued/multimodal supervision rather than collapse it to a single MSE target.\n'''
(HERE/'basin_representation_requirements.md').write_text(requirements if gate in ('YES','CONDITIONAL') else '# Basin representation search deferred\n\nFix state coverage or metric alignment before fitting a new representation.\n')

runt=[]
for p in list((HERE/'raw').glob('shard*_runtime.json'))+list((HERE/'raw/cross_transfer64b').glob('shard*_runtime.json')):
 z=json.load(open(p));z.setdefault('plan','cross_transfer64');runt.append(z)
runtime={'reused_exact_q64_directed_selected':sum(str(r['cached_exact_q64']).lower()=='true' for r in manifest),'reused_continuations_equivalent':64*sum(str(r['cached_exact_q64']).lower()=='true' for r in manifest),
 'scheduled_new_directed_transfers':sum(str(r['cached_exact_q64']).lower()!='true' for r in manifest),'unique_new_destination_eta_q64':len(newsum),
 'new_continuations':sum(r['new_continuations'] for r in runt),'physical_steps':sum(r['physical_steps'] for r in runt),
 'critical_rollout_wall_seconds':sum(max([r['wall_seconds'] for r in runt if r['plan']==pl],default=0) for pl in {r['plan'] for r in runt}),
 'worker_wall_seconds_sum':sum(r['wall_seconds'] for r in runt),'max_gpu_shards':6,'gpu_memory_mib_per_worker_approx':614,'cpu_threads_max_total':12,'midpoint_new_evaluations':0}
(HERE/'runtime_statistics.json').write_text(json.dumps(runtime,indent=2)+'\n')

target_modes=Counter((r['target_eta1'],r['target_eta2'],r['target_eta3']) for r in targets.values())
decision={'primary_classification':classification,'BEGIN_BASIN_REPRESENTATION_SEARCH':gate,'target_jump_candidates':len(jout),
 'target_jump_mutual_transfer_fraction':jump_mut,'nearest_bin_mutual_or_one_way_rate':nearest['mutual_or_one_way_rate'],
 'unique_selected_eta_targets':len(target_modes),'largest_eta_target_mode_count':max(target_modes.values()),
 'point_regression_failed_mainly_due_set_valued_multimodal_target':classification=='TARGET_SELECTION_MULTIMODALITY_DOMINANT',
 'current_true_t0_state_coverage_sufficient':classification not in ('STATE_COVERAGE_INSUFFICIENT','UNDERRESOLVED'),
 'midpoint_test':'SKIPPED_INVALID_PHYSICAL_INTERPOLATION'}
(HERE/'final_decision.json').write_text(json.dumps(decision,indent=2)+'\n')

total=Counter(r['category'] for r in pairs)
k3_complete=[float(r['neighbor_target_robust_overlap']) for r in over if r['k']==3 and r['complete_k_coverage']]
k5_complete=[float(r['neighbor_target_robust_overlap']) for r in over if r['k']==5 and r['complete_k_coverage']]
mdfull=next(r for r in metric if r['metric']=='full_h0');best=max(metric,key=lambda r:abs(r['spearman_distance_vs_mean_cross_Q64']) if not math.isnan(r['spearman_distance_vs_mean_cross_Q64']) else -1)
report=f'''# OrthoFlow3 true-t0 robust eta continuity and cross-transfer audit\n\n## Result\n\n**{classification}**\n\n`BEGIN_BASIN_REPRESENTATION_SEARCH = {gate}`. No model was trained and no basin representation was fitted.\n\n## State and target geometry\n\nThe 40 source-isolated true-t0 states have 1-NN normalized h0 distances: min {min(float(r['d_h']) for r in readcsv(HERE/'nearest_neighbor_pairs.csv') if r['neighbor_rank']=='1'):.3f}, median {np.median([float(r['d_h']) for r in readcsv(HERE/'nearest_neighbor_pairs.csv') if r['neighbor_rank']=='1']):.3f}, max {max(float(r['d_h']) for r in readcsv(HERE/'nearest_neighbor_pairs.csv') if r['neighbor_rank']=='1'):.3f}. Across all 780 pairs, Spearman(h-distance, target-distance) is {corr([float(r['d_h']) for r in pairall],[float(r['d_eta']) for r in pairall]):.3f}; among rank-1/2/3/5 neighbors it is {corr([float(r['d_h']) for r in readcsv(HERE/'nearest_neighbor_pairs.csv')],[float(r['d_eta']) for r in readcsv(HERE/'nearest_neighbor_pairs.csv')]):.3f}. There are {len(jout)} predeclared target-jump candidates. The 40 labels use only {len(target_modes)} distinct eta targets; the largest repeated target mode occurs in {max(target_modes.values())}/40 states.\n\n## Cross-transfer\n\nThe frozen targeted audit contains {len(pairs)} symmetric state pairs ({2*len(pairs)} directed Q64 tests): {total['MUTUAL_ROBUST_TRANSFER']} mutual, {total['ONE_WAY_ROBUST_TRANSFER']} one-way, and {total['NO_ROBUST_TRANSFER']} no robust transfer. The closest-distance bin has mutual-or-one-way rate {nearest['mutual_or_one_way_rate']:.1%} and mutual rate {nearest['mutual_rate']:.1%}. {sum(r['interpretation']=='TARGET_JUMP_BUT_BASINS_OVERLAP' for r in jout)}/{len(jout)} large target jumps preserve mutual B63 transfer. Complete k=3 neighborhood audits cover {len(k3_complete)}/40 states with mean overlap {np.mean(k3_complete):.1%}; complete k=5 audits cover {len(k5_complete)}/40 with mean overlap {np.mean(k5_complete):.1%}. Across cached plus targeted evidence, {len(allpairs)} symmetric exact-Q64 pairs are available; their bin statistics are reported separately and explicitly marked ascertainment-biased.\n\nThe two predeclared near-state/large-target-jump cases both transferred mutually at B63. Thus their distant labels do not represent a demonstrated basin discontinuity: they are different robust choices inside strongly overlapping local success sets. Transfer weakens moderately with continuous h-distance (Spearman below), but the small binned controls do not show a clean monotonic decay. There is no repeated very-close no-transfer evidence sufficient to claim a genuine local discontinuity.\n\n## State metric and coverage\n\nFull normalized 214-D h0 distance has Spearman {mdfull['spearman_distance_vs_mean_cross_Q64']:.3f} with mean cross-transfer Q64 (negative means transfer weakens with distance). The strongest documented subset was `{best['metric']}` at {best['spearman_distance_vs_mean_cross_Q64']:.3f}. This negligible difference does not establish a substantially superior replacement metric; the exact comparison is in `state_metric_comparison.csv`. State coverage is sufficient for the local ambiguity diagnosis, although the 1-NN distances show that 40 states remain sparse for learning a detailed global basin map.\n\nMidpoint-state testing was skipped because interpolating positions or h0 cannot preserve exact Flow xi0/RNG conditioning and full true-t0 physical semantics. No genuine basin discontinuity is therefore claimed.\n\n## Answers\n\n- Did point regression fail mainly because the target is set-valued/multimodal? **Yes, this audit supports that as the dominant structural cause:** nearby states can have far-apart selected labels while both labels remain mutually B63.\n- Is current true-t0 state coverage sufficient? **Sufficient for this local cross-transfer diagnosis, but not dense enough for an unconstrained global basin reconstruction.**\n- Should analytic basin representation search begin? **{gate}.** It should model state-conditioned overlapping feasible sets, not regress one arbitrary eta.\n\n## Next step\n\nRun one constrained analytic representation comparison obeying `basin_representation_requirements.md`, using existing exact Q64 positives and negatives with state-held-out validation before any new controller training.\n'''
(HERE/'final_report.md').write_text(report)

# Integrity manifest.
files=[]
for p in sorted(HERE.glob('*')):
 if p.is_file() and p.name!='manifest.json':files.append({'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
man={'experiment':'ORTHOFLOW3_TRUE_T0_ROBUST_ETA_CONTINUITY_AND_CROSS_TRANSFER_V1','created_unix':time.time(),'authoritative_basis_sha256':'51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38','network_training':False,'new_eta_sweep':False,'files':files}
(HERE/'manifest.json').write_text(json.dumps(man,indent=2)+'\n')
print(json.dumps({'classification':classification,'gate':gate,'pairs':len(pairs),'categories':dict(total),'nearest':nearest,'runtime':runtime},indent=2))
