#!/usr/bin/env python3
"""Freeze state/direction manifests and prove the prescribed audit exceeds cap."""
from __future__ import annotations
import csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT=Path('/home/zhihan/research/Basin_C1');HERE=ROOT/'diagnostics/orthoflow3_conservative_basin_ball_v1';Q=ROOT/'diagnostics/orthoflow3_q_learnability_v2';BASE=ROOT/'diagnostics/orthoflow3_direct_eta_baseline_v1';QG=ROOT/'diagnostics/orthoflow3_q_guided_direct_eta_v1';TOKEN='orthoflow3_conservative_basin_ball_v1_states_20260927'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rank(x):return hashlib.sha256(f'{TOKEN}|{x}'.encode()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def load(root,stem):
 out=[]
 for p in sorted((root/'raw'/stem).glob('shard*.jsonl')):
  if p.stem.removeprefix('shard').isdigit():out += [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 return out
def empty(name,fields):
 with (HERE/name).open('w',newline='') as f:csv.writer(f).writerow(fields)
def main():
 HERE.mkdir(parents=True,exist_ok=True);source=json.loads((Q/'eligible_state_manifest.json').read_text());states=source['selected_states'];ordered=sorted(states,key=lambda r:rank(r['state_id']));selected=ordered[:24]
 manifest={'schema':'orthoflow3_conservative_ball_state_manifest_v1','frozen_before_new_outcomes':True,'permutation_seed':TOKEN,'eligible_count':len(states),'selection_rule':'SHA256(seed|state_id) uniform deterministic ordering; first 24','selected_count':24,'selected_states':selected,'complete_permutation_state_ids':[r['state_id'] for r in ordered],'split_composition':{s:sum(r['split']==s for r in selected) for s in ('train','val','test')}};dump(HERE/'state_manifest.json',manifest)
 # Frozen directions: axes then 12 midpoint Fibonacci-sphere directions.
 dirs=[]
 for i in range(3):
  for sign in (1,-1):v=[0.,0.,0.];v[i]=float(sign);dirs.append((f'axis_{i+1}_{"p" if sign>0 else "m"}',v,'axis'))
 phi=(1+math.sqrt(5))/2
 for k in range(12):
  z=1-2*(k+.5)/12;theta=2*math.pi*k/phi;r=math.sqrt(max(0,1-z*z));dirs.append((f'fib_{k:02d}',[r*math.cos(theta),r*math.sin(theta),z],'fibonacci_midpoint'))
 with (HERE/'ray_directions.csv').open('w',newline='') as f:w=csv.writer(f);w.writerow(['direction_id','d1','d2','d3','construction']);w.writerows([[n,*v,c] for n,v,c in dirs])
 # Common-cloud exact cache inventory. All reused data use the same Q-v2 future root.
 cloud=list(csv.DictReader(open(Q/'eta_probe_cloud.csv')));sids={r['state_id'] for r in selected};records=[]
 for stem in ('base_rollout_plan','zero_followup_plan'):records+=load(Q,stem)
 for stem in ('zero_completion_plan','active_promotion_round1_plan','active_promotion_round2_plan','active_promotion_round3_plan','active_promotion_round4_plan'):records+=load(BASE,stem)
 unique={}
 for r in records:
  if r['state_id'] in sids and r['probe_id'] in {c['probe_id'] for c in cloud}:unique[(r['state_id'],r['probe_id'],int(r['future_index']))]=r
 common=[];reuse8=0;steps=[]
 for st in selected:
  for c in cloud:
   rr=[unique[k] for k in unique if k[0]==st['state_id'] and k[1]==c['probe_id'] and k[2]<8];reuse8+=len(rr);steps += [r['continuation_steps'] for r in rr]
   common.append({'state_id':st['state_id'],'split':st['split'],'probe_id':c['probe_id'],'eta1':c['eta1'],'eta2':c['eta2'],'eta3':c['eta3'],'cached_trials_of_8':len(rr),'cached_successes':sum(r['success'] for r in rr),'screening_complete_from_cache':len(rr)==8,'new_trials_needed_for_8':8-len(rr)})
 with (HERE/'common_cloud_results.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(common[0]));w.writeheader();w.writerows(common)
 desired=24*24*8;common_new=desired-reuse8;mean_steps=float(np.mean(steps));cache={'status':'PASS','compatible_future_root_seed':2026092702,'selected_common_cloud_desired_trials':desired,'exact_reused_trials':reuse8,'new_trials_needed_to_complete_common_screen':common_new,'source_records_scanned':len(records),'tuple_key':'state_id, probe_id/exact eta, future_index, fixed OrthoFlow3/projection/horizon/RNG semantics','mean_cached_continuation_steps':mean_steps,'other_diagnostics_inventoried':['representation migration','local continuity','bilateral canonical','Q-v2','Direct-eta baseline','Q-guided Direct-eta','min-def audits'],'note':'Only exact Q-v2-root common-cloud tuples counted; approximate eta/state matches rejected.'};dump(HERE/'cache_reuse_audit.json',cache)
 auth={'orthoflow3_path':str(ROOT/'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py'),'orthoflow3_sha256':sha(ROOT/'diagnostics/double_bottleneck_eta_basis_redesign/tools/bases.py'),'basis_interface':str(ROOT/'shared_control/basis_families.py'),'basis_interface_sha256':sha(ROOT/'shared_control/basis_families.py'),'control_chain':'Flow -> first projection -> fixed-eta OrthoFlow3 correction -> second projection -> environment','eta_fixed_full_continuation':True,'basis_recomputed_each_step':True,'B63':'>=63/64 successes','screening':'8 matched futures; 8/8 is not B63','termination_categories':['success','safe_deadlock','timeout','collision','other_numerical']};dump(HERE/'authoritative_semantics.json',auth)
 norm={'active_eta_box_low':[.5,-.5,0.],'active_eta_box_high':[1.25,.5,.75],'affine_center':[.875,0.,.375],'scale_high_minus_low':[.75,1.,.75],'normalized_rule':'(eta-center)/scale for active-domain geometry','explicit_zero_special_case':[0.,0.,0.],'zero_retained_without_clipping':True,'source':str(ROOT/'diagnostics/orthoflow3_representation_migration_v1/orthoflow3_search_config.json'),'source_sha256':sha(ROOT/'diagnostics/orthoflow3_representation_migration_v1/orthoflow3_search_config.json')};dump(HERE/'eta_normalization.json',norm)
 # Exact prescribed minimum after centers for a useful nonzero-radius ball.
 ray_min=24*(18+6)*8 # one point every ray + a failure bracket point for six limiting rays
 boundary_extend=24*6*(64-8);inside_screen=24*12*8;inside_extend=24*3*(64-8);outside=24*8*8;post_center=ray_min+boundary_extend+inside_screen+inside_extend+outside
 expected_center=24*(64-8);projected_min=common_new+expected_center+post_center;worst_ray=24*18*(5+3)*8;projected_worst=common_new+expected_center+worst_ray+boundary_extend+inside_screen+inside_extend+outside
 first6_ids={s['state_id'] for s in selected[:6]}
 first6_common_new=sum(int(r['new_trials_needed_for_8']) for r in common if r['state_id'] in first6_ids)
 first6_min=first6_common_new+6*(56+856)
 first6_worst=first6_common_new+6*(56+1816)
 cost={'automatic_new_continuation_cap':15000,'automatic_physical_step_cap':8000000,'common_cloud_new_exact':common_new,'center_promotion_nominal_one_per_state':expected_center,'nontrivial_ray_minimum_screening':ray_min,'six_boundary_B63_extensions':boundary_extend,'inside_12x8_screening':inside_screen,'inside_3x56_B63_extensions':inside_extend,'outside_shell_8x8':outside,'post_center_nontrivial_minimum_new':post_center,'projected_minimum_new_total':projected_min,'worst_case_ray_screening_5_coarse_plus_3_bisection':worst_ray,'projected_worst_case_new_total':projected_worst,'mean_cached_steps_per_continuation':mean_steps,'projected_minimum_physical_steps_using_cached_mean':int(round(projected_min*mean_steps)),'projected_worst_physical_steps_using_cached_mean':int(round(projected_worst*mean_steps)),'exceeds_continuation_cap_even_before_optional_pairs_or_ellipsoid':projected_min>15000,'exceeds_physical_cap_at_projected_minimum':projected_min*mean_steps>8000000,'smallest_next_experiment':{'description':'Protocol-preserving feasibility pilot on the first 6 states of the already frozen permutation','common_cloud_new_exact':first6_common_new,'projected_minimum_new_total':first6_min,'projected_worst_case_new_total':first6_worst,'projected_minimum_physical_steps_using_cached_mean':int(round(first6_min*mean_steps)),'projected_worst_physical_steps_using_cached_mean':int(round(first6_worst*mean_steps))},'decision':'STOP_BEFORE_NEW_ROLLOUTS'};dump(HERE/'cost_preflight.json',cost)
 headers={'center_selection.csv':['state_id','status','center_probe_id','reason'],'robust_center_results.csv':['state_id','eta1','eta2','eta3','successes','trials','B63'],'ray_screening.csv':['state_id','direction_id','radius','successes','trials'],'boundary_brackets.csv':['state_id','direction_id','rho_success','rho_failure'],'robust_directional_radii.csv':['state_id','direction_id','confirmed_radius','successes','trials'],'conservative_ball_parameters.csv':['state_id','c1','c2','c3','r_raw','r_ball','kappa'],'independent_inside_points.csv':['state_id','point_id','eta1','eta2','eta3','normalized_radius'],'inside_ball_validation.csv':['state_id','point_id','successes','trials','B63'],'outside_shell_validation.csv':['state_id','point_id','successes','trials'],'empirical_ball_coverage.csv':['state_id','known_robust_inside','known_robust_outside','known_failure_inside'],'anisotropy_statistics.csv':['state_id','anisotropy_ratio','directional_cv','bin'],'local_pair_ball_stability.csv':['pair_id','center_displacement','radius_displacement','intersection_union'],'center_cross_transfer.csv':['pair_id','direction','successes','trials','B63'],'q_exploitation_case_check.csv':['state_id','Qhat','true_Q64','selected_in_audit','ball_available','inside_ball']}
 for n,h in headers.items():empty(n,h)
 # Read-only exploitation membership check; no ball exists.
 exploit=[]
 for r in csv.DictReader(open(QG/'critic_exploitation_audit.csv')):
  if r['critic_exploitation_candidate']=='True':exploit.append({'state_id':r['state_id'],'Qhat':r['Qhat'],'true_Q64':r['true_Q64'],'selected_in_audit':r['state_id'] in sids,'ball_available':False,'inside_ball':'UNRESOLVED'})
 if exploit:
  with (HERE/'q_exploitation_case_check.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(exploit[0]));w.writeheader();w.writerows(exploit)
 decision={'classification':'BASIN_BALL_AUDIT_UNDERRESOLVED','reason':'The mandatory prescribed 24-state nontrivial-ball core exceeds the automatic continuation cap before optional local pairs or ellipsoid work. B63 semantics and population were not weakened.','new_rollouts_executed':0,'future_H_c_r_supported':False,'center_deployment_supported':False,'sphere_vs_ellipsoid':'UNRESOLVED','single_smallest_next_experiment':'Run a protocol-preserving feasibility pilot on the first 6 states of the already frozen permutation, using the frozen 18 directions and unchanged B63/inside/shell validation. Its deterministic estimate is 5,832-11,592 new continuations (about 1,665,667-3,310,769 physical steps).'};dump(HERE/'final_decision.json',decision)
 runtime={'reused_continuations_inventoried':reuse8,'new_continuations':0,'physical_steps':0,'wall_time_seconds':0,'gpu_shards':0,'gpu_memory':0,'cpu_threads':1,'ram_observed':'negligible metadata-only preflight','stopped_before_rollouts':True};dump(HERE/'runtime_statistics.json',runtime)
 report=f'''# OrthoFlow3 conservative basin-ball audit v1\n\n## Decision\n\n**BASIN_BALL_AUDIT_UNDERRESOLVED**\n\nThe 24-state population and 18 directions were frozen before outcomes. No new rollout was launched because the mandatory core cannot fit the 15,000-continuation automatic cap without weakening the protocol.\n\n- Selected states: 24 ({manifest['split_composition']}).\n- Exact common-cloud cache reuse: {reuse8}/{desired}; {common_new} new trials would be needed merely to finish 8-seed common screening.\n- Minimum post-center work for a nontrivial ball: {post_center} new continuations.\n- Projected minimum including common completion and one center promotion/state: {projected_min} new continuations, approximately {cost['projected_minimum_physical_steps_using_cached_mean']} physical steps.\n- Projected worst-case with five coarse radii and three bisections: {projected_worst} continuations, approximately {cost['projected_worst_physical_steps_using_cached_mean']} steps.\n\nThe fixed post-center validation alone contains six boundary B63 extensions ({boundary_extend}), inside screening/promotions ({inside_screen+inside_extend}), and outside shell ({outside}); it cannot be removed without changing the requested science. Local-pair and ellipsoid stages were not included in these estimates.\n\nNo evidence about ball validity, center robustness, anisotropy, or Q-exploitation rejection was inferred from this stop. Training `H(h)->[c,r]` is **not yet supported**.\n''';(HERE/'conservative_basin_ball_report.md').write_text(report)
 with (HERE/'conservative_basin_ball_report.md').open('a') as f:
  f.write(f"\nThe single smallest justified next experiment is a protocol-preserving pilot on the first 6 states in the frozen permutation. It keeps all 18 directions and validation rules, with a deterministic estimate of {first6_min}-{first6_worst} new continuations (approximately {cost['smallest_next_experiment']['projected_minimum_physical_steps_using_cached_mean']}-{cost['smallest_next_experiment']['projected_worst_physical_steps_using_cached_mean']} physical steps).\n")
 files=[]
 for p in sorted(HERE.iterdir()):
  if p.is_file() and p.name!='manifest.json':files.append({'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size})
 dump(HERE/'manifest.json',{'schema':'orthoflow3_conservative_basin_ball_v1','classification':'BASIN_BALL_AUDIT_UNDERRESOLVED','frozen_before_new_outcomes':True,'files':files});print(json.dumps({'state_manifest':manifest,'cache':cache,'cost':cost,'decision':decision},indent=2))
if __name__=='__main__':main()
