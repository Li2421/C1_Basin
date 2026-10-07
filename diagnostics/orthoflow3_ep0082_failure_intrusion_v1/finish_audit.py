from core import *
import argparse
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--classification',required=True);ap.add_argument('--route',required=True);ap.add_argument('--reason',required=True);a=ap.parse_args()
 permitted={'SHARP_BOUNDARY_INTRUSION_DOMINANT':'ANALYTIC_EXCLUSION_MODEL_STRONGLY_MOTIVATED','ENCLOSED_HOLE_SUPPORTED':'ANALYTIC_MODEL_WITH_HOLES_REQUIRED','INTERLEAVED_BOUNDARY_SUPPORTED':'SIMPLE_ANALYTIC_MODEL_UNLIKELY','MIXED_NOTCH_AND_LOCAL_INTERLEAVING':'MIXED_REPRESENTATION_REQUIRED','TOPOLOGY_UNDERRESOLVED':'STILL_UNDERRESOLVED'}
 assert permitted[a.classification]==a.route
 lookup={r['eta_key']:r for r in read(H/'exact_q64_ep0082.csv')};stats=read(H/'transition_statistics.csv');corr=read(H/'failure_corridor_results.csv');conn=json.load(open(H/'connectivity_audit_summary.json'));regions=read(H/'region_connectivity_summary.csv');old=read(H/'historical_interpolation_explanation.csv');holes=read(H/'hole_tests.csv');cachelines=read(H/'cached_conditional_lines.csv');brackets=read(H/'opposite_label_brackets.csv');rt=json.load(open(H/'runtime_statistics.json'))
 assert all(r['complete']=='True' for r in stats+corr)
 assert all(float(r['width'])<=.010000001 for r in brackets),'Do not finalize unrefined registered transitions'
 assert all(r['exact_Q64_available']=='True' for r in read(H/'adaptive_probe_results.csv'))
 hist=dict(sorted(Counter(int(r['transition_count']) for r in stats).items()));n=len(stats);minimum=min(float(r['width']) for r in brackets);cached_min=min(float(r['min_transition_spacing']) for r in cachelines if r['min_transition_spacing'])
 if a.classification=='SHARP_BOUNDARY_INTRUSION_DOMINANT':
  assert conn['representatives_with_sampled_exterior_paths']==conn['working_regions']
  assert sum(int(r['transition_count'])<=1 for r in stats)/n>=.75
  assert sum(int(r['transition_count'])>=3 for r in stats)<2
 # Other positive classifications require an explicit separately documented scientific review.
 if a.classification not in ('SHARP_BOUNDARY_INTRUSION_DOMINANT','TOPOLOGY_UNDERRESOLVED'):assert (H/'positive_topology_classification_review.json').exists()
 frac={str(k):sum(int(r['transition_count'])==k for r in stats)/n for k in (0,1,2)};frac['>=3']=sum(int(r['transition_count'])>=3 for r in stats)/n
 completed_rounds=[];critical=0.;physical_rt=0;newcon=0;wall_starts=[];wall_ends=[]
 for d in sorted((H/'raw').iterdir()):
  shards=[json.load(open(p)) for p in d.glob('shard*_runtime.json')]
  if not shards:continue
  crit=max(r['wall_seconds'] for r in shards);critical+=crit;completed_rounds.append(dict(batch=d.name,new_continuations=sum(r['new_continuations'] for r in shards),physical_steps=sum(r['physical_steps'] for r in shards),critical_worker_seconds=crit))
  for p in d.glob('shard*_runtime.json'):
   r=json.load(open(p));wall_ends.append(p.stat().st_mtime);wall_starts.append(p.stat().st_mtime-r['wall_seconds'])
 rt.update(critical_rollout_wall_sum_seconds=critical,rollout_execution_window_seconds=max(wall_ends)-min(wall_starts),wall_time_note='Batch critical-time sum excludes user-directed switch to another experiment; execution window includes that interruption and analysis. Worker durations exclude initialization.',completed_batches=completed_rounds,new_eta_queries=rt['new_exact_Q64'],GPU_memory_MiB='See sampled resource_observations.jsonl; allocation is not measured utilization.',CPU_threads=4,RAM_allocation_GiB_total=16,failed_submission=dict(job=993,reason='Missing array index; terminated before any continuation',physical_steps=0,new_continuations=0))
 dump('runtime_statistics.json',rt)
 decision=dict(classification=a.classification,ANALYTIC_BASIN_ROUTE=a.route,scope='ep0082 only; empirical matched64-seed B63 geometry',reason=a.reason,existing_exact_Q64_reused=rt['existing_exact_Q64_reused'],new_exact_Q64=rt['new_exact_Q64'],total_exact_Q64=len(lookup),internal_candidates=conn['original_internal_candidates'],working_regions=conn['working_regions'],representatives_with_sampled_exterior_paths=conn['representatives_with_sampled_exterior_paths'],individual_internal_points_with_sampled_exterior_paths=conn['original_internal_points_with_paths'],original_internal_points_unresolved=conn['original_internal_candidates']-conn['original_internal_points_with_paths'],candidate_enclosed_after_six_rays=sum(r['six_direction_surrounding_support']=='True' and r['validated_sampled_escape']!='True' for r in holes),confirmed_enclosed_holes=0,unresolved_representative_regions=conn['working_regions']-conn['representatives_with_sampled_exterior_paths'],conditional_lines=n,transition_histogram=hist,transition_fractions=frac,minimum_new_transition_bracket=minimum,maximum_final_transition_bracket=max(float(r['width']) for r in brackets),minimum_existing_opposite_label_bracket=cached_min,cached_lines_with_at_least3_transitions=sum(int(r['transitions'])>=3 for r in cachelines),independent_local_lines_with_at_least3_transitions=sum(int(r['transition_count'])>=3 for r in stats),historical_interpolation_failures=conn['old_interpolation_failures'],historical_interpolation_failures_with_exterior_paths=conn['old_interpolation_failures_with_paths'],no_continuous_path_certificate=True,hole_absence_proven=False,global_topology_proven=False,neural_training=False,family_fitting=False,new_states=False,new_scenarios=False)
 auxiliary=json.load(open(H/'corridor_line_summary.json'))
 decision['auxiliary_corridor_line_transition_histogram']=auxiliary['histogram']
 decision['auxiliary_corridor_lines_with_at_least3_transitions']=len(auxiliary['repeated_lines'])
 next_step='Evaluate one frozen low-complexity broad-support-minus-exclusions hypothesis on independently chosen eta; retain the present sharp boundary locations and state-dependent deformation as constraints.' if a.route=='ANALYTIC_EXCLUSION_MODEL_STRONGLY_MOTIVATED' else 'Target the remaining ambiguous failure pocket(s) with an independent transverse slice and explicit escape test; do not fit or train until resolved.'
 decision['next_scientific_step']=next_step;dump('topology_decision.json',decision)
 implication=f'''# Representation implication

`ANALYTIC_BASIN_ROUTE = {a.route}`

{a.reason}

Finite exact-Q64 paths are observed failure corridors at the reported sampling resolution. They do not prove continuous connectedness, absence of enclosed holes between probes, or a low-complexity global boundary. A single historical S-F-S-F-S segment is not sufficient for interleaving: two failure branches may join the same external region. No analytic family was fit in this audit.

Next scientific step: {next_step}
'''
 (H/'representation_implication.md').write_text(implication)
 out=['# ep0082 failure-intrusion slice audit','',f'Classification: **{a.classification}**  ',f'`ANALYTIC_BASIN_ROUTE = {a.route}`','',a.reason,'',
 f'Reused {rt["existing_exact_Q64_reused"]} exact-Q64 eta ({rt["initial_cache_exact"]} frozen original + {rt["supplemental_cache_exact"]} compatible later cache); acquired {rt["new_exact_Q64"]} new exact-Q64 eta. Total {len(lookup)} exact eta. Every topology label uses64 distinct matched future indices, with B63 defined as >=63 successes. No screening-only point is a topology label.','',
 '## Failure regions and exterior evidence','',
 'The four working regions come from a descriptive0.20-radius graph over40 original hull-interior negative observations; they are not asserted to be four actual topological components. Scale sensitivity and exact support/density are in internal_failure_candidates.csv and region_definitions.json.','',
 '|Working region|Original internal observations|Representative sampled escape|Individual original negatives with sampled escape|Still unlinked|',
 '|---|---:|---|---:|---:|']
 for r in regions:out.append(f'|{r["region_id"]}|{r["original_internal_candidates"]}|{r["representative_has_sampled_exterior_path"]}|{r["original_candidates_with_sampled_exterior_path"]}|{r["unresolved_original_candidates"]}|')
 out += ['',f'{sum(r["sampled_failure_path_valid"]=="True" for r in corr)}/{len(corr)} complete candidate corridors have non-B63 at every registered location. Alternate routes do not erase interrupted original routes. Negative links have maximum normalized sampling gap0.05 and exclude known collinear B63 points. This is empirical support, not a continuum certificate.','',
 f'Six angular rays were tested per representative, with feasible clipping near E_bridge. No representative has six-direction validated success enclosure. Confirmed enclosed holes:0; absence of holes is not established. Unresolved representative regions:{decision["unresolved_representative_regions"]}.','',
 '## Exact conditional lines','',
 '|Line|Observed sequence (S=B63, F=non-B63)|Transitions|Smallest opposite-label bracket|',
 '|---|---|---:|---:|']
 for r in stats:out.append(f'|{r["line_id"]}|{r["label_sequence"]}|{r["transition_count"]}|{float(r["min_transition_width"]):.6f}|' if r['min_transition_width'] else f'|{r["line_id"]}|{r["label_sequence"]}|{r["transition_count"]}|—|')
 out += ['',f'Transition histogram:{hist}. Fractions:0={frac["0"]:.3f},1={frac["1"]:.3f},2={frac["2"]:.3f},>=3={frac[">=3"]:.3f}. All final registered opposite-label brackets are <=0.01; minimum new bracket={minimum:.6f}. Historical minimum opposite-label spacing={cached_min:.6f}. These bracket an empirical label change, not a measured smooth probability gradient.','',
 f'The historical cache contains one S-F-S-F-S line spanning0.92039 normalized units with large unsampled gaps; it is reported separately from the prospective local-line denominator. It cannot by itself distinguish a folded/branched notch from interleaving. Newly tested independent local lines with>=3 transitions:{decision["independent_local_lines_with_at_least3_transitions"]}. Auxiliary deduplicated straight corridor segments have transition histogram {auxiliary["histogram"]}; see corridor_line_transitions.csv. These are post-acquisition diagnostics, not an inflated prospective-line denominator.','',
 '## Earlier interpolation failures','',
 f'Of12 earlier ep0082 success-success probes,4 were non-B63. {conn["old_interpolation_failures_with_paths"]}/4 now have an exact sampled negative-link path to the domain boundary. The per-point links are in historical_interpolation_explanation.csv. Failure points without such a path remain unresolved; geometric proximity to another linked point is not counted.','',
 '## Observed-only figures','',
 '- eta_3d_success_failure.svg: normalized native eta and isometric observed labels.',
 '- tangential_slices.svg, mixed_s1_n_slices.svg, mixed_s2_n_slices.svg: fixed R0 coordinate bands (not exact planes).',
 '- failure_corridor_view.svg: each tested route separately; dashed unobserved gaps remain unknown.',
 '- transition_line_examples.svg: exact Q64 conditional sequences.','',
 '## Resources and integrity','',
 f'New continuations:{rt["new_continuations"]}; physical steps:{rt["new_physical_steps"]}; sum batch critical rollout time:{critical/60:.2f}min; execution window including the user-directed task switch:{rt["rollout_execution_window_seconds"]/60:.2f}min. Max2GPU shards and4CPU threads;16GiB RAM allocated in total. Job993 failed before execution; no scientific result was produced by that submission. No duplicate eta/future was rerun by resubmission.','',
 'Basis SHA256, current Flow/h conditioning, source group, matched future namespace, horizon, both projections and monitor/history semantics remain frozen. Authoritative source hashes and accepted numerical replay h aliases are rechecked in completion_audit.json. No network training, new state/scenario, full Basin sweep, or family fitting occurred within this audit.','',
 '## Decision limit and next step','',
 'Classification concerns the sampled representatives and registered slices, not every unobserved point in the hull. Neither an enclosure theorem nor a theorem excluding microscopic interleaving is claimed.','',next_step,'']
 (H/'final_report.md').write_text('\n'.join(out))
 dump('hypothesis_status.json',[
  dict(hypothesis_id='H1_BOUNDARY_INTRUSION',mathematical_form='Internal non-B63 support connected by sampled negative paths to E_bridge exterior',current_status='SUPPORTED_EMPIRICALLY' if conn['representatives_with_sampled_exterior_paths']==4 else 'PARTIAL_SUPPORT',key_support=conn,key_failure='Some tested routes interrupted by B63; all failures retained',unresolved_property='Continuum connectivity and unlinked individual candidates',next_discriminating_test=next_step),
  dict(hypothesis_id='H2_ENCLOSED_HOLE',mathematical_form='Enclosed failure pocket surrounded by successful support',current_status='NOT_CONFIRMED',key_support=None,key_failure='No validated six-direction success enclosure; sampled exterior paths where documented',unresolved_property='Absence of unobserved holes is not proved',next_discriminating_test=None,confirmed=0,absence_proven=False),
  dict(hypothesis_id='H3_INTERLEAVING',mathematical_form='Repeated exact robust label alternation on independent local lines, not explained by one branched notch',current_status='NOT_INDEPENDENTLY_DEMONSTRATED' if decision['independent_local_lines_with_at_least3_transitions']<2 else 'REQUIRES_FOLDED_NOTCH_DISCRIMINATION',key_support={'historical_repeated_lines':decision['cached_lines_with_at_least3_transitions']},key_failure={'prospective_repeated_lines':decision['independent_local_lines_with_at_least3_transitions'],'auxiliary_corridor_repeated_lines':len(auxiliary['repeated_lines'])},unresolved_property='Subsampling-scale alternation and folded-notch distinction',next_discriminating_test=None)])
 state=json.load(open(H/'working_state.json'));state.update(status='SCIENTIFIC_WORK_FINISHED_PENDING_COMPLETION_AUDIT',classification=a.classification,next_action='Independent consistency/completion check, artifact hashes, then final handoff; no additional experiment.');dump('working_state.json',state)
 print(json.dumps(dict(classification=a.classification,route=a.route,new_exact=rt['new_exact_Q64'],regions_connected=conn['representatives_with_sampled_exterior_paths'],old_interpolation_failures_connected=conn['old_interpolation_failures_with_paths'])))
if __name__=='__main__':main()
