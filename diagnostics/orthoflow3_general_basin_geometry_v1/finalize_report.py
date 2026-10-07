#!/usr/bin/env python3
"""Finalize only the explicitly audited saturation branch. Never launch work."""
from audit import *
import subprocess

TAG='minimal_polyhedral_refined_r7_stable'
CLASS='NO_LOW_COMPLEXITY_ANALYTIC_FAMILY_EVIDENCED'
NEXT='Target a preregistered small conditional slice around ep0082\'s known internal non-B63 region to distinguish a few sharp, boundary-connected intrusion branches from genuinely interleaved feasibility; do not train or merely add unconstrained fitting capacity.'

def main():
 sat=json.load(open(HERE/'scientific_saturation_audit.json'));assert sat['two_consecutive_sub5pp_refinements'] and not sat['fresh_precision_pass']
 assert sha(D/'double_bottleneck_eta_basis_redesign/tools/bases.py')==SHA
 jobs=subprocess.run(['squeue','-h','-u','zhihan','-o','%i %j'],capture_output=True,text=True,check=True).stdout.strip();assert not jobs,'Finish/resolve active jobs before finalizing'
 findings=json.load(open(HERE/'current_findings.json'));runtime=json.load(open(HERE/'runtime_statistics_current.json'))
 corrected=json.load(open(HERE/'corrected_frozen_model_gates.json'));best=next(r for r in corrected if r['tag']==TAG)
 assert findings['inventory']['conflicts']==0 and findings['adequate_states']==12
 decision=dict(classification=CLASS,scope='No shared low-complexity family has passed all required gates on this12state2scenario panel; not a nonexistence theorem.',scientific_stop='SECTION20A_OPERATIONAL_SATURATION',saturation_audit='scientific_saturation_audit.json',selected_family=None,margin_label_ready=False,training_executed=False,representation_gate_pass=False,
 best_tested_candidate=best,fresh_retained_validation=dict(false_inclusions=5,total=72,new_Toy_points=48,independent_unchanged_DB_points_reused=24),
 DB_subpanel='Four frozen DB instances passed24/24 retained B63 and have favorable descriptive recall; not sufficient to rescue the failed shared-family gate or establish broad DB population reliability.',
 topology=dict(main_component_dominance='UNDERRESOLVED: finite validated-link graph does not meet90percent criterion',enclosed_holes='NONE_CONFIRMED; absence not proved',notches='Seven sampled failure-to-boundary routes support intrusions; twelve tested routes interrupted, other pockets unresolved',invariance='Anisotropic support and state-dependent exclusions broadly recur; topological invariance/change not established'),
 integrity_correction='Manifest-derived validation roles supersede substring-derived historical recall. Raw Q64 and frozen fresh validation unchanged; corrected intended fit exceeds8-facet recipe onep0082. This capacity issue is NOT the stop condition.',next_scientific_step=NEXT)
 dump('final_decision.json',decision);dump('cross_scenario_family_decision.json',decision)
 dump('selected_family.json',dict(selected=False,family=None,reason='Fresh retained precision failed; operational saturation reached',best_rejected_candidate='minimal_polyhedral_support_quadratic_exclusion',frozen_rejected_parameters=f'family_models_{TAG}.json',deploy_or_train=False))
 dump('representation_gate.json',dict(pass_all=False,corrected_nonfitting_positive_metrics=best,fresh_inside_false_inclusions=5,required_false_inclusions=0,states=12,compatible_scenarios=2,primary_failed_gate='RETAINED_PRECISION',topology_certificate=False,train=False))
 dump('runtime_statistics.json',runtime)
 # State/scenario alignment is descriptive, not a success-occupancy comparison.
 aligned=read(HERE/'cross_state_alignment.csv');al=[]
 for name,predicate in [('within_Toy',lambda r:not r['state_a'].startswith('DB') and not r['state_b'].startswith('DB')),('within_DB',lambda r:r['state_a'].startswith('DB') and r['state_b'].startswith('DB')),('cross_scenario',lambda r:r['state_a'].startswith('DB')!=r['state_b'].startswith('DB'))]:
  rr=[r for r in aligned if predicate(r)];al.append(dict(group=name,pairs=len(rr),median_scaled_aligned_chamfer=float(np.median([float(r['scaled_aligned_chamfer']) for r in rr])) if rr else None,interpretation='Positive-cloud PCA/extent alignment, sign ambiguity minimized; nonuniform sampling, no occupancy/topology equivalence claim'))
 dump('alignment_summary.json',al)
 sc=read(HERE/'scenario_compatibility.csv')
 for r in sc:
  if 'exact_Q64_cache' in r:r['initial_exact_Q64_cache']=r.pop('exact_Q64_cache')
  ss=next((a for a in findings['scenarios'] if a['scenario']==r['scenario']),None);r['current_exact_Q64']=ss['exact_eta'] if ss else 0;r['adequate_states']=ss['adequate_states'] if ss else 0
 write('scenario_compatibility.csv',sc)
 math=r'''# No selected general analytic family

The best tested candidate below is **REJECTED**, not a conservative learning label.
Raw fitted parameters are frozen in `family_models_minimal_polyhedral_refined_r7_stable.json`.

Let z=(eta-(0.875,0,0.375))/(0.75,1,0.75), with componentwise division.
For state i the finite candidate is

    S_i = { z in E_bridge : a_ij^T z + b_ij <= 0, j=1..K_i;
                           q_i(z)=z^T A_i z+d_i^T z+e_i >=0 }, K_i<=8.

Unit-normal support faces are selected from the fit-positive hull by optimal
minimum set cover of known exterior non-B63 observations. A single quadratic
rejects the remaining hard negatives. No hull mesh or point-cloud lookup is
stored for membership. The finite deployed formula has at most42 raw coefficients,
or33 meaningful zero-set parameters after normal/scaling redundancies; the
exception beyond the preferred4pieces/32parameters was declared before validation.
The mathematical class can express intrusions, but neither connectedness nor
absence of disconnected pieces is guaranteed by the formula.

Retained construction, delta=0.025 normalized Euclidean units:

    domain unit halfspaces <= -delta;
    a_ij^T z+b_ij <= -delta;
    q_i(z) >= delta*||2 A_i z+d_i||_2 + delta^2*||A_i||_2.

The last inequality is a sufficient Taylor bound for a delta ball to remain
inside the *fitted quadratic inequality*. It is NOT a certificate for actual
closed-loop success. Five final retained samples were confirmed non-B63.

Reproducibility pseudocode (diagnostic only):

    fit_basin(evidence):
        resolve exact acquisition roles from frozen state/eta manifests
        retain non-validation hash-fit B63 + explicitly promoted development B63
        keep all exact non-B63 as hard negative constraints
        compute normalized fit-positive hull, no heldout-positive use
        choose optimal minimal facet cover of exterior negatives
        if more than8 faces required: return UNRESOLVED, never silently enlarge
        fit one bounded-coefficient quadratic against remaining negatives
        return state parameters and provenance snapshot, NOT verified label
    contains(eta): evaluate E_bridge, selected affine faces and q>=0
    retained_contains(eta): evaluate the three eroded inequality groups above
    sample_inside_retained(): frozen scrambled Sobol geometric pool;
        reject outside; freeze6 central/maximin points; exact-Q64 all6

The corrected intended-development fit required9 facets for ep0082 while retaining
all fit positives. This is not a lower bound for a70percent-recall model and does
not prove that a slightly larger family cannot work. No ninth face was silently
added, and no partial corrected fit is selected.
'''
 (HERE/'selected_family_math.md').write_text(math)
 (HERE/'margin_loss_spec.md').write_text('''# Margin-loss readiness: NO

No family passed the empirical retained-precision gate. Do not use these sets as
zero-loss training labels; no controller was trained.

For reproducing the rejected geometric candidate only, a piecewise-differentiable
normalized diagnostic violation can combine positive violations of the eroded
domain faces, support faces, and quadratic Taylor bound. Unit-normal affine
violations are already in normalized eta-length units; polynomial violations
require a fixed positive coefficient/gradient scale frozen before any future
training. No loss normalization or controller hyperparameters were selected here.

`contains` and `retained_contains` are implemented in families.py. This file
deliberately does NOT recommend a training loss for a failed label representation.
''')
 # Compact comprehensive scientific handoff.
 lines=['# OrthoFlow3 general Basin geometry discovery — final report','',f'**Classification: {CLASS}.** No general analytic family selected; no controller/network training.','',
 'This means the tested evidence-driven search did not produce a family passing every gate. It does not prove analytic representability impossible. The stop is the predeclared two-round saturation criterion, not a time/rollout cap, the number of hypotheses, or the eight-face fitting limit.','',
 '## Evidence, conditioning and integrity','',
 f'Reused1,131 prior exact-Q64 state–eta tuples (72,384 continuations) plus976 compatible lower-seed continuation records. Added{runtime["complete_new_exact_Q64"]:,} complete exact-Q64 tuples; final inventory{findings["inventory"]["exact_state_eta"]:,} tuples,{findings["inventory"]["B63"]:,} B63 across44 true-t0 source states. Twelve states satisfy frozen density criteria:8 ToyGiveWay and4 joint4-agent DoubleBottleneck. All40 original Toy source groups and four DB groups remain fixed.','',
 'Three historical controller TEST outputs lie outside E_bridge (one B63,two non-B63). They remain explicitly flagged in the provenance inventory but are excluded from geometry summaries:2,308 in-domain exact tuples,1,607 B63. No new probe was outside E_bridge. One old out-of-domain negative was redundantly constrained by earlier analytic fits; it was already rejected by the domain inequality and cannot explain retained false inclusion. Original frozen fits are preserved.','',
 'The authoritative basis hash is `'+SHA+'`. Eta is selected once, latched, and both projections and timestep-wise basis recomputation remain unchanged. Scenario-specific Flow policies are frozen; same eta basis semantics does not mean identical features or identical dynamics. DB current Flow/h is fixed before varying future seeds.','',
 'One DB projection/numerical attempt is quarantined; its eta has only63 valid seeds and is NOT Q64/B63. No replacement seed or controller tolerance adjustment was used. Historical batch-feature aliases were resolved by exact replay within the original1e-10 protocol tolerance (maximum h difference2.60e-15); eta dedup remains exact float64. No conflicting exact outcomes.','',
 'An end-of-run audit found that the Toy rollout wrapper overwrote the intended validation phase suffix. This had contaminated historical *primary recall* denominators and meant only hash-fit, rather than all promoted, validation positives entered some refits. Raw outcomes and prospective validation are unaffected. Frozen manifests now authoritatively annotate roles; all validation points are excluded from the corrected primary recall. Historical raw data/models are preserved, not silently rewritten. `corrected_frozen_model_metrics.csv` and `corrected_frozen_model_gates.json` supersede historical recall figures. See `acquisition_role_anomaly.json`.','',
 '## Geometry and topology answers','',
 f'**One large connected component? Not established.** No dense state meets the requested90percent validated-link criterion; strict sampled-link largest fractions range{100*findings["strict_sampled_component_range"][0]:.1f}–{100*findings["strict_sampled_component_range"][1]:.1f}percent. Sparse validated links do not prove fragmentation. kNN connectivity is not substituted for rollout-supported paths.','',
 'The newly tested star segments had79/96 successful interior probes;25/32 segments had all three quarter-points B63. This supports many filled directions but does not establish one star kernel or global star-convexity. The earlier88/96 interpolation result used a different selected batch. Five targeted two-leg detours around known failed straight links yielded16/30 B63 probes and0/5 fully successful routes; this rejects those routes, not every possible connecting path.','',
 '**Enclosed holes versus intrusions:** zero enclosed holes are confirmed; absence is not proved. Seven of19 sampled failure-to-boundary routes stayed non-B63; twelve were interrupted by success. In the explicitly investigated internal ep0195 pocket,3/6 axial routes reach the domain boundary through sampled failures. Resolved cases favor boundary intrusions/notches, but the evidence cannot quantify that most failure volume is of this type.','',
 '**Conditional intervals:** Toy eta1 has8/9 single-run lines and one two-success-run counterexample (`FSFFSSS`); eta2 has9/9 single-run lines; eta3 has8/9 single-run lines and one all-failure line. DB eta1 and eta2 each have16/16 single-run lines; eta3 has10 single-run and6 all-failure lines. These are discrete selected line scans, not proofs of conditional convexity or absence of narrow intervening gaps.','',
 '**Common core:** eta=(0.7421875,0.46875,0.7265625) is B63 on35/40 Toy states;38/40 have Q64>=0.90; mean Q64=0.9828125. No tested eta is B63 on all40. This is a genuine high-coverage finite core candidate, not training degeneracy, but no positive-volume universal core is certified. All10 Toy modes were non-B63 on all4 DB states (0/40 robust transfers). Shared family must allow large state/scenario displacement; it cannot mean one universal eta region.','',
 '**Common-core-minus-exclusions:** qualitatively supported within Toy by frequent shared solutions and sampled failure intrusions. The tested low-complexity realizations fail precision/recall gates. Do not equate this qualitative explanation with a validated analytic family.','',
 '## Cross-state and cross-scenario signatures','',
 '| Scenario | States / adequate | Exact eta / B63 | Median diameter | Tangent1/normal span | Corrected full / retained recall* |','|---|---:|---:|---:|---:|---:|']
 for r in findings['scenarios']:
  lines.append(f'| {r["scenario"]} | {r["total_states"]} / {r["adequate_states"]} | {r["exact_eta"]} / {r["B63"]} | {r["median_diameter"]:.3f} | {r["median_anisotropy"]:.2f} | {r["median_full_recall"]:.3f} / {r["median_retained_recall"]:.3f} |')
 lines+=['','*Best rejected F3 instance, common retrospective non-fitting positive panel; not population recall.','',
 'Both scenarios show anisotropic support with narrower directions and substantial conditional intervals. Native location, orientation, extent, thickness and boundary exclusions vary. PCA/anisotropic-scale alignment was computed with sign ambiguity removed, but on nonuniform positive clouds only; similarity does not establish identical occupancy. No extra shear was justified. Exact Basin equality, canonical shape identity, and topology invariance are not established. Broad morphology recurs; parameters and some sampled line patterns differ.','',
 '| Dense state | B63 / non-B63 | Diameter | Anisotropy | Strict linked fraction | Nearest opposite label |','|---|---:|---:|---:|---:|---:|']
 adequate={r['state_id'] for r in read(HERE/'corrected_frozen_model_metrics.csv') if r['tag']==TAG}
 for r in read(HERE/'per_state_geometry_signature.csv'):
  if r['state_id'] in adequate:lines.append(f'| {r["state_id"]} | {r["B63"]} / {r["non_B63"]} | {float(r["diameter"]):.3f} | {float(r["tangent1_normal_ratio"]):.2f} | {float(r["sampled_link_component_fraction"]):.3f} | {float(r["nearest_opposite_label_distance"]):.4g} |')
 lines+=['','No true success-volume fraction is inferred from adaptive samples. Boundary sharpness is represented by measured opposite-label spacing, not an assumed differentiable surface. Component/hole counts and topology beyond validated paths remain unresolved.','',
 '## Analytic families and falsification','',
 '| Family / frozen fit | States | Corrected full recall | Retained recall | Current known retained non-B63 |','|---|---:|---:|---:|---:|']
 for r in corrected:
  lines.append(f'| {r["family"]} / {r["tag"]} | {r["states"]} | {r["median_full_recall"]:.3f} | {r["median_retained_recall"]:.3f} | {r["current_known_retained_false"]} |')
 lines+=['','These are unchanged frozen parameters, re-evaluated on the current manifest-corrected non-fitting panel. Later acquired negatives can falsify an older instance; this is not its original fitting loss. Family A uses up to4 boundary-open caps; B a native band plus cuts; C a PCA asymmetric slab plus cuts; D an ellipsoidal radial body plus cuts; E one quadratic inequality. None satisfies all gates.','',
 'Synthesized F1 added bounded concave-quadratic support plus one quadratic exclusion because the first unbounded polynomial extrapolated beyond successful support. F2 used at most3 supporting halfspaces plus one quadratic because smooth bounded support still admitted unsupported corners. F3 allowed optimal minimum support facets up to8 plus one quadratic: measured2–8 support directions motivated the explicit complexity exception (maximum33 meaningful parameters,9pieces). No neural implicit field, dense mesh, kNN membership or point-cloud lookup was introduced.','',
 '| Prospective validation attempt | Confirmed retained non-B63 / tested |','|---|---:|',
 '| Single quadratic | 21/42 |','| F1 bounded, first | 15/48 |','| F1 bounded, refined | 22/72 |','| F2 three-face support | 16/72 |','| F3 minimal support, first | 7/72 |','| F3 refinement6 | 7/72 |','| F3 refinement7 | 5/72 |','',
 'The last two F3 attempts reuse the SAME24 independent DB validation points only after exact mathematical-instance and fitting-input equality checks; each adds48 genuinely new Toy points. Thus these rows are not144 additional independent observations. The four unchanged DB instances passed24/24; the five last failures occur in Toy (ep0082:3, ep0074:1, ep0139:1). Two are inside the prior successful-point hull; three are outside.','',
 f'The best tested model has corrected median full recall{best["median_full_recall"]:.3f}, retained recall{best["median_retained_recall"]:.3f};11/12 states have full recall>=0.50; robust transfer capture{best["pooled_transfer_recall"]:.3f}. These favorable recall metrics do NOT compensate for5 retained non-B63 points. It is not a reliable margin label.','',
 'After role correction, a new intended-protocol fit was attempted without any new rollout. Preserving all promoted fit positives while excluding exterior negatives requires9 supporting faces for ep0082; the frozen8-face fitter therefore returned unresolved. This is a fitting-recipe limitation, NOT a lower bound at70percent recall or proof that9faces are unreasonable. The partial artifact is explicitly unselected.','',
 '## Scientific saturation and limitations','',
 'On the SAME correctly excluded primary holdout, the last three tested fits have full-recall medians'+', '.join(f'{r["full_recall_fixed_holdout"]:.4f}' for r in sat['rounds'])+' and retained medians'+', '.join(f'{r["retained_recall_fixed_holdout"]:.4f}' for r in sat['rounds'])+'. Successive full-recall gains are below5percentage points; retained recall does not improve. Prospective false-inclusion gains are0 then2.78percentage points. Morphology medians and inferred topology classes show no material new change. Section20A operational saturation is therefore met after correcting the grouping error.','',
 'This is evidence-limited saturation, not exhaustive optimization over all analytic families. Future targeted evidence or a new mathematically justified form could change the conclusion. Repeated adaptive family comparison and a small fresh panel do not provide population calibration. Empirical B63 uses the fixed64 matched futures, not a confidence-bound theorem for the true success probability.','',
 'No general family is selected. Exact rejected mathematical form and erosion are preserved in `selected_family_math.md`; `margin_loss_spec.md` explicitly prohibits treating it as a ready training label. Mathematical erosion guarantees only membership in the fitted inequalities, not closed-loop success. Topology may be state-dependent, but neither invariance nor actual topology transitions have been demonstrated.','',
 '## Runtime, files and next step','',
 f'New continuation attempts:{runtime["new_continuations"]:,}; new physical steps:{runtime["new_physical_steps"]:,}; complete new Q64:{runtime["complete_new_exact_Q64"]:,}. Rollout execution window approximately{runtime["rollout_execution_window_seconds"]/3600:.2f}hours, including inter-stage analysis and excluding part of initial setup. Maximum6GPU shards; maximum12 allocated CPU threads only on verified idle server. Observed GPU memory peak{runtime["GPU_memory_peak_observed_MiB"]}MiB; observed GPU-worker RSS peak{runtime["GPU_worker_RSS_peak_observed_MiB"]/1024:.2f}GiB; observed RAM headroom at least{runtime["minimum_RAM_available_observed_GiB"]:.1f}GiB. Snapshots are sampled peaks, not guaranteed lifetime maxima; Slurm accounting unavailable. Training time0.','',
 'Observed-only six-panel figures are in `figures/<state_id>.svg`, including `figures/T0_WIDE_perm00_ep0082.svg` and `figures/DB_T0_0.svg`. Blank space is UNKNOWN; interpolation is not colored as verified success. All detailed point/path/interval/geometry tables are indexed by `manifest.json`.','',
 '**Single next scientific step:** '+NEXT,'',
 'No new controller, new base Flow, or follow-on geometry branch was started.']
 (HERE/'final_report.md').write_text('\n'.join(lines)+'\n')
 anomaly=json.load(open(HERE/'acquisition_role_anomaly.json'));anomaly.update(status='RESOLVED_IN_DERIVED_EVIDENCE_AND_FINAL_REPORT',corrected_metrics='corrected_frozen_model_gates.json',raw_Q64_changed=False,original_frozen_models_changed=False,old_saturation_conclusion='RECOMPUTED_ON_MANIFEST_CORRECTED_PRIMARY_HOLDOUT_STILL_SECTION20A',corrected_fit='fitting_failure_minimal_polyhedral_corrected_roles_r8.json');dump('acquisition_role_anomaly.json',anomaly)
 ws=json.load(open(HERE/'working_state.json'));ws.update(goal_status='COMPLETE_SCIENTIFIC_SATURATION',current_stage='FINAL_HANDOFF',next_action='None. Do not train or start next experiment without a new request.',final_decision=CLASS,saturation_status='SECTION20A_MET_AFTER_MANIFEST_ROLE_CORRECTION',candidate_status={'selected':False,'best_rejected_tag':TAG,'fresh_false':5,'fresh_n':72},live_job_observation=[],final_report='final_report.md',unresolved_questions=['True connectedness/topology beyond finite links','Unresolved failure pockets','Whether a different future analytic family can be reliable','State/scenario population generalization']);dump('working_state.json',ws)
 hs=json.load(open(HERE/'hypothesis_status.json'))
 for h in hs:
  h['automatic_next_test']=None
  if h.get('family') and h.get('family')!='minimal_polyhedral_support_quadratic_exclusion':
   h['prior_search_status']=h['current_status'];h['current_status']='FROZEN_REJECTION_SEARCH_CLOSED';h['next_discriminating_test']='No automatic rerun. See final_report and the single recommended future experiment.'
  if h.get('family')=='minimal_polyhedral_support_quadratic_exclusion':h.update(current_status='REJECTED_PROSPECTIVE_PRECISION_SEARCH_SATURATED',key_support=best,key_failure={'fresh_false':5,'n':72},next_discriminating_test=NEXT,latest_gate_file='corrected_frozen_model_gates.json')
 dump('hypothesis_status.json',hs)
 history=json.load(open(HERE/'search_history.json'))
 for r in history['rounds']:
  if r['id']==TAG:r.update(status='REJECTED',false=5,n=72)
  gg=[]
  for batch in r.get('jobs',{}):
   gp=HERE/f'targeted_probe_rounds/{batch}/gate.json'
   if gp.exists():gg.append(json.load(open(gp)))
  if gg and all(g.get('complete') for g in gg) and any(g.get('false_inclusions',0)>0 for g in gg):r['status']='REJECTED_BY_COMPLETED_PROSPECTIVE_VALIDATION'
 history['final_stop']=dict(reason='Section20A',audit='scientific_saturation_audit.json',integrity_correction='acquisition_role_anomaly.json',selected_family=None,training=False);dump('search_history.json',history)
 evidence=json.load(open(HERE/'evidence_index.json'))
 for name in ['final_report.md','final_decision.json','scientific_saturation_audit.json','acquisition_role_anomaly.json','corrected_frozen_model_gates.json','runtime_statistics.json','extent_piece_diagnosis_corrected_roles.json']:
  evidence['FINAL:'+name]={'path':str(HERE/name),'sha256':sha(HERE/name)}
 dump('evidence_index.json',evidence)
 files=[p for p in HERE.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.md','.py') and p.name!='manifest.json']
 manifest=dict(task='ORTHOFLOW3_GENERAL_BASIN_GEOMETRY_DISCOVERY_V1',status='COMPLETE_SCIENTIFIC_SATURATION',classification=CLASS,selected_family=None,training_runs=0,basis_sha256=SHA,final_inventory_sha256=sha(HERE/'exact_q64_inventory.csv'),raw_records_and_frozen_models_preserved=True,
 authoritative_recall='corrected_frozen_model_metrics.csv; corrected_frozen_model_gates.json',deprecated_statistics='Historical family_metrics/gates remain frozen provenance but their validation-substring primary-recall figures are superseded by manifest-corrected metrics. Initial invalid dispatch tag and incomplete inventory tag are separately invalidated.',
 files={p.name:sha(p) for p in files},raw_evidence_index='cache_provenance.json',acquisition_manifests='targeted_probe_rounds/*/manifest.csv',no_automatic_followup=True)
 dump('manifest.json',manifest);print(json.dumps({'classification':CLASS,'retained_false':5,'n':72,'selected':False,'training':False,'report':str(HERE/'final_report.md')}))
if __name__=='__main__':main()
