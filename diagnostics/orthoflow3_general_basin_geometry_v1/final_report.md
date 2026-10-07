# OrthoFlow3 general Basin geometry discovery — final report

**Classification: NO_LOW_COMPLEXITY_ANALYTIC_FAMILY_EVIDENCED.** No general analytic family selected; no controller/network training.

This means the tested evidence-driven search did not produce a family passing every gate. It does not prove analytic representability impossible. The stop is the predeclared two-round saturation criterion, not a time/rollout cap, the number of hypotheses, or the eight-face fitting limit.

## Evidence, conditioning and integrity

Reused1,131 prior exact-Q64 state–eta tuples (72,384 continuations) plus976 compatible lower-seed continuation records. Added1,180 complete exact-Q64 tuples; final inventory2,311 tuples,1,608 B63 across44 true-t0 source states. Twelve states satisfy frozen density criteria:8 ToyGiveWay and4 joint4-agent DoubleBottleneck. All40 original Toy source groups and four DB groups remain fixed.

Three historical controller TEST outputs lie outside E_bridge (one B63,two non-B63). They remain explicitly flagged in the provenance inventory but are excluded from geometry summaries:2,308 in-domain exact tuples,1,607 B63. No new probe was outside E_bridge. One old out-of-domain negative was redundantly constrained by earlier analytic fits; it was already rejected by the domain inequality and cannot explain retained false inclusion. Original frozen fits are preserved.

The authoritative basis hash is `51c2cbcc235a5ba9db6f73b3d449e0bc631bfbb0e1ed3184c4fed97c8fa01c38`. Eta is selected once, latched, and both projections and timestep-wise basis recomputation remain unchanged. Scenario-specific Flow policies are frozen; same eta basis semantics does not mean identical features or identical dynamics. DB current Flow/h is fixed before varying future seeds.

One DB projection/numerical attempt is quarantined; its eta has only63 valid seeds and is NOT Q64/B63. No replacement seed or controller tolerance adjustment was used. Historical batch-feature aliases were resolved by exact replay within the original1e-10 protocol tolerance (maximum h difference2.60e-15); eta dedup remains exact float64. No conflicting exact outcomes.

An end-of-run audit found that the Toy rollout wrapper overwrote the intended validation phase suffix. This had contaminated historical *primary recall* denominators and meant only hash-fit, rather than all promoted, validation positives entered some refits. Raw outcomes and prospective validation are unaffected. Frozen manifests now authoritatively annotate roles; all validation points are excluded from the corrected primary recall. Historical raw data/models are preserved, not silently rewritten. `corrected_frozen_model_metrics.csv` and `corrected_frozen_model_gates.json` supersede historical recall figures. See `acquisition_role_anomaly.json`.

## Geometry and topology answers

**One large connected component? Not established.** No dense state meets the requested90percent validated-link criterion; strict sampled-link largest fractions range5.3–17.2percent. Sparse validated links do not prove fragmentation. kNN connectivity is not substituted for rollout-supported paths.

The newly tested star segments had79/96 successful interior probes;25/32 segments had all three quarter-points B63. This supports many filled directions but does not establish one star kernel or global star-convexity. The earlier88/96 interpolation result used a different selected batch. Five targeted two-leg detours around known failed straight links yielded16/30 B63 probes and0/5 fully successful routes; this rejects those routes, not every possible connecting path.

**Enclosed holes versus intrusions:** zero enclosed holes are confirmed; absence is not proved. Seven of19 sampled failure-to-boundary routes stayed non-B63; twelve were interrupted by success. In the explicitly investigated internal ep0195 pocket,3/6 axial routes reach the domain boundary through sampled failures. Resolved cases favor boundary intrusions/notches, but the evidence cannot quantify that most failure volume is of this type.

**Conditional intervals:** Toy eta1 has8/9 single-run lines and one two-success-run counterexample (`FSFFSSS`); eta2 has9/9 single-run lines; eta3 has8/9 single-run lines and one all-failure line. DB eta1 and eta2 each have16/16 single-run lines; eta3 has10 single-run and6 all-failure lines. These are discrete selected line scans, not proofs of conditional convexity or absence of narrow intervening gaps.

**Common core:** eta=(0.7421875,0.46875,0.7265625) is B63 on35/40 Toy states;38/40 have Q64>=0.90; mean Q64=0.9828125. No tested eta is B63 on all40. This is a genuine high-coverage finite core candidate, not training degeneracy, but no positive-volume universal core is certified. All10 Toy modes were non-B63 on all4 DB states (0/40 robust transfers). Shared family must allow large state/scenario displacement; it cannot mean one universal eta region.

**Common-core-minus-exclusions:** qualitatively supported within Toy by frequent shared solutions and sampled failure intrusions. The tested low-complexity realizations fail precision/recall gates. Do not equate this qualitative explanation with a validated analytic family.

## Cross-state and cross-scenario signatures

| Scenario | States / adequate | Exact eta / B63 | Median diameter | Tangent1/normal span | Corrected full / retained recall* |
|---|---:|---:|---:|---:|---:|
| DoubleBottleneck_4A | 4 / 4 | 331 / 159 | 0.836 | 6.62 | 0.967 / 0.465 |
| ToyGiveWay_2A | 40 / 8 | 1977 / 1448 | 1.835 | 2.23 | 0.908 / 0.443 |

*Best rejected F3 instance, common retrospective non-fitting positive panel; not population recall.

Both scenarios show anisotropic support with narrower directions and substantial conditional intervals. Native location, orientation, extent, thickness and boundary exclusions vary. PCA/anisotropic-scale alignment was computed with sign ambiguity removed, but on nonuniform positive clouds only; similarity does not establish identical occupancy. No extra shear was justified. Exact Basin equality, canonical shape identity, and topology invariance are not established. Broad morphology recurs; parameters and some sampled line patterns differ.

| Dense state | B63 / non-B63 | Diameter | Anisotropy | Strict linked fraction | Nearest opposite label |
|---|---:|---:|---:|---:|---:|
| DB_T0_0 | 38 / 45 | 0.615 | 4.28 | 0.132 | 0.025 |
| DB_T0_1 | 40 / 43 | 0.805 | 6.81 | 0.125 | 0.025 |
| DB_T0_2 | 38 / 45 | 0.908 | 7.10 | 0.132 | 0.0209 |
| DB_T0_3 | 43 / 39 | 0.867 | 6.42 | 0.116 | 0.025 |
| T0_WIDE_perm00_ep0082 | 117 / 68 | 1.472 | 2.60 | 0.077 | 0.003125 |
| T0_WIDE_perm01_ep0217 | 169 / 20 | 1.954 | 1.99 | 0.172 | 0.001563 |
| T0_WIDE_perm02_ep0182 | 169 / 20 | 1.886 | 2.44 | 0.053 | 0.0125 |
| T0_WIDE_perm03_ep0205 | 163 / 27 | 1.882 | 2.73 | 0.153 | 0.00625 |
| T0_WIDE_perm04_ep0195 | 143 / 65 | 1.947 | 2.81 | 0.084 | 0.001563 |
| T0_WIDE_perm05_ep0179 | 151 / 48 | 1.592 | 1.76 | 0.119 | 0.003125 |
| T0_WIDE_perm06_ep0074 | 142 / 60 | 1.788 | 2.01 | 0.120 | 0.004687 |
| T0_WIDE_perm07_ep0139 | 165 / 47 | 1.701 | 1.80 | 0.109 | 0.00625 |

No true success-volume fraction is inferred from adaptive samples. Boundary sharpness is represented by measured opposite-label spacing, not an assumed differentiable surface. Component/hole counts and topology beyond validated paths remain unresolved.

## Analytic families and falsification

| Family / frozen fit | States | Corrected full recall | Retained recall | Current known retained non-B63 |
|---|---:|---:|---:|---:|
| asymmetric_slab_with_notches / after_round1_complete | 10 | 0.626 | 0.286 | 4 |
| common_core_minus_exclusions / after_round1_complete | 10 | 0.625 | 0.410 | 63 |
| conditional_band_with_cuts / after_round1_complete | 10 | 0.557 | 0.329 | 16 |
| semialgebraic / after_round1_complete | 10 | 0.876 | 0.500 | 67 |
| star_convex_with_exclusions / after_round1_complete | 10 | 0.453 | 0.309 | 14 |
| bounded_quadratic_with_exclusion / bounded_r2 | 11 | 0.913 | 0.522 | 40 |
| bounded_quadratic_with_exclusion / bounded_refined_r3 | 12 | 0.901 | 0.531 | 30 |
| polyhedral_support_quadratic_exclusion / polyhedral_r4 | 12 | 0.945 | 0.474 | 24 |
| minimal_polyhedral_support_quadratic_exclusion / minimal_polyhedral_r5_corrected | 12 | 0.902 | 0.457 | 15 |
| minimal_polyhedral_support_quadratic_exclusion / minimal_polyhedral_refined_r6 | 12 | 0.918 | 0.443 | 11 |
| minimal_polyhedral_support_quadratic_exclusion / minimal_polyhedral_refined_r7_stable | 12 | 0.923 | 0.443 | 5 |

These are unchanged frozen parameters, re-evaluated on the current manifest-corrected non-fitting panel. Later acquired negatives can falsify an older instance; this is not its original fitting loss. Family A uses up to4 boundary-open caps; B a native band plus cuts; C a PCA asymmetric slab plus cuts; D an ellipsoidal radial body plus cuts; E one quadratic inequality. None satisfies all gates.

Synthesized F1 added bounded concave-quadratic support plus one quadratic exclusion because the first unbounded polynomial extrapolated beyond successful support. F2 used at most3 supporting halfspaces plus one quadratic because smooth bounded support still admitted unsupported corners. F3 allowed optimal minimum support facets up to8 plus one quadratic: measured2–8 support directions motivated the explicit complexity exception (maximum33 meaningful parameters,9pieces). No neural implicit field, dense mesh, kNN membership or point-cloud lookup was introduced.

| Prospective validation attempt | Confirmed retained non-B63 / tested |
|---|---:|
| Single quadratic | 21/42 |
| F1 bounded, first | 15/48 |
| F1 bounded, refined | 22/72 |
| F2 three-face support | 16/72 |
| F3 minimal support, first | 7/72 |
| F3 refinement6 | 7/72 |
| F3 refinement7 | 5/72 |

The last two F3 attempts reuse the SAME24 independent DB validation points only after exact mathematical-instance and fitting-input equality checks; each adds48 genuinely new Toy points. Thus these rows are not144 additional independent observations. The four unchanged DB instances passed24/24; the five last failures occur in Toy (ep0082:3, ep0074:1, ep0139:1). Two are inside the prior successful-point hull; three are outside.

The best tested model has corrected median full recall0.923, retained recall0.443;11/12 states have full recall>=0.50; robust transfer capture0.838. These favorable recall metrics do NOT compensate for5 retained non-B63 points. It is not a reliable margin label.

After role correction, a new intended-protocol fit was attempted without any new rollout. Preserving all promoted fit positives while excluding exterior negatives requires9 supporting faces for ep0082; the frozen8-face fitter therefore returned unresolved. This is a fitting-recipe limitation, NOT a lower bound at70percent recall or proof that9faces are unreasonable. The partial artifact is explicitly unselected.

## Scientific saturation and limitations

On the SAME correctly excluded primary holdout, the last three tested fits have full-recall medians0.9022, 0.9183, 0.9232 and retained medians0.4571, 0.4429, 0.4429. Successive full-recall gains are below5percentage points; retained recall does not improve. Prospective false-inclusion gains are0 then2.78percentage points. Morphology medians and inferred topology classes show no material new change. Section20A operational saturation is therefore met after correcting the grouping error.

This is evidence-limited saturation, not exhaustive optimization over all analytic families. Future targeted evidence or a new mathematically justified form could change the conclusion. Repeated adaptive family comparison and a small fresh panel do not provide population calibration. Empirical B63 uses the fixed64 matched futures, not a confidence-bound theorem for the true success probability.

No general family is selected. Exact rejected mathematical form and erosion are preserved in `selected_family_math.md`; `margin_loss_spec.md` explicitly prohibits treating it as a ready training label. Mathematical erosion guarantees only membership in the fitted inequalities, not closed-loop success. Topology may be state-dependent, but neither invariance nor actual topology transitions have been demonstrated.

## Runtime, files and next step

New continuation attempts:74,608; new physical steps:40,853,451; complete new Q64:1,180. Rollout execution window approximately3.47hours, including inter-stage analysis and excluding part of initial setup. Maximum6GPU shards; maximum12 allocated CPU threads only on verified idle server. Observed GPU memory peak3696MiB; observed GPU-worker RSS peak10.63GiB; observed RAM headroom at least110.7GiB. Snapshots are sampled peaks, not guaranteed lifetime maxima; Slurm accounting unavailable. Training time0.

Observed-only six-panel figures are in `figures/<state_id>.svg`, including `figures/T0_WIDE_perm00_ep0082.svg` and `figures/DB_T0_0.svg`. Blank space is UNKNOWN; interpolation is not colored as verified success. All detailed point/path/interval/geometry tables are indexed by `manifest.json`.

**Single next scientific step:** Target a preregistered small conditional slice around ep0082's known internal non-B63 region to distinguish a few sharp, boundary-connected intrusion branches from genuinely interleaved feasibility; do not train or merely add unconstrained fitting capacity.

No new controller, new base Flow, or follow-on geometry branch was started.
