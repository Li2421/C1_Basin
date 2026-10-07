# Double-Bottleneck eta basis comparison

## 1. P0 basis redundancy audit

P0 is exactly `eta_g bound(goals-positions) + eta_s u_safe + eta_r mean_j bound(p_i-p_j)`.
The row bound preserves direction and caps norm at 0.5 m/s; relations point away from other agents and are bounded before averaging over three others. At N=2 this is the single-opponent basis. Both projections retain their original constraints and speed limits.

The pilot was sampled uniformly without P0-positive/negative labels: 12 of 61 targets, 12 of 24 controls. Four uniform action-state indices per trajectory yielded 96 baseline anchors. The same four-anchor rule was applied to all three stored eta-success traces (12 states). These stored eta-success traces were selected in the earlier study as representatives, so that stratum is descriptive and not population-representative.

| Trajectory class | States | Median per-agent goal/Flow cosine | Median joint goal/Flow cosine | P0/P1 raw rank median |
|---|---:|---:|---:|---|
| baseline_success | 48 | 0.9687 | 0.7069 | 3/3 |
| safe_timeout | 48 | 0.9701 | 0.6940 | 3/3 |
| successful_eta_corrected | 12 | 0.9320 | 0.6919 | 3/3 |

All pair cosines, per-agent 2D and joint 8D Gram matrices, singular values, rank thresholds, and effective condition numbers are saved per state in `offline/basis_state_audit.jsonl`. Near-zero vectors are excluded from cosine statistics, rather than treated as aligned.

## 2. Post-projection effective-rank audit

Central differences use 0.015625 times each domain width at eta=0 and registered successful P0 eta. Eta=0 is outside the positive-goal search interval; these unconstrained local probes characterize the operator at baseline and do not add rollout search points. The numerical rank threshold is max(0.001, 0.01*sigma_max). Finite differences describe scale-dependent responses of a nonsmooth projection, not a guaranteed analytic Jacobian.

- baseline_success: median executable rank at zero P0/P1 = 3/3; effective condition number = 4.802/4.368.
- safe_timeout: median executable rank at zero P0/P1 = 3/3; effective condition number = 4.451/4.258.
- successful_eta_corrected: median executable rank at zero P0/P1 = 3/3; effective condition number = 5.404/4.117.

At successful P0 eta references, median executable rank is 2. Per-dimension removed perturbation fractions and executable response cosines are in `extended_basis_statistics.json`. P1 does not demonstrate an increase in median rank at eta=0; its benefit must not be described as simply restoring a missing third rank.

At eta=0, median executable goal/Flow cosine on baseline-success states is 0.7117 for P0 and approximately zero for P1; on timeout states it is 0.6592 versus approximately zero. Around the successful P0 eta references the median is -0.2911, showing that active projection can change direction relationships rather than merely rescale parallel columns. Rank-zero responses are logged explicitly; their cosine is undefined in physical terms even though the stored numerical cosine matrix uses zero for null columns.

## 3. P1-OrthoFlow3 definition and energy comparison

For each agent `r = u_safe - B_goal * dot(u_safe,B_goal)/(||B_goal||² + epsilon²)`, epsilon = 1e-6*max_speed. Then `g = eta_g B_goal + eta_perp c*r + eta_r B_rel`.

The globally fixed c = 3.303687238760696 was computed before P1 rollouts from the 96 baseline anchors: safe-Flow row RMS 0.259393 m/s divided by residual RMS 0.0785162 m/s. No per-state normalization is used. The smooth zero-goal fallback approaches u_safe, with no invented direction. An exactly goal-aligned nonzero Flow has only a regularization-sized residual. No sampled scaled residual was <=1e-6 m/s; relative near-degeneracy fractions are also saved.

The same coefficient domain [0.5,1.25] × [-0.5,0.5] × [0,0.75] and same 256 points were used. RMS calibration matches a typical unit basis scale, not a pointwise energy cap. Scaling can amplify individual residuals; the experiment tests orthogonalization plus this fixed calibration together. It cannot attribute all improvement to orthogonality alone.

Selected-center trial step-weighted raw/executed mean norms: P0 0.5275/0.2016 m/s; P1 0.5541/0.1320 m/s. Second-projection removal norms: 0.4467 versus 0.4869. These use different trajectories and selected centers and are descriptive, not matched-state causal comparisons.

## 4. Pilot P0 vs P1 comparison

Common-256 single-seed basin existence: P0 10/12, P1 12/12. Median positive fractions: 1/256 versus 6/256. Cached P0 rollouts have identical states, seeds, eta points, horizon, and pipeline. Diagnostic P0 arithmetic passed 100 exact canonical-equivalence checks.

## 5. Seed-robustness comparison

Candidate rule: all successful global points when <=8; otherwise deterministic top eight by successful 8-neighbor count, distance to zero, eta index. Both use seeds 2001–2016. Thus the maximum effort per episode is identical; actual candidate counts vary with basin size.

Pilot median Q_max: P0 0.1875, P1 1.0. P0 counts at Q>=0.25/0.50/0.75: 5/0/0; P1: 12/12/12. This triggered the pre-registered full study via median gain >=0.15 and >=3 newly robust cases.

## 6. Local basin-width comparison

Centers with Q_max>=0.50 receive exactly 16 local Sobol points, normalized radius 0.05 per coordinate, clipped to the fixed domain, and seeds 3001–3008. P0 has no eligible centers. P1 has 61 eligible centers. The median fraction of local points guaranteed to have Q>=0.50 is 1.0000; its possible upper-bound median is 1.0000. Median per-episode local Q lies between 1.0000 and 1.0000. This independent seed set checks selection optimism of Q_max, although these remain measured local neighborhoods, not continuous volume certificates.

Exactly 1 of 7,808 registered local rollouts is `NUMERICAL_SOLVER_FAILURE`: `fresh_untouched_test|042`, local point 3, seed 3004, at step 601 in the second projection. After the initial rejection, all three authorized additional invocations reused byte-identical state and nominal action, the same problem, constraints, tolerances, controller, eta, and seed; all failed the unchanged certificate. It is excluded from task outcomes. Local statistics above use strict success lower/upper bounds, so it is never imputed as success or task failure. The lower and upper threshold classifications coincide for this local point, hence the population median is unaffected.

Clipping places 6.66% of local points on a domain boundary. Thus the local success fraction describes the prescribed clipped perturbation law, not an unbiased three-dimensional volume estimate.

## 7. Control preservation

Every distinct full-study P1 center was tested on all 24 frozen controls with seeds 4001–4004; identical center tuples were reused. Median preservation probability over selected episode centers: P0 0.1875, P1 1.0000. Outcomes: {'success': 278, 'timeout': 10}. No eta was optimized for controls. Preservation across new seeds is not a paired estimate of damage versus eta=0 under those same new seeds; controls were successful under their original frozen seeds.

On the matched 12-control pilot subset, median preservation across available selected centers is P0 0.2083, P1 1.0000. Episodes with no successful global candidate have no center and are excluded from this center-preservation statistic.

## 8. Full P1 results

| Measure | P0 | P1-OrthoFlow3 |
|---|---:|---:|
| Observed basin /61 | 38 | 61 |
| Median positive sampled fraction | 1/256 | 5/256 |
| Median Q_max among basin-positive episodes | 0.21875 | 1.00000 |
| Q_max>=0.50 /61 | 0 | 61 |
| Q_max>=0.75 /61 | 0 | 61 |

New rollout records: 32256; valid scientific task outcomes: 32255; numerical solver failures: 1; collision rollouts among valid outcomes: 0. Each registered tuple has a saved record. Counts exclude copied cache rows.

Two global eta points (indices 37 and 73) each cover 60/61 targets under the original episode sampling seeds. The selected centers use only indices 37, 73, and 217. This is evidence of shared structure, but is not a cross-seed certificate for one universal eta. Coarse global fractions remain small even when local neighborhoods around selected centers are robust.

## 9. Expert executable-action span diagnostic

The same 96 uniform baseline anchors were queried with the unchanged expert; 12 successful continuations were available. No failed anchor was replaced. Equal 1024 Sobol candidates plus bounded Powell refinement (400 evaluations) fit each representation inside the same domain. These are best-found residuals, not certified global minima.

Median executable-action residual: eta=0 0.6087755985753099; P0 0.576806572241912; P1 0.5852786643520393. Recovery failures and all residuals are retained. This conditional, small recoverable subset and one selected expert mode cannot establish full joint-action span sufficiency. Low-dimensional correction need not reproduce an arbitrary expert action to improve task completion.

Offline expert re-query restores sampled positions and velocity with fresh monitor history and an 850-step continuation budget. It tests local recoverability and action fit, not completion within the original episode's remaining time. All scientific controller rollouts retain the original 850-step horizon and terminal semantics.

## 10. Final decision and scientific answers

**USE-ORTHOFLOW3**.

Q1: Goal and Flow bases overlap strongly per agent; joint rank nevertheless usually remains three.
Q2: Their executable effects are compressed further at nonzero successful eta; the saved response matrices quantify overlap and clipping separately.
Q3: Orthogonalization improves conditioning but does not increase median executable rank at zero.
Q4: P1 materially improves measured seed robustness in the matched pilot and the triggered full study.
Q5: The independent-seed local tests establish neighborhood robustness under strict lower-bound accounting; P0 has no eligible robust center. One numerical failure does not change its local point's Q>=0.50 classification.
Q6: Preservation is measured on all 24 controls and reported above; it is not used for selecting eta.
Q7: P1's success removes the immediate need to choose between agent-specific coefficient expansion and missing-direction redesign. The intervention changes the permissible shared-coefficient field as well as conditioning: per-agent removed parallel Flow components have different magnitudes and signs, so this is not a mere invertible change of the original three shared coordinates.

## 11. Single recommended next step

Freeze this OrthoFlow3 definition and test cross-state reuse of its small set of robust centers on fresh held-out stochastic episodes in a later study, before deciding whether a learned state-to-eta mapping is needed. No learning or representation expansion was started here; canonical P0 is retained unchanged pending review.

## Reproducibility and limitations

Source/data/environment/projection/Toy hashes match the frozen reference and the full regression suite passed. Diagnostic source and all jobs/results remain isolated in this directory. The initial PREREGISTRATION `registered_at` literal incorrectly states midnight September 27; tool execution records establish it was written before the September 26 19:08 basis audit and 19:10 pilot launch. This timestamp clerical error is preserved and disclosed; it must not be treated as a cryptographic external preregistration. No sampling, scale, or decision rule was changed using P1 outcomes. State-audit finite-step removal logging was corrected during analysis to measure around its stated center; rollout/controller code was unaffected. CPU execution was used because CUDA initialization failed; evaluation semantics match the preceding CPU studies.

![Robustness comparison](figures/robustness_comparison.svg)
