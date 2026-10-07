# Double-Bottleneck broad initial-state coverage study

Date: 2026-09-24  
Scope: data-coverage-only scaling of the unchanged Toy-sourced joint Stage-I MACFlow `p(u | x)`.

## Summary

Broad, independently sampled initial-state coverage has a clear causal effect, but the resulting policy is not yet a reliable Stage-I baseline. On the frozen untouched test, increasing the number of training initial states from 24 to 72 to 144 changed success from **0/96 to 10/96 to 26/96**, reduced mean timestep-0 support distance from **0.1049 to 0.0870 to 0.0806**, and reduced whole-rollout OOD fraction from **89.64% to 49.63% to 31.49%**. S-Large also surpassed the diagnostic targeted D2-T result on this test (26/96 versus 21/96) without targeted acquisition.

This is not a pass. S-Large still produced 50 wall collisions, 5 agent collisions, and 15 timeouts; every untouched initial state remains outside the fixed strict support threshold at timestep 0. The evidence supports continued globally uniform initial-state scaling, not an architectural conclusion and not local failure targeting.

![Coverage scaling](figures/coverage_scaling.svg)

## 1. Pre-registered initial-state distribution

The protocol was saved before generation in [`PREREGISTRATION.json`](PREREGISTRATION.json), SHA-256 `35a7ac45953fa84ddf3f439c8698748b2d9fc81cffcc2b3ed2cd1dc7fc9a1700`.

For each of the three existing regimes, the same side-exchange-symmetric global law was used:

- independent common travel-coordinate offsets for each side: `Uniform[-0.10, 0.10] m`;
- within-side travel spacing: `Uniform[-0.05, 0.05] m`, applied as `+/- 0.5` spacing;
- independent common lateral offsets for each side: `Uniform[-0.015, 0.015] m`;
- within-side lateral spacing: `Uniform[-0.010, 0.010] m`, applied as `+/- 0.5` spacing;
- per-agent parallel velocity: `Uniform[-0.05, 0.05] m/s` in the local travel coordinate;
- per-agent lateral velocity: `Uniform[-0.02, 0.02] m/s`.

The laws and ranges were identical for both sides. No failure seed, phase, wall distance, bottleneck label, goal label, or previous diagnostic entered the sampling probability.

The root seed label was `double_bottleneck_initial_coverage_v1|20260924`. Each split/regime used a separate PCG64 stream seeded by the first unsigned 64 bits of `SHA256(root|initial|split|regime)`; exact numeric seeds are stored in the final manifests. A candidate was accepted only if the existing centralized expert completed all eight coordination hypotheses safely. A rejected candidate would have been recorded once and replaced by the next draw from the same global IID stream, with no local resampling. In fact, all 192 sampled states across train, validation, and test were accepted on their first draw; there were no expert-generation rejections.

For each successful expert trajectory, the same generic local recovery rule was retained:

- exactly 64 unique anchor indices selected uniformly by a trajectory-specific permutation;
- per-coordinate position perturbation `Uniform[-0.02, 0.02] m`;
- per-coordinate velocity perturbation `Uniform[-0.2, 0.2] m/s`, followed by the common `0.5 m/s` radial speed bound;
- one stored recovery transition after a full successful terminal recovery check;
- failed/invalid queries would be recorded and discarded without replacement.

All 73,728 train recovery queries and all 12,288 validation recovery queries succeeded. Their naturally induced phase distribution was measured only after construction; it was not rebalanced. For S-Large it contained 32,386 waiting/yielding, 11,070 coordination-transition, 8,243 final-approach, 6,763 bottleneck-traversal, 5,958 second-bottleneck, 5,699 chamber, 1,978 near-goal, 904 initial-approach, and 727 first-bottleneck-approach anchors.

The realized train/test draws are consistent with the same intended family. For example, mean initial speed was 0.0286 m/s in S-Large train and 0.0276 m/s in test; mean absolute longitudinal offset was 0.0507 m versus 0.0514 m; mean absolute lateral offset was 0.00803 m versus 0.00820 m. No post-hoc balancing was applied.

## 2. Train/validation/test separation

The pools were independently generated and frozen before any model training:

| Pool | Independent states | Expert trajectories | Nominal transitions | Purpose |
|---|---:|---:|---:|---|
| Train maximum pool | 144 | 1,152 | 846,430 | Nested S-Small/Medium/Large training |
| Validation | 24 | 192 | 141,180 | Loss monitoring only; no checkpoint search |
| Untouched test | 24 | 192 | 141,076 | Primary result |

Each state has all eight genuinely distinct, successful crossing-order trajectories: four LTR-first and four RTL-first variants. Thus the maximum train pool contains 144 examples of each of the eight realized mode signatures; validation and test contain 24 of each.

Initial-state hashes have zero intersection for train–validation, train–test, and validation–test. Test states received no recovery perturbations, were never used for checkpoint selection, and were not inspected individually until all seven frozen model evaluations had completed and `comparison.json` had been written. The old repeatedly inspected 12 rollouts remain development diagnostics only.

Final pool manifests:

- [`S-Small`](manifests/s_small.json)
- [`S-Medium`](manifests/s_medium.json)
- [`S-Large`](manifests/s_large.json)
- [`validation`](manifests/validation.json)
- [`untouched test`](manifests/untouched_test.json)

## 3. Dataset scaling table

The training pools are strictly nested, use the same 64 recovery anchors per expert trajectory, and differ only in independent initial-state count and the resulting proportional data volume.

| Dataset | States (per regime) | Expert trajectories | Nominal transitions | Recovery transitions | Total transitions | Updates | Approx. sample exposures |
|---|---:|---:|---:|---:|---:|---:|---:|
| S-Small | 24 (8) | 192 | 141,130 | 12,288 | 153,418 | 39,000 | 65.08 |
| S-Medium | 72 (24) | 576 | 423,380 | 36,864 | 460,244 | 116,000 | 64.52 |
| S-Large | 144 (48) | 1,152 | 846,430 | 73,728 | 920,158 | 231,000 | 64.27 |

Expert episode length was 698–772 steps, median 735 and mean 734.75. Every nominal trajectory was successful and collision-free; minimum nominal wall and pair clearances were 0.0533 m and 0.0865 m. Recovery mode-signature agreement was 100%.

The MACFlow implementation remained unchanged: 72D joint condition, 8D joint action, official `ActorVectorField` with hidden dimensions 256×3, 154,632 parameters, conditional flow matching, Adam at `3e-4`, batch 256, 10 Euler integration steps, coordinatewise corpus normalization using the same procedure, and fresh Gaussian base noise at every environment step. Training sampled uniformly from the concatenated nominal and recovery transitions. Updates were pre-registered as `ceil((64 * N / 256) / 1000) * 1000`, so approximate sample exposure—not raw update count—was held constant. The final update was used; neither validation nor rollout performance selected checkpoints.

| Model | Fixed train loss | Fixed validation loss |
|---|---:|---:|
| S-Small | 0.09877 | 0.11424 |
| S-Medium | 0.06791 | 0.06718 |
| S-Large | 0.05134 | 0.05335 |

These losses are secondary endpoints. No architecture, optimizer family, inference rule, explicit mode input, persistent latent, recurrent state, horizon, action chunk, eta logic, or `G_phi` was introduced.

## 4. Initial-state support analysis

Support uses the same fixed diagnostic as the preceding studies: nearest-neighbor distance in the canonical D0-normalized 72D observation space, divided by `sqrt(72)`, with the fixed calibrated q99 threshold `0.0395697929`. Each variant's actual training rows form its nearest-neighbor library; coordinates and threshold remain canonical.

| Model | Training states | Mean x0 distance | Strict x0 OOD | Mean rollout distance | Rollout OOD |
|---|---:|---:|---:|---:|---:|
| S-Small | 24 | 0.10488 | 100% | 0.07961 | 89.64% |
| S-Medium | 72 | 0.08698 | 100% | 0.08221 | 49.63% |
| S-Large | 144 | 0.08065 | 100% | 0.06041 | 31.49% |

The continuous x0 distance decreases monotonically, but the binary x0 metric does not yet cross its deliberately strict threshold for any of the 24 unique test states. Therefore this experiment cannot separate later drift conditional on a supported x0: there are still zero strictly supported untouched starts. What it does establish is a simultaneous reduction in initial distance, rollout OOD, local prediction error, and an increase in long-rollout success.

## 5. Closed-loop results

### Development set

The development set contains the three old diagnostic initial states with four rollout seeds each.

| Model | Success | Wall collision | Agent collision | Timeout | Median steps | Rollout OOD |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 0/12 | 12 | 0 | 0 | 70.5 | 98.96% |
| D1 | 4/12 | 7 | 1 | 0 | 734.5 | 80.53% |
| U-High | 1/12 | 10 | 1 | 0 | 773.5 | 85.32% |
| D2-T diagnostic | 7/12 | 4 | 1 | 0 | 723.5 | 85.68% |
| S-Small | 0/12 | 12 | 0 | 0 | 239.5 | 92.98% |
| S-Medium | 0/12 | 9 | 1 | 2 | 194.0 | 61.15% |
| S-Large | 4/12 | 7 | 0 | 1 | 189.0 | 26.22% |

These old nominal starts are not the primary gate and remain outside the S-model strict x0 support. The weaker development success than D2-T is therefore reported, not optimized against.

### Untouched test — primary result

Each entry uses the same 24 independently sampled test states and four fixed rollout seeds (`101, 211, 307, 401`), for 96 rollouts. Wall and agent collision columns describe collision involvement and can overlap for a single terminal event.

| Model | Success | Wall collision | Agent collision | Timeout | Median steps | Min wall clearance | Min pair clearance |
|---|---:|---:|---:|---:|---:|---:|---:|
| D0 | 0/96 | 88 | 9 | 0 | 73.5 | -0.01845 | -0.01714 |
| D1 | 8/96 | 67 | 21 | 0 | 385.5 | -0.01630 | -0.01424 |
| U-High | 5/96 | 77 | 9 | 5 | 732.5 | -0.01620 | -0.00898 |
| D2-T diagnostic | 21/96 | 56 | 19 | 0 | 742.0 | -0.01492 | -0.01169 |
| S-Small | 0/96 | 90 | 5 | 1 | 71.5 | -0.01569 | -0.01174 |
| S-Medium | 10/96 | 34 | 21 | 31 | 326.5 | -0.01345 | -0.01288 |
| S-Large | **26/96** | **50** | **5** | **15** | 305.0 | -0.01397 | -0.00711 |

![Closed-loop outcomes](figures/closed_loop_outcomes.svg)

S-Large succeeded comparably across all regimes: 9/32 clearly asymmetric, 9/32 weakly asymmetric, and 8/32 near-symmetric. Successful rollouts include both macro directions (8 LTR-first and 18 RTL-first) and six realized passage-order signatures. Thus the poor total success is not explained by complete collapse of a regime or one macro direction, although the direction counts are imbalanced and the sample is too small to claim calibrated mode probabilities.

### Local and finite-horizon diagnostics

| Model | Teacher RMSE | Waiting/transition RMSE | Goal-region RMSE | Wall-directed error | K=100 position RMSE | K=100 collision |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 0.00806 | 0.01015 | 0.00909 | 0.00170 | 0.09437 | 68.75% |
| D1 | 0.01856 | 0.02163 | 0.01957 | 0.00435 | 0.02392 | 0.35% |
| U-High | 0.01792 | 0.02097 | 0.01957 | 0.00483 | 0.01821 | 0.00% |
| D2-T diagnostic | 0.01879 | 0.02178 | 0.02055 | 0.00411 | 0.01775 | 1.74% |
| S-Small | 0.00778 | 0.00986 | 0.00963 | 0.00165 | 0.02789 | 3.47% |
| S-Medium | 0.00638 | 0.00829 | 0.00665 | 0.00110 | 0.01737 | 0.69% |
| S-Large | **0.00518** | **0.00686** | **0.00540** | **0.00102** | **0.01384** | **0.00%** |

For S-Large, K-step position RMSE at K=`1/5/10/25/50/100` was `0.00007/0.00045/0.00136/0.00493/0.00931/0.01384 m`, with zero collisions in all 288 K=100 continuations. This confirms that broad coverage strongly improves the local field and finite-horizon robustness. The gap between this result and 26/96 full-episode success remains the central unresolved issue.

![K-step divergence](figures/kstep_divergence.svg)

## 6. Coverage scaling curve

The preregistered relation is observed in continuous form:

| Independent states | Mean x0 distance | Rollout OOD | Success |
|---:|---:|---:|---:|
| 24 | 0.10488 | 89.64% | 0.0% |
| 72 | 0.08698 | 49.63% | 10.4% |
| 144 | 0.08065 | 31.49% | 27.1% |

This is strong evidence that broad initial-state coverage is a primary missing ingredient. It is not evidence that initial-state shift is solved: x0 OOD remains 100%, rollout OOD remains 31.5%, and success is far below a practical reliability gate. The curve has not saturated over the tested range, so Case A/B—not Case C/D—best describes the evidence: initial support improves, while both residual initial shift and later drift remain.

## 7. Comparison with U-High and D2-T

U-High densified recovery around too few nominal trajectories. On the new test it obtained 5/96 success, x0 distance 0.1113, and 85.86% rollout OOD. S-Large retained the same globally uniform recovery density per trajectory but broadened nominal initial states, obtaining 26/96, x0 distance 0.0806, and 31.49% rollout OOD. This directly supports the broad-coverage hypothesis over merely adding more local anchors to the old trajectories.

D2-T is a diagnostic targeted acquisition result, not an allowed canonical strategy. It obtained 21/96 on the same new test; S-Large reached 26/96 without using failures or phase labels to acquire data. The difference is modest, but it shows that uniform broad sampling can at least close and slightly reverse the targeted result. D2-T still performs better on the repeatedly inspected development set (7/12 versus 4/12), which is consistent with D2-T's deliberate emphasis on that old distribution and must not be hidden.

## 8. Failure analysis

This analysis was run only after `comparison.json` was frozen. It did not query the expert, generate data, retrain, or alter any sampling rule.

S-Large's 96 primary rollouts ended in 26 successes, 55 collision terminals, and 15 timeouts. For each of the 70 failures, phase was assigned from the nearest state among all eight expert trajectories for the same untouched initial state. Terminal-nearest phase counts were:

- waiting/yielding: 29;
- coordination-mode transition: 23;
- near-goal termination: 15;
- final goal approach: 3.

A conservative “clear divergence” was defined as joint-position RMS to every same-family expert state exceeding 0.08 m for five consecutive steps. Seventeen failures met this definition, with median first divergence at step 351: seven at coordination transitions, one in waiting/yielding, five at final approach, and four near goal. Only 4/17 (23.5%) were inside the strict 72D support threshold at that first clear divergence. The other 53 failures can collide or time out while remaining within 0.08 m of some state from one of the eight expert trajectories; that permissive positional criterion must not be mistaken for full observation/action support.

Post-hoc Spearman correlations between per-state success (four seeds) and the preregistered continuous initial variables were weak and non-significant for S-Large: absolute rho ranged from 0.068 to 0.215, all `p >= 0.31`. This small test does not identify a single failed velocity, spacing, lateral-offset, or asymmetry subfamily. Failures occur across all three regimes, which argues against a new hand-selected family or targeted shell.

The residual evidence therefore has two parts:

1. strict initial and rollout support is still incomplete, so representation/capacity is not yet isolated;
2. full-episode failures remain concentrated in coordination/waiting and terminal phases even though K=100 behavior is excellent, so a future dense-support result that still fails would be meaningful evidence for a model limitation—but this experiment has not reached that condition.

Full per-rollout post-hoc records are in [`posthoc_analysis.json`](posthoc_analysis.json). Frozen raw comparisons are in [`evaluation/comparison.json`](evaluation/comparison.json).

## 9. Decision

**REVISE — broader initial-state coverage clearly helps but remains insufficient.**

The success/OOD trend is monotonic and S-Large surpasses U-High and D2-T on the untouched test, so rejecting data coverage would contradict the causal evidence. Conversely, 26/96 success, 50 wall collisions, and 100% strict x0 OOD are nowhere near a reliable baseline, so a passing decision is not defensible.

## 10. Minimal next step

Pre-register one additional globally uniform, nested initial-state scale—recommended **S-XL = 288 independent states (96 per regime)**—using exactly the same distribution, eight expert modes per state, 64 uniform recovery anchors per trajectory, perturbation law, MACFlow implementation, exposure rule, frozen validation pool, and frozen untouched test. This requires 144 new independent training states, not any failure-near, phase-specific, or velocity-specific acquisition.

The purpose of this single extension is to determine whether the still-improving 24→72→144 curve continues or saturates. Do not add a density sweep or inspect test failures to shape it. If S-XL substantially lowers continuous x0/rollout distance and raises success, continue treating coverage as the active limitation. If support becomes genuinely adequate while success saturates, that would finally justify a representation/model audit.

No safety-baseline, eta/basin, or `G_phi` experiment was started.

## Reproducibility and repository integrity

- Protocol validation: [`protocol_validation.json`](protocol_validation.json), all checks passed.
- Frozen comparison SHA-256: `9342826264f24a095a3246b8b15b175081e907a4a773977239ee95ae80f8b32b`.
- Canonical `double_bottleneck/flowbc_4a_agent.py` SHA-256 remains `02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8`.
- No explicit mode input, targeted acquisition, persistent latent, horizon/action chunks, recurrence, eta logic, or `G_phi` appears in training metadata.
- Toy Give-Way was not modified by this study.
- Regression results: `bash scripts/test.sh` passed its 58-, 2-, and 19-test suites; the explicit `python -m unittest discover -s double_bottleneck/tests` run passed 24/24. This is 103 successful test invocations across the two commands (the package-level command may overlap tests reached by the repository script). The only extra output was the known non-Slurm JAX CUDA-plugin warning; execution used CPU and all commands exited zero.
- Final tracked diff remains limited to the pre-existing `README.md` and `scripts/test.sh` changes; this study is isolated under `diagnostics/double_bottleneck_initial_state_coverage/`. The working tree was already broadly dirty/untracked before this phase, so no ownership claim is made over the other listed paths.
