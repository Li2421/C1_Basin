# Final uniform recovery-density check: S-XL-64 vs S-XL-128

## 1. Frozen 64-anchor reference

The reference is the completed S-XL Stage-I experiment, not a retrained control. It has the frozen 288 uniformly sampled nominal initial states, 2,304 centralized-expert trajectories, eight equally represented coordination hypotheses, 1,693,072 nominal transitions, and 147,456 accepted recovery transitions (64 per trajectory), for 1,840,528 total transitions. Its canonical unconditioned policy is the Toy-sourced `72D condition -> 8D joint action` MACFlow with a `256x3` ActorVectorField, conditional flow matching, Adam, and 10 Euler sampling steps with fresh base noise each environment step.

The S-XL-64 selected checkpoint used 461,000 updates / 64.12 expected sample exposures. Its two frozen 96-rollout untouched tests both achieved 27/96 full-task success. This experiment preserves its nominal train pool, validation pool, development set, existing untouched set, fresh untouched set, rollout seeds, expert, action semantics, perturbation law, and policy formulation exactly.

## 2. 128-anchor dataset construction

The preregistration is [`PREREGISTRATION.json`](PREREGISTRATION.json), SHA-256 `2681e04c6614ec8a491364a6b8ca14f6018abb06a72ad1b6040889a08536264c`.

For every existing expert trajectory, the frozen 64 recovery rows were retained byte-for-byte. A further 64 distinct action indices were selected uniformly without replacement from that trajectory's remaining action indices, using the globally fixed seed rule `SHA256(root_label|recovery_extra128|train|rollout_id)`. Conditional on the original uniformly selected 64, this makes the union a uniform 128-element without-replacement anchor set. Position noise remains coordinatewise `Uniform[-0.02, 0.02] m`; velocity noise remains coordinatewise `Uniform[-0.2, 0.2] m/s`, followed by the same 0.5 m/s radial bound. The perturbation seed is a fixed rollout-ID-derived `perturb_extra128` stream.

Every extra anchor was queried exactly once with the unchanged scenario-local expert reference recovery. The acceptance rule remained successful recovery with no collision or timeout. There was no resampling, phase weighting, failure-driven collection, wall/goal/bottleneck shell, mode conditioning, or RTL-specific collection.

| Quantity | S-XL-64 | S-XL-128 |
|---|---:|---:|
| Nominal transitions | 1,693,072 | 1,693,072 |
| Recovery transitions | 147,456 | 294,912 |
| Total transitions | 1,840,528 | 1,987,984 |
| Recovery anchors / expert trajectory | 64 | 128 |
| Added transitions | — | 147,456 (+8.01%) |
| Extra recovery accepted / attempted | — | 147,456 / 147,456 |

The added rows are balanced by regime (49,152 each) and initial expert direction (73,728 LTR and 73,728 RTL). Their natural phase frequencies were merely logged: 64,642 waiting/yielding, 22,182 coordination-transition, 16,498 final-approach, 13,521 bottleneck, 12,164 second-bottleneck, 11,236 chamber, 4,022 near-goal, 1,714 initial-approach, and 1,477 first-bottleneck. No post-hoc rebalancing occurred. Data details are in [`data/manifest.json`](data/manifest.json).

## 3. Training exposure comparison

The model, observation/action dimensions, optimizer family, batch size (256), learning rate (`3e-4`), loss, normalization procedure, sampling semantics, and Euler integration remained unchanged. The fair primary rule was pre-registered before generation:

`updates = ceil((64 × training transitions / 256) / 1000) × 1000`.

This gives S-XL-128 a primary budget of 497,000 updates / 64.00 expected exposures. The same frozen S-XL convergence rule allowed one continuation to 96 exposures only when the fixed validation loss fell at least 5%, fixed train loss fell at least 3%, and the primary validation/train gap was at most 25% from the 75% milestone to the primary final. It triggered without access to development or untouched rollouts: validation/train reductions were 12.16%/11.79% and the primary gap was 19.15%.

Thus the selected S-XL-128 checkpoint is the pre-registered extension final at 746,000 updates / 96.07 expected exposures, SHA-256 `6e2ed4e31443bbb34457d3b3aabe0e7d741b904259391f8893b1104c143546bd`. Its final fixed train/validation losses were `0.04954 / 0.04072`, compared with S-XL-64's `0.04132 / 0.04925` at its selected primary checkpoint. The different final exposure counts are disclosed rather than hidden: they arise only from the same pre-registered validation-convergence rule, not test-driven tuning. Complete logs are in [`model/training_summary.json`](model/training_summary.json).

## 4. Full-task 64 vs 128 results

Success is actual full task completion by all four agents within the 850-step horizon. K-step survival is not substituted for it. No rollout used safety projection, eta/basin correction, or `G_phi`.

| Frozen set (96 rollouts) | Model | Success | Wall | Agent | Timeout | Mean / median episode steps |
|---|---|---:|---:|---:|---:|---:|
| Existing untouched | 64 anchors | 27 | 11 | 34 | 24 | 534.6 / 718.0 |
| Existing untouched | 128 anchors | **68** | **4** | **6** | 18 | 701.2 / 735.5 |
| Fresh untouched | 64 anchors | 27 | 7 | 43 | 19 | 534.0 / 487.0 |
| Fresh untouched | 128 anchors | **76** | **1** | **3** | 16 | 734.0 / 736.5 |

This is a +41/96 and +49/96 absolute success gain on the existing and independently generated fresh tests. The remaining failures are mostly timeouts, rather than the former recurrent collision modes. The legacy environment deadlock flag remains zero in all of these rollouts; this study did not change the separate 4-agent deadlock definition.

## 5. OOD and K-step diagnostics

Timestep-zero support distance and strict x0 OOD are unchanged, as expected: nominal initial-state coverage is exactly frozen. The difference is closed-loop local support.

| Test | Model | x0 mean distance | x0 strict OOD | Rollout OOD | Teacher RMSE | Waiting/transition RMSE | K=100 position RMSE | K=100 collision |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Existing | 64 | 0.07203 | 95.83% | 30.00% | 0.004682 | 0.006239 | 0.013498 | 6/288 |
| Existing | 128 | 0.07203 | 95.83% | **6.94%** | **0.004356** | **0.005787** | **0.010630** | **0/288** |
| Fresh | 64 | 0.07766 | 100.00% | 31.20% | 0.004646 | 0.006182 | 0.013430 | 3/288 |
| Fresh | 128 | 0.07766 | 100.00% | **5.68%** | **0.004339** | **0.005754** | **0.012218** | **0/288** |

The 64→128 density intervention reduces rollout OOD by 23.1 and 25.2 percentage points, respectively, while improving local action quality and eliminating this diagnostic's K=100 collisions. This is strong evidence for generic local recovery density—not a targeted correction of known regions. See [`figures/density_comparison.svg`](figures/density_comparison.svg).

## 6. LTR / RTL outcome stratification

Direction was inferred only after test results froze, from each successful rollout's realized complete coordination signature. No mode is supplied to the policy and no direction-specific data were added.

| Test | Model | LTR-first success | RTL-first success | Mixed / ambiguous | Complete order signatures |
|---|---|---:|---:|---:|---:|
| Existing | 64 | 25 | 2 | 0 | 6 |
| Existing | 128 | 39 | 29 | 0 | 8 |
| Fresh | 64 | 27 | 0 | 0 | 4 |
| Fresh | 128 | 31 | 45 | 0 | 8 |

The fresh-test all-LTR success pattern has disappeared. S-XL-128 expresses both macro directions on both tests and realizes all eight complete expert order signatures on each untouched test. It is not evidence that the learned distribution is perfectly calibrated, but it removes the previous striking one-direction outcome under balanced expert data. The corresponding plot is [`figures/direction_outcomes.svg`](figures/direction_outcomes.svg).

## 7. Evidence for saturation

The evidence is against saturation at 64 anchors. Doubling only generic local recovery density yields large, replicated full-task gains; wall collisions fall 11→4 and 7→1, agent collisions 34→6 and 43→3, and rollout OOD falls to below 7% on both untouched tests. The improvement occurs with the same 288 initial states and no selected failure locations.

There is no basis to keep scaling anchors indefinitely in this phase. The requested final conventional density check has established that the obvious 64-anchor local-coverage deficit was real and that 128 anchors produces a substantially stronger baseline. Remaining 18/96 and 20/96 failures are mostly timeout/liveness failures and are now legitimate targets for the subsequent safety/liveness analysis, rather than evidence of an obviously unfinished collision-prone imitation baseline.

## 8. Baseline maturity assessment

**Is the conventional Stage-I MACFlow now sufficiently mature for hard-safety evaluation, even if it does not achieve near-perfect task success? Yes.**

The expert is sound; nominal coverage was uniformly expanded to 288 states; both expert modes and all eight orders are balanced; generic perturb-and-requery was scaled uniformly; training was given a pre-registered, validation-triggered convergence extension; local K=100 behavior is collision-free in both tests; rollout support is now mostly retained; and performance improves independently on the fresh test without any location-specific acquisition. There is no observed implementation error or trivial generic coverage omission dominating the result.

It remains imperfect by design: 29.2% and 20.8% of the two test rollouts do not complete, chiefly through timeout. That residual is substantial enough to study, but no longer supports the criticism that the Stage-I baseline was merely left undertrained or deprived of routine recovery coverage.

## 9. Decision

**PASS — baseline sufficiently mature.**

## 10. Minimal next step

Run the **hard-safety baseline evaluation only**: apply the existing hard projection to this fixed S-XL-128 checkpoint, using the same frozen tests, then characterize collision avoidance, projection magnitude/intervention frequency, task completion, timeout, and safe-but-liveness-limited behavior. Do not start eta/basin analysis or `G_phi` in that next phase until the safety baseline itself is reported.

## Integrity and regression

- [`protocol_validation.json`](protocol_validation.json) passes every check: exact S-XL nominal-pool reuse; byte-identical retention of all old recovery rows; 128 distinct source steps per trajectory; zero old/new overlap; globally balanced added directions; unchanged canonical source hash; and no prohibited features.
- Canonical `double_bottleneck/flowbc_4a_agent.py` remains SHA-256 `02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8`. Toy Give-Way was not modified.
- No hard safety evaluation, eta/basin experiment, or `G_phi` training was run here.
- Full regression after report generation passed: `bash scripts/test.sh` passed its 58-, 2-, and 19-test suites, and `python -m unittest discover -s double_bottleneck/tests` passed 24/24. The only extra output was the known CPU-only JAX CUDA-plugin warning; all commands exited zero.
