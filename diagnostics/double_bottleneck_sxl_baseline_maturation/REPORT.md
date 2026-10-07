# S-XL conventional Stage-I MACFlow maturation

## 1. S-XL pre-registered protocol

This study asked whether one failure-independent expansion to 288 independently sampled initial states makes the unchanged joint Stage-I MACFlow a reasonably mature conventional baseline. The preregistration is [`PREREGISTRATION.json`](PREREGISTRATION.json) (SHA-256 `99b5e78f6b133784af9a6a734eafff89e4e862a846b4cb6e76f6e2780d68ddc9`).

The initial-state law is exactly the parent study's symmetric global law: independently sampled side-progress offsets in `[-0.10, 0.10] m`, within-side progress spacings in `[-0.05, 0.05] m`, lateral common offsets in `[-0.015, 0.015] m`, lateral spacings in `[-0.010, 0.010] m`, travel-coordinate velocities in `[-0.05, 0.05] m/s`, and lateral velocities in `[-0.02, 0.02] m/s`. It draws 96 legal states in each of the clearly asymmetric, weakly asymmetric, and near-symmetric regimes. Every accepted state has all eight existing centralized-expert coordination hypotheses, with no mode selected preferentially.

For every one of the resulting 2,304 expert trajectories, exactly 64 distinct action-time anchors were selected uniformly with a rollout-ID-derived PCG64 seed. Position noise was coordinatewise uniform in `[-0.02, 0.02] m`; last-velocity noise was coordinatewise uniform in `[-0.2, 0.2] m/s`, followed by the common `0.5 m/s` radial bound. The recovery horizon was one transition. These rules, including acceptance and rejection, are global—not phase-, region-, failure-, seed-, velocity-family-, or mode-dependent.

The fresh 24-state test pool was independently drawn and written before training. Neither untouched-test set was loaded during training or checkpoint selection. The pre-registration explicitly disallows explicit modes, priority features, persistent latents, temporal models, action chunks, recurrence, eta/basin logic, and `G_phi`.

## 2. Dataset statistics

| Quantity | S-XL value |
|---|---:|
| Independent nominal initial states | 288 (96 per regime) |
| Expert trajectories | 2,304 (8 per state) |
| Nominal transitions | 1,693,072 |
| Uniform recovery transitions | 147,456 |
| Total transitions | 1,840,528 |
| Recovery-query acceptance | 147,456 / 147,456 |
| Nominal episode length, mean / median | 734.84 / 735 steps |

Each of the eight expert passage-order signatures occurs exactly 288 times. The natural, unmodified recovery phase distribution was: initial approach 1,816; first bottleneck 1,444; waiting/yielding 64,670; coordination transition 21,993; bottleneck traversal 13,634; chamber 11,439; second bottleneck 11,888; final approach 16,549; near-goal 4,023. This is an audit outcome of uniform sampling, not a rebalancing rule.

Dataset construction metadata, expert IDs, initial-state seeds, and recovery seeds are retained in [`data/manifest.json`](data/manifest.json) and [`manifest.json`](manifest.json).

## 3. Training convergence and exposure

The policy is the unmodified Toy-sourced joint MACFlow: `ActorVectorField`, 72-D condition, 8-D joint action, hidden `256x3` (154,632 parameters), conditional flow matching, Adam at `3e-4`, batch size 256, and 10 Euler sampling steps with fresh Gaussian base noise at each environment timestep. Transitions were sampled uniformly with replacement from nominal plus recovery rows.

The preregistered exposure rule gave 461,000 optimizer updates, or 64.12 expected samples per transition. Fixed train/validation losses at 75%, 87.5%, and 100% were `0.04479/0.05182`, `0.04375/0.05041`, and `0.04132/0.04925`. Validation fell 4.974% from the 75% checkpoint, narrowly below the preregistered 5% extension trigger; the 96-exposure extension was therefore not run. The capacity diagnostic was also preregisteredly ineligible: final validation loss was better, not 10% worse, than S-Large.

The final preselected checkpoint is step 461,000 (SHA-256 `65472ea5408fbbf18f250040127d2c8784be29925830dc90eb4c1aec65e44ddb`). Development-only checkpoint stability improved from 1/12 to 3/12 to 4/12 successes across 346k, 403k, and 461k; it did not change checkpoint selection. Full details and the loss plot are in [`model/training_summary.json`](model/training_summary.json), [`checkpoint_stability.json`](checkpoint_stability.json), and [`figures/training_convergence.svg`](figures/training_convergence.svg).

## 4. Coverage scaling table

The primary comparison uses the fixed parent 96-rollout untouched test. S-XL contains S-Large as an exact prefix and preserves 64 recovery anchors per expert trajectory.

| Training initial states | x0 mean support distance | Rollout OOD | Success | Wall | Agent | Timeout |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 0.10488 | 89.64% | 0/96 | 90 | 5 | 1 |
| 72 | 0.08698 | 49.63% | 10/96 | 34 | 21 | 31 |
| 144 | 0.08065 | 31.49% | 26/96 | 50 | 5 | 15 |
| 288 (S-XL) | **0.07203** | **30.00%** | **27/96** | **11** | **34** | **24** |

The continuous support distance improves monotonically, but full-task success has flattened between 144 and 288 states. S-XL strongly suppresses wall collisions, but the residual terminal distribution shifts toward agent collisions and timeouts rather than disappearing. See [`figures/coverage_scaling.svg`](figures/coverage_scaling.svg).

## 5. Full-task untouched-test results

Success means all four agents reach their goals in the complete 850-step rollout; finite-horizon survival is not substituted for task completion. No deadlock was reported by the legacy environment criterion in either S-XL test. The more appropriate 4-agent deadlock semantics remain a separate audit and were not changed here.

| Set (96 rollouts each) | Success | Wall | Agent | Timeout | Mean / median steps | x0 OOD | Rollout OOD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing untouched | 27 | 11 | 34 | 24 | 534.6 / 718.0 | 95.83% | 30.00% |
| Fresh untouched | 27 | 7 | 43 | 19 | 534.0 / 487.0 | 100.00% | 31.20% |

Both sets independently yield 28.125% success. Minimum swept clearances were negative because collision terminals occurred: existing wall/pair `-0.01259 / -0.00866 m`; fresh `-0.00878 / -0.01114 m`.

There is no total regime collapse, but coordination sampling is not calibrated: existing successful rollouts were 13/7/7 across clearly/weakly/near-symmetric regimes and 25 LTR-first versus 2 RTL-first; fresh successes were 12/7/8 and all 27 were LTR-first. Training data themselves remain exactly balanced across all eight expert modes. This outcome is therefore a policy behavior observation, not a data-mode imbalance.

Frozen raw results are in [`evaluation/comparison.json`](evaluation/comparison.json), with the outcome plot in [`figures/sxl_outcomes.svg`](figures/sxl_outcomes.svg).

## 6. K-step and support diagnostics

Teacher-forced local quality is good on both tests. Existing/fresh overall joint-action RMSE is `0.004682 / 0.004646`; waiting/transition RMSE is `0.006239 / 0.006182`; goal-region RMSE is `0.004622 / 0.004591`; and wall-directed absolute error is `0.000803 / 0.000795`.

| Horizon | Existing position RMSE | Existing collision | Fresh position RMSE | Fresh collision |
|---:|---:|---:|---:|---:|
| 1 | 0.000060 | 0/288 | 0.000060 | 0/288 |
| 5 | 0.000435 | 0/288 | 0.000422 | 0/288 |
| 10 | 0.001479 | 0/288 | 0.001478 | 0/288 |
| 25 | 0.005412 | 0/288 | 0.005451 | 0/288 |
| 50 | 0.009586 | 0/288 | 0.009547 | 0/288 |
| 100 | 0.013498 | 6/288 | 0.013430 | 3/288 |

This confirms locally stable 100-step behavior but does not establish complete-task robustness. The calibrated support threshold is the parent study's fixed D0-normalized 72-D nearest-neighbor RMS threshold, `0.0395698`. It is stringent for a continuous initial-state distribution: x0 is outside it for nearly all test cases even as mean x0 distance decreases. The important residual fact is that about 30–31% of full rollout states remain outside it.

## 7. Remaining failure taxonomy

Analysis began only after both frozen test results existed. It read saved trajectories and made one deterministic diagnostic reference-continuation probe per failed rollout. It did not add a data row, alter a model, or choose another checkpoint. The formal record is [`posthoc_failure_analysis.json`](posthoc_failure_analysis.json).

A clear divergence is joint-position RMS greater than `0.08 m` from every state in all eight expert trajectories for the same initial state, sustained for five steps.

| Measure | Existing untouched | Fresh untouched |
|---|---:|---:|
| Failed rollouts | 69 | 69 |
| Clear divergences | 27 | 34 |
| Median first clear divergence | step 81 | step 82 |
| At clear divergence, inside strict support | 1/27 | 3/34 |
| At deterministic probe, inside strict support | 11/69 | 8/69 |
| Nearest reference continuation recovered | 69/69 | 69/69 |

Clear divergence phases are primarily coordination transitions: 17/27 existing and 27/34 fresh; the rest are waiting, final approach, near-goal, plus one fresh bottleneck case. Terminal primary categories are `11 wall / 34 agent / 2 waiting-or-coordination timeout / 22 terminal-or-goal timeout` on the existing set, and `7 wall / 43 agent / 1 long-horizon drift / 18 terminal-or-goal timeout` on the fresh set.

The reference-continuation result says each chosen valid probe state is locally recoverable under the normal, same-family expert trajectory. It does not prove that arbitrary off-manifold states are globally planner-recoverable, and it must not be read as new recovery training data. The earlier broad fresh-planner probe is retained only as [`posthoc_failure_analysis_fresh_planner_superseded.json`](posthoc_failure_analysis_fresh_planner_superseded.json); it is superseded because it did not exactly match the preregistered reference-continuation semantics.

## 8. Baseline maturity assessment

**Is Stage-I MACFlow now sufficiently strong for the later safety/liveness experiment? No.**

It is a much stronger conventional baseline than nominal-only, V1, uniform-low-density, or S-Large in local quality and wall robustness. Expert generation is sound; nominal initial-state coverage is broad and pre-registered; local recovery is generic; optimization converged under the frozen rule; and 100-step behavior is stable.

However, 27/96 full success on each independently frozen test, 34/43 agent-collision terminals, 19/24 timeouts, approximately 30% rollout OOD, and the fresh-set one-direction success pattern are not yet a credible mature task-completion baseline. The evidence also does not isolate a representational limitation: most first persistent divergences and deterministic probes remain outside strict support. Consequently, neither a safety/basin experiment nor a claim that failure is purely an intrinsic Stage-I limitation is justified.

## 9. Decision

**REVISE — conventional baseline still improving.**

S-XL improves continuous support, teacher-forced error, finite-horizon robustness, and wall collision rate relative to S-Large, but its full-task success saturates at 27/96 on two independent tests. The residuals are not predominantly strong-support failures, so the data-coverage explanation is not exhausted; at the same time, another blind expansion of nominal initial-state count alone is not well motivated by the 144-to-288 success plateau.

## 10. Minimal next step

Pre-register one small **uniform recovery-density** ablation on the already frozen 288 nominal trajectories: retain the exact same global perturbation law, expert, mode balance, initial-state pool, and MACFlow implementation, and compare the existing 64 anchors per trajectory with one higher globally identical anchor count (for example 128). Scale the update budget by the pre-declared exposure rule, retain both current untouched tests, and create a third fresh test before training.

This is the smallest conventional, non-targeted test that can reduce local support holes without teaching any known failure region or changing the policy formulation. If it does not materially lower rollout OOD or improve complete-task success, the evidence for a representation/model audit becomes substantially stronger. Do not begin hard-safety, eta/basin, or `G_phi` work in this phase.

## Repository integrity and regression

- Protocol validation: [`protocol_validation.json`](protocol_validation.json) — every check passed: all four initial-state pools are disjoint, test pools were frozen before training, every source trajectory has exactly 64 anchors, all eight modes are balanced, and the selected checkpoint is the preregistered one.
- Canonical `double_bottleneck/flowbc_4a_agent.py` SHA-256 remains `02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8`.
- The study did not modify Toy Give-Way, canonical MACFlow source, the older datasets/checkpoints, safety logic, eta/basin logic, or `G_phi`.
- `bash scripts/test.sh` passed 58, 2, and 19 tests; `python -m unittest discover -s double_bottleneck/tests` passed 24/24. The only extra output was the known CPU-only JAX CUDA-plugin warning; both commands exited zero.
- The working tree was already dirty before the study (`README.md`, `scripts/test.sh`, and multiple untracked scenario/diagnostic paths). This study is isolated under `diagnostics/double_bottleneck_sxl_baseline_maturation/`; no ownership claim is made over unrelated changes.
