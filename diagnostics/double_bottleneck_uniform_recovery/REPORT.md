# Double-Bottleneck Uniform Recovery Study

Date: 2026-09-23  
Scope: failure-independent data coverage only; unchanged Toy-sourced Stage-I joint MACFlow

## 1. Pre-registered sampling protocol

The protocol was frozen in [`PREREGISTRATION.json`](PREREGISTRATION.json) before uniform data construction, training, or inspection of the new test outcomes. Its SHA-256 is `7168d2bfd222a2af28e8c3eebbb1a4443d945230851529f491a5a4566588f1dd`.

The selection rule is independent of observed policy failures, trajectory phase, geometry region, wall distance, bottleneck distance, goal distance, and diagnostic error:

- source corpus: all 72 original successful training expert trajectories and all 24 original validation expert trajectories;
- eligible anchors: every expert action index `[0,T)`;
- per-trajectory randomization: PCG64 seeded by the first 64 bits of `SHA256("uniform_recovery_v1|20261001|split|rollout_id")`;
- generate one random permutation of all eligible indices per trajectory;
- U-Low takes its first 64 indices, U-Mid its first 256, and U-High its first 640;
- the three sets are nested, sampled without replacement, and use the same K for every trajectory, regime, and coordination mode;
- indices are sorted only after selection; no phase labels participate in selection.

The globally fixed perturbation distribution is the Toy-derived V1 distribution:

- every agent position coordinate: independent `Uniform[-0.02,0.02] m`;
- every last-velocity coordinate: independent `Uniform[-0.2,0.2] m/s`, followed by the common `0.5 m/s` radial bound;
- one fixed recovery `(x,u)` row per selected anchor (`H=1`);
- the scale does not vary with phase, geometry, clearance, mode, or prior failure history.

This phase reused the immutable V1 re-query cache rather than recomputing identical expert queries. That cache has exactly one independent perturb-and-requery result for every nominal transition. Every Double-Bottleneck row was accepted on attempt 1, and every full continuation had already been replay-validated as successful and collision-free. The source cache hashes are frozen in the preregistration. Selecting a pre-registered uniform subset of this exhaustive cache is equivalent to querying those selected anchors anew; it cannot introduce failure-location selection.

There was no rejection or resampling in any arm. The rule for a hypothetical invalid/unrecoverable query was to record and discard it once without replacement or neighboring acquisition. No DAgger or `failed rollout -> local data -> retrain` loop was used.

### Untouched test preregistration

Before training, a new primary test distribution was also frozen:

- two new initial states per each of clearly asymmetric, weakly asymmetric, and near-symmetric regimes;
- agentwise `dx ~ Uniform[-0.06,0.06] m`, `dy ~ Uniform[-0.02,0.02] m` around the declared regime state;
- every initial last-velocity coordinate `Uniform[-0.04,0.04] m/s`;
- seed 20261002;
- all eight existing coordination hypotheses generated for each exact initial state;
- no redraw if a state or expert plan failed;
- closed-loop seeds 101, 211, 307, and 401.

All 6 first draws were physically valid. All 48 expert trajectories succeeded without collision or timeout and realized all eight actual passage signatures for each initial state. Minimum expert wall/pair clearance was `0.05333/0.08649 m`. The isolated dataset uses `val` as a storage-format label because the shared loader admits only `train/val`; its manifest explicitly records `scientific_split=untouched_test`, and no test transition was used for training or normalization. The test manifest is [`data/untouched_test_dataset/manifest.json`](data/untouched_test_dataset/manifest.json).

No individual model result or failure location was inspected until all six checkpoints had completed both evaluations and [`evaluation/comparison.json`](evaluation/comparison.json) had been written.

## 2. Dataset statistics

| Dataset | Nominal rows | Uniform recovery rows | Total train rows | Anchors per trajectory |
|---|---:|---:|---:|---:|
| D0 | 52,873 | 0 | 52,873 | 0 |
| U-Low | 52,873 | 4,608 | 57,481 | 64 |
| U-Mid | 52,873 | 18,432 | 71,305 | 256 |
| U-High | 52,873 | 46,080 | 98,953 | 640 |
| D1/full-per-transition reference | 52,873 | 52,873 | 105,746 | every transition |

Validation recovery counts were 1,536, 6,144, and 15,360 for U-Low/Mid/High. U-High selects 640 of each trajectory's 702–764 transitions; D1 is the exhaustive one-row-per-transition uniform endpoint.

Every uniform set is exactly balanced across the three regimes and the two first-direction hypotheses because every source trajectory receives the same K. Training recovery counts were:

- each regime: 1,536 / 6,144 / 15,360 for Low/Mid/High;
- LTR-first and RTL-first: 2,304 / 9,216 / 23,040 each;
- natural source-family mix in U-High: 15,360 bottleneck-distance, 15,360 lateral-offset, 5,120 nominal, and 10,240 velocity/symmetry-probe rows.

Actual position perturbation RMS mean was `0.01136/0.01138/0.01140 m` for Low/Mid/High; velocity perturbation RMS mean was `0.10682/0.10672/0.10692 m/s`. U-High instantaneous perturbed-state wall clearance had min/median/p95 `0.03333/0.06554/0.26410 m`; pair clearance was `0.04605/0.30182/0.57586 m`. All selected states were valid and all cached full recoveries successful.

Exact hashes, distributions, clearances, counts, and untouched-test initial states are in [`data/manifest.json`](data/manifest.json). Compact dataset counts are in [`tables/datasets.csv`](tables/datasets.csv).

## 3. Natural phase distribution

Phase labels were computed only after construction. No row was added, removed, weighted, or resampled in response.

| Natural phase | U-Low | U-Mid | U-High |
|---|---:|---:|---:|
| Initial approach | 1.15% | 1.24% | 1.21% |
| First bottleneck approach | 1.00% | 1.00% | 0.99% |
| Waiting/yielding | 44.08% | 44.02% | 43.83% |
| Coordination transition | 14.32% | 14.80% | 14.98% |
| Bottleneck traversal | 8.62% | 9.06% | 9.24% |
| Chamber traversal | 7.70% | 7.81% | 7.72% |
| Second bottleneck | 8.44% | 8.30% | 8.11% |
| Final goal approach | 11.61% | 11.09% | 11.17% |
| Near-goal termination | 3.08% | 2.67% | 2.74% |

The similar percentages across density levels are the expected result of global uniform sampling. Waiting/transition rows are common only because those phases occupy much of the expert trajectories—not because they were preferentially selected. Full counts are in [`tables/natural_phase_distribution.csv`](tables/natural_phase_distribution.csv).

## 4. Training configuration

Every new arm used the unchanged canonical Toy-sourced Stage-I MACFlow:

- condition: canonical 72D four-agent observation in order A1, A2, B1, B2;
- output: 8D joint instantaneous action;
- official `ActorVectorField`, hidden sizes `256 x 3`, 154,632 parameters;
- conditional flow-matching objective;
- train-corpus coordinate normalization with standard-deviation floor `0.01`;
- Adam, learning rate `3e-4`, batch size 256;
- seed 0, 25,000 updates, 10 Euler flow steps;
- uniform transition sampling with replacement;
- fixed final checkpoint, with no rollout-informed selection;
- canonical fresh base noise at every environment step and no safety projection.

| Arm | Train rows | Expected passes | Final fixed train loss | Final fixed validation loss |
|---|---:|---:|---:|---:|
| U-Low | 57,481 | 111.34 | 0.11644 | 0.12265 |
| U-Mid | 71,305 | 89.76 | 0.14857 | 0.15822 |
| U-High | 98,953 | 64.68 | 0.18489 | 0.18587 |

The 25k update budget was deliberately fixed rather than scaled by dataset size. Losses are measured on different mixture distributions and are not model-selection scores. Checkpoint hashes and exact configs are in [`models/training_comparison.json`](models/training_comparison.json) and [`tables/training.csv`](tables/training.csv).

The canonical policy source retained SHA-256 `02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8`. No mode labels, phase flags, recurrent state, persistent latent, horizon/action chunks, trajectory prediction, `G_phi`, or eta/basin code was added.

## 5. Development results

The repeatedly inspected 12 cases are reported only for continuity.

| Model | Success | Wall | Agent | Timeout | Median steps | OOD | K=100 collision |
|---|---:|---:|---:|---:|---:|---:|---:|
| D0 | 0/12 | 12 | 0 | 0 | 70.5 | 98.96% | 72.92% |
| D1 | 4/12 | 7 | 1 | 0 | 734.5 | 80.53% | 0.69% |
| D2-T diagnostic | **7/12** | 4 | 1 | 0 | 723.5 | 85.68% | **0%** |
| U-Low | 0/12 | 7 | 5 | 0 | 258.0 | 96.82% | 17.36% |
| U-Mid | 3/12 | 5 | 4 | 0 | 516.5 | **67.46%** | 6.94% |
| U-High | 1/12 | 10 | 1 | 0 | **773.5** | 85.32% | 1.39% |

Development success is not monotonic in uniform density. Density strongly improves K=100 safety and survival but does not convert those gains into reliable episode completion.

## 6. Untouched-test results

This is the primary result: 6 unseen initial-state families x 4 fixed rollout seeds = 24 rollouts per policy.

| Model | Success | Wall | Agent | Timeout | Median steps | Min wall | Min pair | OOD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 | 0/24 | 21 | 3 | 0 | 70.0 | -0.01819 | -0.01535 | 100.00% |
| D1 | 3/24 | 15 | 6 | 0 | 736.5 | -0.01360 | -0.01266 | 86.98% |
| D2-T diagnostic | **7/24** | 13 | 4 | 0 | **758.0** | -0.00740 | -0.00948 | 84.86% |
| U-Low | 0/24 | 14 | 9 | 1 | 282.0 | -0.00943 | -0.01011 | 98.83% |
| U-Mid | 3/24 | 13 | 8 | 0 | 303.0 | **-0.00750** | -0.01482 | **66.67%** |
| U-High | 3/24 | 15 | 5 | 1 | 724.5 | -0.01358 | **-0.00841** | 86.57% |

Regime success for U-Mid was clearly/near/weak `2/8, 1/8, 0/8`; for U-High it was `0/8, 2/8, 1/8`. Thus both exhibit systematic regime families despite successes in both LTR-first and RTL-first modes. The result is not a one-direction collapse, but it is far from reliable.

![Untouched-test outcomes](plots/untouched_test_outcomes.svg)

### Matched K-step divergence on untouched test

Each cell below is joint-position RMSE in metres; all models use the same 288 continuations per horizon.

| Model | K=1 | K=5 | K=10 | K=25 | K=50 | K=100 |
|---|---:|---:|---:|---:|---:|---:|
| D0 | 0.00014 | 0.00158 | 0.00527 | 0.02467 | 0.06416 | 0.09722 |
| D1 | 0.00044 | 0.00154 | 0.00299 | 0.00984 | 0.02149 | 0.03018 |
| D2-T | 0.00046 | 0.00146 | 0.00277 | 0.00909 | 0.01836 | 0.02612 |
| U-Low | 0.00014 | 0.00107 | 0.00287 | 0.00983 | 0.02085 | 0.03081 |
| U-Mid | 0.00019 | **0.00097** | **0.00232** | **0.00859** | 0.01928 | 0.02797 |
| U-High | 0.00041 | 0.00141 | 0.00271 | 0.00861 | **0.01804** | **0.02600** |

K=100 collision rates were D0 `70.49%`, D1 `0.69%`, D2-T `1.39%`, U-Low `17.36%`, U-Mid `5.90%`, and U-High `0.69%`. U-High therefore matches D1's short-horizon collision rate and D2-T's K=100 positional deviation, but still succeeds in only 3/24 full episodes. K=100 no longer predicts long-horizon completion.

### Teacher-forced action quality

| Model | Action RMSE | Waiting/transition RMSE | Goal RMSE | Wall-directed error |
|---|---:|---:|---:|---:|
| D0 | **0.00805** | **0.01012** | **0.00916** | **0.00168** |
| D1 | 0.01849 | 0.02149 | 0.01957 | 0.00431 |
| D2-T | 0.01871 | 0.02161 | 0.02061 | 0.00405 |
| U-Low | 0.00879 | 0.01106 | 0.01064 | 0.00213 |
| U-Mid | 0.01057 | 0.01315 | 0.01283 | 0.00281 |
| U-High | 0.01781 | 0.02078 | 0.01958 | 0.00478 |

The nominal model again has the best one-step averages and the worst rollout. Increasing recovery density trades one-step imitation precision for much better recovery behavior, but that trade does not yield reliable completion.

Exact closed-loop, K-step, and teacher-forced tables are [`tables/closed_loop.csv`](tables/closed_loop.csv), [`tables/k_step.csv`](tables/k_step.csv), and [`tables/teacher_forced.csv`](tables/teacher_forced.csv).

## 7. Recovery-density effect

Generic uniform recovery has a real but limited effect:

- relative to D0, U-Mid/U-High greatly improve survival, K-step divergence, K=100 collision, and support occupancy;
- U-High reaches only 3/24 on untouched test, the same success count as U-Mid and the exhaustive D1 uniform reference;
- development success is `0 -> 3 -> 1` from Low to High; untouched success is `0 -> 3 -> 3`;
- aggregate OOD is also non-monotonic (`98.83% -> 66.67% -> 86.57%`) because it measures the states induced by each different closed-loop policy, not static dataset coverage alone;
- D1 perturbs every source transition and still reaches only 3/24 on the untouched test.

Therefore increasing K improves local stability but does not produce a monotonic or sufficient improvement in the scientific endpoint. The data do not support another anchor-count sweep over the same 72 source trajectories.

![Uniform density effect](plots/uniform_density_effect.svg)

## 8. Targeted versus uniform comparison

D2-T remains stronger in episode completion: 7/12 versus U-High 1/12 on development, and 7/24 versus U-High 3/24 on untouched test. Its advantage is long-horizon rather than local K=100 safety: on untouched test D2-T has a 1.39% K=100 collision rate versus U-High's 0.69%.

This is evidence that performance is disproportionately sensitive to particular portions of the trajectories. It is **not** permission to use D2-T as the canonical method. D2-T intentionally changes phase density and was selected after observing failures; it remains a diagnostic ablation only. It also contains D1's exhaustive recovery set plus targeted rows, so its row count and mixture are not identical to U-High.

The uniform data naturally already contain about 44% waiting and 15% transition rows, yet failures remain concentrated in coordination changes and waiting. This suggests those states are intrinsically harder for the present observation/model/training combination, rather than simply absent as named phases. The present experiment does not isolate which part of that combination is responsible.

## 9. Remaining failures

Failure analysis began only after the frozen comparison was written. It made zero expert queries and generated zero new training rows. The full record is [`failure_analysis.json`](failure_analysis.json).

| Arm | Failures | Clear divergence (`>0.08 m`) | Clear divergence inside calibrated support |
|---|---:|---:|---:|
| U-Low | 24 | 13 | 0 |
| U-Mid | 21 | 12 | 4 |
| U-High | 21 | 16 | 0 |

U-High's 16 clear-divergence phases were: 6 coordination transitions, 4 waiting/yielding, 3 final approach, 2 near-goal, and 1 second bottleneck. All 16 were outside the fixed support threshold. Its 21 failures span all regimes (8 clear, 6 near, 7 weak), so no single family explains the result. The five failures without a `0.08 m` event include small-error collision/timeout behavior that this coarse position threshold cannot detect early.

U-Mid had the lowest aggregate OOD, but 4 of its 12 clear divergences occurred inside calibrated support. This is limited evidence that coverage alone is not the whole problem. The other 8 clear divergences remained unsupported, so the study does not justify attributing every failure to representation or capacity.

All new test initial states were outside the stringent nominal q99 support threshold at step 0. They were deliberately new but only mildly perturbed from the declared regime states. This identifies a broader issue than phase targeting: the original training corpus contains only nine training initial-condition families. Uniformly perturbing intermediate points on those same 72 mode trajectories does not create broad coverage over independent initial-state families.

No remaining failure was used to select or acquire data in this task.

## 10. Decision

**REVISE — generic recovery clearly helps but coverage is still insufficient.**

The strongest uniform arm achieves large short-horizon and survival gains but only 3/24 success, with systematic failures across every regime. This fails a practical reliability gate of at least 20/24 overall, at least 6/8 per regime, no systematic collision family, and both coordination directions represented. The latter direction criterion is met among the few successes; the reliability criteria are not.

A model/representation rejection conclusion is not yet warranted because most clear U-High divergence still occurs outside support, every new test family begins outside the strict training support threshold, and the source corpus has very few independent initial states. At the same time, the density plateau means that adding more anchor indices along the same trajectories is not justified.

## 11. Next step

The smallest evidence-supported next step is a **pre-registered expansion of base expert initial-condition diversity**, not targeted recovery and not a larger K sweep:

1. define a failure-independent distribution over longitudinal offsets, lateral offsets, and initial last-velocity for every regime before generation;
2. draw substantially more independent initial states uniformly from that distribution, keeping the same number of eight-mode expert trajectories per state;
3. apply the same global uniform perturb-and-requery rule to those trajectories, with no phase weighting and no use of current test failures;
4. create another untouched set from the same predeclared distribution before training;
5. retain the identical Stage-I `p(u|x)` implementation and test whether initial-state OOD and full-episode success improve.

If a broader independent-state corpus places test rollouts inside support but success remains saturated, then a controlled representation/model-capacity diagnostic becomes justified. The four U-Mid supported divergences should be retained as evidence for that later question, not used as acquisition targets.

No safety-baseline, eta/basin, or `G_phi` experiment was run.

## Reproducibility and cleanup

- All previous D0, D1, D2-T, D2-G, D2-V, and D2 artifacts remain untouched.
- Accepted phase-specific files are isolated under `diagnostics/double_bottleneck_uniform_recovery/`; no diagnostic hook was added to the canonical policy or rollout path.
- The first deterministic test-dataset save encountered only an unsupported `test` storage label. The incomplete output was moved outside the formal result path; the same preregistered six states were rebuilt without redraw using the loader's `val` storage label and an explicit `scientific_split=untouched_test` manifest field.
- Toy Give-Way was not modified.
- [`protocol_validation.json`](protocol_validation.json) passes every preregistration/hash check: exact K per episode, unique transitions, Low ⊂ Mid ⊂ High nesting in both splits, common frozen preregistration hash, evaluation freeze ordering, and unchanged canonical-agent hash.
- Final regression: **103 tests passed** across the maintained main suite (58), Toy regression (2), Double-Bottleneck integration suite (19), and scenario unit suite (24). JAX emitted the known unavailable-CUDA warning and used the expected CPU fallback.
- The worktree was already dirty/untracked before this phase (`README.md`, `scripts/test.sh`, scenario and historical diagnostic paths). Those pre-existing changes were preserved; no reset, deletion, or overwrite was performed.
