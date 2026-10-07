# Double-Bottleneck MACFlow Recovery V2 Report

Date: 2026-09-23  
Scope: data-coverage intervention only; no change to the Stage-I MACFlow formulation

## Frozen experimental contract

The pre-intervention contract is recorded verbatim in [`BASELINE_FREEZE.json`](BASELINE_FREEZE.json). The policy is the official Toy-sourced joint Stage-I MACFlow `p(u|x)`: a 72D condition, 8D joint action, `ActorVectorField` with three 256-unit hidden layers (154,632 parameters), conditional flow-matching loss, Adam at `3e-4`, batch size 256, 25,000 updates, and 10 Euler integration steps. Observations, actions, train-only coordinate normalization (standard-deviation floor `0.01`), radial `0.5 m/s` action bound, fresh Gaussian base noise at every environment step, and all evaluation seeds were frozen before V2 generation.

The canonical source hash for `double_bottleneck/flowbc_4a_agent.py` is still `02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8`. No explicit mode input, persistent latent, recurrence, action horizon, trajectory prediction, `G_phi`, or eta/basin mechanism was introduced. Raw closed-loop evaluation used no hard-safety projection.

The fixed evaluation set contains three held-out families—clearly asymmetric, near-symmetric, and weakly asymmetric with an unseen velocity/symmetry probe—each evaluated with seeds 0, 1, 2, and 3. D0 and D1 checkpoint/data hashes, evaluation semantics, and prior outcomes are frozen in the baseline record.

An operational PASS gate was set as: at least 10/12 raw closed-loop successes, at least 3/4 in every regime, no systematic wall/agent-collision or velocity-family failure, successful behavior in both major first-direction classes, and support OOD below D1. A hard-safety intervention-rate check would follow only after this raw-policy gate; it was not run because no V2 arm met the raw gate.

## 1. V1 coverage audit

The audit used all 52,873 V1 training recovery transitions and 17,629 validation recovery transitions. Phases were assigned exclusively with documented precedence, so the counts do not double-count transition neighborhoods. Complete definitions and per-phase clearance/perturbation distributions are in [`v1_coverage_audit.json`](v1_coverage_audit.json).

| V1 training phase | Samples | Share |
|---|---:|---:|
| Initial approach | 635 | 1.20% |
| First bottleneck approach | 524 | 0.99% |
| Waiting/yielding | 23,178 | 43.84% |
| Coordination/mode transition | 7,913 | 14.97% |
| Bottleneck traversal | 4,866 | 9.20% |
| Chamber traversal | 4,100 | 7.75% |
| Second bottleneck | 4,308 | 8.15% |
| Final goal approach | 5,909 | 11.18% |
| Near-goal termination | 1,440 | 2.72% |

V1 was not globally missing waiting examples: waiting and transition together comprised 58.80% of its rows. The important gaps were more structured:

- transition samples were broad perturbations around all nominal states, not dense shells around the exact yield/release, convoy-start, bottleneck-entry/exit, and re-acceleration boundaries implicated by failed rollouts;
- near-goal termination was only 2.72% of training recovery data;
- the exact held-out negative initial-velocity family did not occur in training;
- the V1 training family mix was bottleneck-distance 17,613, lateral-offset 17,629, nominal 5,840, and velocity-probe 11,791 transitions; its modes were balanced (26,268 LTR-first and 26,605 RTL-first).

V1 perturbations were local: position RMS mean/median/p95 were `0.01140/0.01147/0.01435 m`; velocity RMS was `0.10687/0.10746/0.13768 m/s`; maximum coordinatewise position displacement was at most `0.02 m`. All 52,873 training and 17,629 validation queries recovered successfully, with no collision or timeout. Minimum full-recovery wall/pair clearance in training was `0.03333/0.04605 m`.

Thus V1's deficiency was not simply too few recovery rows. It was insufficient density around particular closed-loop transition and terminal-error geometries, plus incomplete initial-velocity coverage.

## 2. V2 dataset construction

### Transition and waiting recovery

Anchors were selected from real expert trajectories around activity changes, action changes, waits, yield-to-move/move-to-yield changes, convoy start/release, bottleneck entry/exit, and chamber re-acceleration. Each anchor received two independently seeded, physically valid perturbations:

- longitudinal position: uniform `[-0.025, 0.025] m`;
- lateral position: uniform `[-0.015, 0.015] m`;
- within-convoy relative spacing: uniform `[-0.020, 0.020] m`;
- velocity: uniform `[-0.080, 0.080] m/s` per relevant component.

These scales were calibrated against the D1 K=25/K=50 position deviations (`0.01355/0.02732 m`) and action deviations (`0.03175/0.03868 m/s`) and observed pre-collision drift. They are local rather than arbitrary state-space noise.

The transition arm produced 5,184 training and 1,728 validation recovery rows. Source phases included 2,136/708 coordination transitions and 2,014/664 waiting states (train/validation), with smaller neighborhoods spanning bottleneck entry/exit and chamber/goal transitions.

### Goal recovery

Goal anchors covered lateral offsets, longitudinal over/undershoot, residual velocity, imperfect arrival direction, asymmetric timing, and cases where one agent was nearly complete while others remained active. Perturbations were:

- longitudinal position: uniform `[-0.040, 0.040] m`;
- lateral position: uniform `[-0.025, 0.025] m`;
- relative spacing: uniform `[-0.015, 0.015] m`;
- velocity: uniform `[-0.100, 0.100] m/s`, with an additional travel-direction residual up to `0.04 m/s`.

This arm produced 2,592 training and 864 validation rows.

### Initial-velocity families

Two nearby, nonduplicated negative-velocity families were added at 0.75x and 1.25x of the held-out pattern; the exact held-out evaluation state was not copied. All eight expert passage orders were generated for each family. The result was 16/16 successful trajectories and 11,670 transitions, balanced 8/8 between LTR-first and RTL-first, with zero wall/agent collision or timeout and minimum wall/pair clearance `0.05333/0.08649 m`.

Because the plant is first-order in commanded velocity, the initial velocity is overwritten after the first action. Consequently, most of these 11,670 rows are nearly ordinary weak-family trajectory bodies; only the short prefix directly represents the intended initial-velocity intervention. This becomes important when interpreting D2-V.

### Recovery validation and provenance

All 7,776 targeted training queries and all 2,592 validation queries succeeded on the first accepted attempt, with zero rejected/invalid/colliding/timeout recovery. Actual passage-order signatures matched their normal expert source in every case; the policy never receives this metadata. Targeted data were exactly balanced between LTR-first and RTL-first. Minimum full-recovery wall/pair clearance was `0.02834/0.02490 m` in training and `0.02843/0.03691 m` in validation; maximum recovery length was 764 steps.

The expert-side recovery uses the existing centralized planner's normal coordination hypotheses and a bounded local reference tracker. This selects a feasible expert continuation, but does not add mode information or future information to MACFlow. Seeds were 20260931 (train) and 20260932 (validation). Dataset hashes, source IDs, phase counts, perturbations, clearances, and success statistics are in [`data/manifest.json`](data/manifest.json).

## 3. Dataset comparison

| Dataset | Composition | Training rows | Approx. data passes in 25k updates |
|---|---|---:|---:|
| D0 | Nominal expert | 52,873 | 121.0 |
| D1 | Nominal + Recovery V1 | 105,746 | 60.5 |
| D2-T | D1 + transition/waiting targeted | 110,930 | 57.7 |
| D2-G | D1 + goal targeted | 108,338 | 59.1 |
| D2-V | D1 + velocity-family trajectories | 117,416 | 54.5 |
| D2 | D1 + transition + goal + velocity | 125,192 | 51.1 |

All arms used the identical architecture, optimizer, batch size, integration/sampling semantics, seed 0, and 25,000-update budget. Updates were not increased proportionally: this preserves a clean fixed-budget coverage comparison, although larger datasets receive fewer expected passes. Sampling was uniform over concatenated transitions, and checkpoint selection used the fixed final update rather than rollout-informed selection.

The respective final validation losses were D0 `0.05305`, D1 `0.19625`, D2-T `0.19846`, D2-G `0.19015`, D2-V `0.19687`, and D2 `0.19246`. These losses are evaluated on different mixture distributions and are reported for numerics only; they are not comparable selection scores and do not predict closed-loop ranking.

## 4. Closed-loop results

All values below use the same held-out states and seeds.

| Dataset | Success | Wall | Agent | Timeout | Median steps | Mean steps | Min wall clearance | Min pair clearance | Successful first-direction signatures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| D0 | 0/12 | 12 | 0 | 0 | 70.5 | 63.3 | -0.01672 | 0.05652 | 0 |
| D1 | 4/12 | 7 | 1 | 0 | 734.5 | 564.3 | -0.01078 | -0.00035 | 2 |
| D2-T | **7/12** | **4** | 1 | 0 | 723.5 | **621.3** | -0.01038 | -0.00740 | **4** |
| D2-G | 1/12 | 7 | 4 | 0 | 751.0 | 612.4 | -0.00512 | -0.00135 | 1 |
| D2-V | 1/12 | 6 | 5 | 0 | 644.5 | 561.9 | -0.01651 | -0.00465 | 1 |
| D2 | 3/12 | 8 | 1 | 0 | **764.5** | **700.8** | -0.01036 | -0.00010 | 2 |

D2-T is the only positive controlled transfer. Its regime successes were 1/4 clearly asymmetric, 2/4 near-symmetric, and 4/4 weak velocity/symmetry probe, versus D1's systematic 0/4 weak-family result. Its successful rollouts include both LTR-first and RTL-first behavior, so the improvement did not visibly collapse to one major coordination direction.

The requested combined D2 arm was not reliable: 3/12 success, with 8 wall collisions. Its three regimes were each 1/4, and its two successful passage signatures were both LTR-first. The sample is too small to establish permanent mode collapse, but it fails the no-obvious-collapse and all-regime reliability requirements.

Goal-only and velocity-only additions each fell to 1/12 and converted some wall failures into agent collisions. This does not establish that goal or velocity coverage is intrinsically harmful. It establishes that these specific additions, uniformly mixed at a fixed 25k budget, are not beneficial causal interventions. In particular, D2-V heavily dilutes the mixture with post-initial rows that no longer express the velocity-family distinction.

Full per-family results and raw trajectories are preserved in [`evaluation/comparison.json`](evaluation/comparison.json) and [`evaluation/trajectories/`](evaluation/trajectories/). The compact table is [`tables/closed_loop.csv`](tables/closed_loop.csv).

![Closed-loop outcomes](plots/closed_loop_outcomes.svg)

## 5. OOD and divergence analysis

Support distance is the same calibrated nearest-neighbor RMS in normalized observation space used in the prior diagnosis. The common threshold is the fixed nominal calibration q99, `0.0395698`; it is not recomputed per arm.

| Dataset | Closed-loop OOD | K=100 collision rate | K=100 position RMSE | K=100 action RMSE |
|---|---:|---:|---:|---:|
| D0 | 98.96% | 72.92% | 0.08907 m | 0.10336 m/s |
| D1 | **80.53%** | 0.69% | 0.02757 m | 0.04097 m/s |
| D2-T | 85.68% | **0%** | **0.02334 m** | 0.04129 m/s |
| D2-G | 87.66% | 2.08% | 0.02080 m | **0.04015 m/s** |
| D2-V | 85.76% | 0.69% | 0.02620 m | 0.04053 m/s |
| D2 | 84.42% | **0%** | 0.02525 m | 0.04198 m/s |

D2-T eliminates collision in all 144 K=100 continuations and lowers K=100 positional deviation relative to D1, consistent with better short-to-medium recovery around transition boundaries. Yet its global OOD fraction is worse than D1 despite better closed-loop success. Therefore this scalar global OOD fraction is useful as a warning but is not a sufficient control-quality metric; where support is added matters more than aggregate nearest-neighbor coverage.

The earlier Toy-transfer audit reported D0/D1 K=100 collision rates of 98.61%/6.25% on its smaller continuation sample. This phase reran every arm with a denser, fixed matched protocol (24 trajectories x 6 semantic anchors = 144 continuations per K), yielding 72.92%/0.69% for D0/D1. The historical numbers remain frozen; all V2 causal comparisons use only the new matched protocol.

For D2-T, position RMSE at K=`1,5,10,25,50,100` was `0.00053, 0.00159, 0.00265, 0.00835, 0.01627, 0.02334 m`; collision rate was zero at every horizon. D2 was similar through K=100 (`0.00049, 0.00149, 0.00256, 0.00778, 0.01724, 0.02525 m`, zero collision), but failed later in complete episodes. Thus K=100 is no longer long enough to screen terminal-region failures.

Teacher-forced action RMSE (overall / waiting-transition / goal) was:

| Dataset | Overall | Waiting/transition | Goal | Wall-directed absolute error |
|---|---:|---:|---:|---:|
| D0 | 0.00807 | 0.01014 | 0.00916 | 0.00172 |
| D1 | 0.01833 | 0.02123 | 0.01959 | 0.00424 |
| D2-T | 0.01857 | 0.02139 | 0.02064 | **0.00400** |
| D2-G | 0.01896 | 0.02173 | 0.02216 | 0.00475 |
| D2-V | **0.01785** | **0.02071** | **0.01933** | 0.00441 |
| D2 | 0.01838 | 0.02120 | 0.02123 | 0.00441 |

The same fixed held-out nominal states and four action samples were used. Average local errors do not explain the rollout ranking: D2-G worsened goal RMSE despite goal data, while D2-T improved closed-loop behavior without improving average waiting/transition RMSE. The causal signal is closed-loop survival/success, not average one-step loss.

![Support and K=100 comparison](plots/support_and_k100.svg)

Exact K-step results are in [`tables/k_step.csv`](tables/k_step.csv); teacher-forced results are in [`tables/teacher_forced.csv`](tables/teacher_forced.csv).

## 6. Failure-mode attribution

### Transition/waiting coverage: supported positive effect

D2-T raises success from 4/12 to 7/12, reduces wall collisions from 7 to 4, eliminates K=100 collisions, and changes the weak velocity/symmetry family from 0/4 to 4/4. Because D2-T is the only changed variable, targeted transition/waiting coverage is a credible cause of these gains. The weak-family recovery also shows that the original family failure was not necessarily an initial-velocity representation failure; better downstream transition robustness can solve it.

### Goal coverage: this construction is a negative result

D2-G gives 1/12 success, 7 wall and 4 agent collisions, and worse goal-region and wall-directed errors than D1. Its samples are valid recoveries, but uniformly adding isolated goal-anchor states at the fixed budget did not create a stable terminal field. Plausible mechanisms—mixture dilution, locally conflicting actions, or insufficient trajectory-connected terminal shells—remain hypotheses, not established explanations.

### Velocity coverage: this construction does not isolate the causal variable

D2-V also gives 1/12. The two velocity families are valid and multimodal, but after one control step the environment's last-applied-velocity feature is set by the action. Training on complete 700+-step trajectories therefore adds 11,670 rows while only a tiny prefix carries the intended initial-velocity variation. D2-T's 4/4 weak-family result is stronger evidence that more full velocity-family trajectories are not the next priority.

### Combined D2: negative interaction, not “more data helps”

The full union regresses to 3/12 even while median survival increases to 764.5 and K=100 collisions fall to zero. This is a direct negative result for indiscriminate unioning at fixed budget. Transition recovery supplies the useful effect; goal/velocity rows dilute or interfere with it under the current uniform sampler. No architecture conclusion follows from this interaction alone.

## 7. Residual failures

Failure replay is preserved in [`failure_replay.json`](failure_replay.json) and summarized in [`tables/failure_replay.csv`](tables/failure_replay.csv).

For D2-T, five rollouts failed. Four crossed the clear positional-divergence threshold (`0.08 m`): two first diverged in waiting/yielding, one in a coordination transition, and one in final-goal approach. Every one of these four states was outside the fixed support threshold and had no nearby D2-T recovery sample. The fifth was an agent collision at step 302 without first crossing the `0.08 m` threshold, demonstrating a small-error collision case that the coarse divergence event misses.

All four clear-divergence states and all five last-safe pre-collision states were successfully recovered by the scenario-local expert continuation. Therefore the failures are not demonstrated physical irrecoverability and remain consistent with missing local recovery support. The fresh centralized expert initialized from the arbitrary divergence state succeeded in only 1/4 cases; this reflects the current scheduled-start planner's weakness when restarted from off-schedule states. The local continuation's 4/4 success is the relevant recoverability evidence, but should not be overstated as a fully general replanning result.

For full D2, 9 rollouts failed. Eight had clear divergence: five near-goal termination, one final-goal approach, one transition, and one waiting state. All eight were outside calibrated support with no nearby recovery example; local expert continuation recovered 8/8. All 9 pre-collision states were locally recoverable. The remaining weak-family wall collision at step 344 occurred without crossing the `0.08 m` threshold.

These replays distinguish the present case from a supported-state/model-capacity failure: no clear-divergence failure occurred inside strong support. They also show why full D2's 100-step metric is misleading—the dominant errors now emerge hundreds of steps later, especially at terminal entry.

## 8. Stage-I decision

**REVISE — recovery coverage clearly helps but remains incomplete.**

D2-T supplies a meaningful causal improvement but reaches only 7/12 and remains weak in clearly asymmetric (1/4) and near-symmetric (2/4) regimes. Its OOD fraction does not fall below D1, and five collision failures remain. Full D2 reaches only 3/12. No arm meets the frozen reliability gate, so the canonical Stage-I baseline is not ready for safety-baseline evaluation and no hard-safety, eta/basin, or `G_phi` experiment was started.

## 9. Minimal next step

Run one bounded Recovery V2.1 coverage iteration using **D1 + the accepted transition arm only** as the starting dataset:

1. collect small, trajectory-connected recovery shells around analogous *training/development* rollout failures in waiting/release, final approach, and near-goal entry;
2. match the D2-T residual support distances and include the small-error collision neighborhood, while retaining the same perturbation bounds and expert-success filters;
3. do not add more full velocity-family trajectories; if initial velocity must be tested again, add only initial-state/short-prefix examples so that the intervention is not diluted;
4. do not reuse the present held-out states for training—because they informed this diagnosis, treat them as development cases and create a new untouched three-regime held-out set for the next gate;
5. keep MACFlow, normalization, optimizer, sampler, update budget, and inference semantics fixed and rerun the same closed-loop and long-terminal diagnostics.

This is the smallest change supported by the evidence. If that bounded iteration puts clear divergences inside strong recovery support but the policy still chooses incorrect actions, then representation/capacity becomes a justified next hypothesis. Current results do not yet justify changing the Stage-I formulation.

## Cleanup and repository status

- Accepted scenario-specific infrastructure remains in `double_bottleneck/recovery_v2.py`; it is dataset-generation code, not policy code.
- Reproducibility tools remain under `diagnostics/double_bottleneck_recovery_v2/tools/`; generated datasets, manifests, checkpoints, numerical summaries, plots, and trajectories are retained.
- D2-G and D2-V are rejected interventions, but their immutable artifacts are retained as required negative experimental results. No rejected hook was added to canonical policy or rollout code.
- No temporary explicit-mode, persistent-noise, recurrent, horizon, safety, eta, or `G_phi` implementation exists.
- Toy Give-Way was not modified.
- Canonical `flowbc_4a_agent.py` retained its frozen SHA-256 hash.
- Final regression after generation/training/evaluation: **103 tests passed** across the maintained main suite (58), Toy regression (2), Double-Bottleneck integration suite (19), and scenario unit suite (24, including the three Recovery V2 tests). The known unavailable CUDA-plugin warning was followed by the expected CPU fallback.
- The worktree was already non-clean before this phase, with scenario and historical diagnostic paths untracked plus pre-existing `README.md` and `scripts/test.sh` modifications. No reset or deletion of those user changes was performed. The phase-specific artifacts are isolated under this diagnostic directory and the scenario-specific V2 module/test.
