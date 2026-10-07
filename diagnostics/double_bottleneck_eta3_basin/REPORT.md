# Double-Bottleneck fixed 3D eta success-basin evaluation

Date: 2026-09-24  
Scope: frozen S-XL-128 Stage-I MACFlow, frozen four-agent hard-safety filter, and the existing fixed episode-level three-dimensional diagnostic corrector. No model training, `G_phi`, eta-dimension change, basis redesign, horizon change, or success-definition change was performed.

## 1. Frozen controller and implementation audit

Every rollout used the exact pipeline

```text
u_flow = MACFlow(x)
u_safe = Pi_U(x)(radial_bound_0.5(u_flow))
g^eta   = eta1 B_goal + eta2 u_safe + eta3 B_rel
u_exec = Pi_U(x)(u_safe + g^eta)
```

with one eta fixed for the entire episode. The bases were recomputed from the current state at every physical step:

- `B_goal,i = radial_bound_0.5(goal_i - position_i)`;
- the second basis is the first-projected action `u_safe,i`;
- `B_rel,i = (1/(N-1)) sum_(j!=i) radial_bound_0.5(position_i-position_j)`;
- pairwise relative vectors are bounded before averaging;
- no clipping is applied to raw `g^eta`;
- the second projection enforces the same walls, all six agent pairs, per-agent `0.5 m/s` speed balls, and action semantics as the frozen hard-safety baseline.

For `N=2`, the relational mean has one term and exactly reduces to the original single-opponent basis. Each basis row has norm at most `0.5`; the four-agent mean therefore does not inflate the Toy relational scale.

The frozen hashes all match their pre-experiment values:

| Frozen object | SHA-256 |
|---|---|
| S-XL-128 checkpoint | `6e2ed4e31443bbb34457d3b3aabe0e7d741b904259391f8893b1104c143546bd` |
| S-XL-128 dataset manifest | `771b575f5641562a4b4631da02c70ac06fea993c29c1f2fb802b10e3d17c6c56` |
| MACFlow source | `02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8` |
| environment | `3159b98f180f18d2f270d2b093e547d7d9f3c9f5b25347fb60d42b3ada149cdc` |
| hard projection | `847f7045ffb617f403abb5af3a4edd092c70eef7734e819d42718c3391b0ff79` |
| 3D corrector | `48f73555d542d77581852450edb0d0c7d9c582ff9262d063384df88b08dadf40` |

## 2. Global eta domain and pre-registered search

The same global domain was used for every target and control:

```text
eta1 in [ 0.50, 1.25]   goal feedback
eta2 in [-0.50, 0.50]   safe-action feedback
eta3 in [ 0.00, 0.75]   relative feedback
```

This is the exact closed box spanned by the retained Toy Give-Way SBMA Phase-A lattice. It remains scale-compatible because every basis row is bounded by `0.5`. The largest possible raw correction-row norm bound is `1.25 m/s`. `eta=(0,0,0)` was evaluated separately as an out-of-domain baseline sentinel and excluded from basin fractions.

The protocol was frozen before scientific evaluation in `PREREGISTRATION.json`:

- **Stage A:** 32 scrambled Sobol points, seed `351903`, plus the Toy reference `(1,0,0.25)`; the same 33 points for all 85 episodes.
- **Targets:** all 61 frozen hard-safety timeouts.
- **Controls:** 24 uniformly sampled frozen hard-safety successes, without replacement, PCG64 seed `832041`.
- **Stage B after a Stage-A success:** the fixed 14-point neighborhood (`±1/16` normalized on each axis and eight cube corners) around the successful point nearest the Toy anchor.
- **Stage B after no Stage-A success:** the next 32 points of the same nested scrambled Sobol design over the unchanged box.
- **Stage C:** the same 14-point local rule only if the dense Stage-B global set first found success. No episode met this condition, so Stage C contained zero jobs.

The search executed 4,572 complete rollouts: 85 eta-zero sentinels, 3,695 nonzero target rollouts, and 792 nonzero control rollouts. Eta-zero reproduced all 61 frozen timeouts and all 24 frozen control successes exactly, including episode length. No eta rollout had a wall or agent collision.

An initial numerical attempt stopped because canonical Clarabel candidates labeled `Solved` exceeded the frozen external speed certificate by roughly `5e-10` to `8e-10`; its partial results are isolated in `invalid_numeric_attempt_v1/` and excluded. The final run always called the unchanged canonical projector first and, only on numerical rejection, re-solved the identical objective and identical feasible set with tighter tolerances. No slack, clipping, constraint relaxation, or fallback action was introduced. Identical-problem retry was needed on 811 of 7,596,518 projection calls (`0.0107%`); every retained action passed the original certificate.

## 3. Basin existence and geometry

### Primary existence result

Successful eta was observed for:

```text
15 / 61 = 24.59% of frozen safe-timeout episodes.
```

The result replicated across both untouched sets:

| Population | Basin exists | No observed basin |
|---|---:|---:|
| Existing untouched timeouts | 7 / 32 | 25 / 32 |
| Fresh untouched timeouts | 8 / 29 | 21 / 29 |
| Combined | **15 / 61** | **46 / 61** |

The 46 Type-IV episodes were each evaluated at all 65 globally shared points: the Stage-A 33 plus the second nested Sobol 32. None succeeded. The domain was not expanded and no per-case optimizer was used.

### Coarse basin fraction and registered class

Each of the 15 observed basins hit exactly one of the 33 Stage-A points, so every nonzero coarse fraction was `1/33 = 3.03%`. Across all 61 targets, the mean Stage-A basin fraction was `0.745%` and the median was zero.

Under the pre-registered classification:

| Basin class | Episodes |
|---|---:|
| Type I — broad | 0 |
| Type II — narrow | **15** |
| Type III — fragmented / multimodal | 0 |
| Type IV — no observed basin | **46** |

No topology stronger than the sampling resolution is claimed.

### Local robustness

The fixed neighborhoods around the 15 successful centers contained 83 successes among 210 tested neighbors (`39.52%`). Per-episode local success fraction had:

- median `5/14 = 35.71%`;
- mean `39.52%`;
- range `2/14 = 14.29%` to `11/14 = 78.57%`;
- only 5/15 episodes at or above 50% local success.

Thus the observed region is not a collection of isolated single points, but its robust width varies materially with state and is too small to qualify as broad under the registered rule.

![Basin volume and registered classes](figures/basin_fraction_and_types.png)

## 4. Cross-state eta structure

All 15 Stage-A successes occurred at the same shared point:

```text
A004 = [0.5364547465, 0.1891057156, 0.0249288732].
```

This is strong evidence for a shared low-goal-gain, positive-safe-feedback, near-zero-relative region among the recoverable subset. It is not evidence of universal applicability: A004 rescued only `15/61` targets.

Existence is not confined to one original direction or one test set:

| Group | Basin exists / targets |
|---|---:|
| Baseline LTR-first | 7 / 29 |
| Baseline RTL-first | 8 / 32 |
| Clearly asymmetric | 7 / 26 |
| Weakly asymmetric | 3 / 14 |
| Near-symmetric | 5 / 21 |

Successful eta rollouts were not always order-preserving. Among the 98 successful coarse/local target rollouts, 58 retained RTL-first, 23 retained LTR-first, and 17 changed from baseline LTR-first to RTL-first. No mode label or order information was supplied to the controller. This shows that eta can alter the realized coordination solution, but also that the useful region has an RTL-skewed behavioral effect.

![Shared eta coverage and success-control tradeoff](figures/cross_state_coverage.png)

![Eta projections colored by cross-state coverage](figures/eta_2d_projections.png)

## 5. Baseline-success preservation

A004 preserved only `6/24 = 25%` of the randomly pre-registered baseline-success controls; the other 18 became safe timeouts. Every other Stage-A eta preserved 0/24 controls. Across all 792 control/eta pairs:

| Outcome | Count |
|---|---:|
| Success | 6 |
| Safe timeout | 734 |
| Runtime strict deadlock | 52 |
| Collision | 0 |

Eta zero preserved all 24 controls exactly. Consequently no tested fixed nonzero eta offered a good global rescue/preservation tradeoff. A later state-dependent selector could in principle choose eta zero on already-successful states, but this experiment does not establish that such a selector is learnable, and no `G_phi` was trained.

## 6. Correction and second-projection interaction

There were 98 successful target/eta rollouts across coarse and local samples. They completed in a mean 757.18 steps (`37.86 s`) and median 755 steps (`37.75 s`), while remaining collision-free.

| Successful-eta metric | Mean |
|---|---:|
| Raw joint correction `||g^eta||` | `0.5086 m/s` |
| Second-projection correction | `0.4365 m/s` |
| Executed change from `u_safe` | `0.1527 m/s` |
| Second-projection / raw-correction ratio | `85.31%` |
| Second projection active | `99.97%` of steps |
| At least 25% of correction removed | `99.69%` of steps |
| At least 75% of correction removed | `83.39%` of steps |

Minimum clearances among successful eta rollouts were positive: `0.00510 m` to walls and `0.04453 m` between agent surfaces. Across all 4,572 rollouts, the minima were also positive (`0.00510 m` wall, `0.00445 m` agent).

This is not a mild-correction regime. Successful behavior is produced by the feasible residual of a large raw eta command after near-continuous second projection. The second projection guarantees safety as intended, but it makes the raw three-basis correction a weak direct description of the executed action. That interaction is important evidence against treating the current 3D eta coordinates as a clean, broadly learnable action correction.

The eta sweep also produced 287 runtime strict-deadlock terminations (235 target-domain and 52 control rollouts). This does not change the ground-truth label of the original 61 targets: their eta-zero outcomes remain safe finite-horizon timeouts, not validated deadlocks.

## 7. What successful eta changes

Four deterministic representative pairs were saved across all three initial-state regimes. Relative to eta zero:

- last clearance of the second resource moved from `37.7–40.0 s` to `35.15–36.05 s`;
- mean low-speed waiting per agent fell from `22.53–22.86 s` to `16.31–17.15 s`;
- mean time between the two resources changed only modestly, from about `7.35–7.38 s` to `7.05–7.09 s`;
- full completion occurred at `37.55–38.45 s`, leaving 4–5 seconds of horizon margin;
- total terminal goal error fell from `0.216–0.637 m` at timeout to `0.073–0.118 m` at successful termination.

The mechanism is therefore mainly reduced prolonged waiting and earlier release/forward progress throughout the episode, not escape from a verified bottleneck deadlock and not a dramatic speed-up inside the chamber itself. One near-symmetric representative changes the macro direction rather than merely accelerating the eta-zero order.

![Representative eta=0 versus successful eta progress](figures/fresh_untouched_test_077_mechanism.png)

The paired SVG trajectories and compressed time series are retained under `representatives/`. The other representative mechanism plots cover clearly asymmetric, weakly asymmetric, and near-symmetric cases.

## 8. Answers to the six scientific questions

### Q1 — For what fraction does a successful 3D eta exist?

`15/61 = 24.59%` under the fixed domain and pre-registered global/refinement design. For the other 46 episodes, no success was observed after 65 global points over the same domain.

### Q2 — Are observed basins broad enough to be practically learnable?

Not generally. All are registered Type II narrow basins. The local median success fraction is `35.71%`; five of 15 reach at least 50%, but none has broad global support. The successful raw corrections are also mostly rewritten by projection.

### Q3 — Shared or state-dependent?

The observed basins share one coarse center, A004, so there is real cross-state structure. Applicability and local width are strongly state-dependent: the center works for only 15 targets, local success ranges from 14.29% to 78.57%, and 46 targets have no observed basin.

### Q4 — Does one fixed eta generalize across many timeout states?

Only partially. A004 is the best and only rescuing shared Stage-A eta, covering 15/61 targets. It is not a broadly general solution.

### Q5 — Can eta rescue failures without systematically damaging successes?

Not as one fixed tested eta. A004 rescues 15 targets but preserves only 6/24 controls; 18/24 become timeout. Safety is retained—there are no collisions—but task success is not.

### Q6 — Is the present 3D representation sufficient to justify `G_phi(state)->eta`?

No. There is useful low-dimensional structure in a recoverable subset, but 75.41% of targets have no observed basin, all observed basins are narrow, control preservation is poor, and the second projection removes most of the raw correction on most successful steps. Training `G_phi` now would ask it to solve many states for which the frozen output representation has no demonstrated successful target.

## 9. Capacity interpretation and decision

The result is not “eta never works”: one coherent region converts 15 safe timeouts into success, spans both baseline directions and all three regimes, and has nonzero local robustness. However, the primary capacity question is answered negatively for the current frozen representation:

- most target episodes (`46/61`) have no observed basin after both global stages;
- all observed basins are narrow under the registered geometry rule;
- the single useful shared eta damages 75% of baseline-success controls;
- successful raw corrections are substantially altered almost everywhere and mostly rewritten on 83.39% of steps.

**REJECT — 3D representation is inadequate.**

This decision applies to the frozen three-basis, fixed-episode eta formulation over the globally frozen Toy-derived domain. It does not prove that every possible three-dimensional parameterization is impossible, nor does it identify eta dimensionality alone as the sole cause; relation aggregation, basis expressivity, the search domain, and projection interaction remain candidate explanations. Per the task constraint, none was changed here.

The smallest scientifically justified next action is to review this measured capacity failure before specifying a separate representation study. Do not train `G_phi` against the present eta target space, and do not silently enlarge eta or redesign `B_rel` within this experiment.

## 10. Reproducibility, artifacts, and regression

Machine-readable artifacts include:

- `PREREGISTRATION.json`, `global_eta_domain.json`, and `eta_samples.json`;
- `episode_catalog.json` and `jobs/stage_a.json`, `stage_b.json`, `stage_c.json`;
- raw resumable JSONL outcomes under `raw/`;
- `all_rollout_outcomes.json`;
- `per_episode_basin.json`, `per_episode_basin.csv`, and `basin_membership.csv`;
- `baseline_success_controls.json`;
- `per_eta_cross_state_coverage.json`;
- `correction_projection_statistics.json` and `SUMMARY.json`;
- `representative_trajectory_analysis.json`, representative NPZ traces, and SVG trajectories;
- figures under `figures/`;
- `regression_results.json`.

The full maintained regression suite passed 79/79 tests: 58 shared/single-integrator, 2 Toy Give-Way, and 19 Double-Bottleneck. JAX emitted its known unavailable-CUDA warning and ran on CPU; this was not a test failure. Toy Give-Way, S-XL-128 MACFlow, the environment, the hard projection, and the diagnostic corrector remain unchanged. All new executable logic is isolated under this diagnostic directory.
