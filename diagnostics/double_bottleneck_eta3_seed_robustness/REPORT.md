# Double-Bottleneck P0-3D MACFlow seed-robustness audit

## 1. Frozen protocol and completeness

This study reused only the 58 eta points already observed as successful in the completed common-256 P0 search for the 38 basin-positive timeout episodes. No new global eta point, domain expansion, representation, training, or controller change was introduced. Because every frozen basin contained at most 4 successful points, all successful points were tested; the top-8 truncation rule was never invoked.

Completed evaluations:

| Stage | Rollouts | Completion |
|---|---:|---:|
| Candidate eta × 16 MACFlow seeds | 928 | 100% |
| Robust-center × 24 controls × 4 seeds (7 unique centers, deduplicated) | 672 | 100% |
| Robust-center cross-state screen (7 unique centers × 61 states) | 427 | 100% |
| Top-five reuse confirmation | 392 | 100% |
| Local robust-basin confirmation | 0 | not eligible: no `Q_max >= 0.50` |
| **Total scientific rollouts in this phase** | **2419** | **100%** |

All frozen hashes matched before and after execution. The regression suite passed 79/79 tests. Toy Give-Way, S-XL-128 MACFlow, the environment, hard-safety projection, canonical P0-3D implementation, eta domain, and common-256 source results are unchanged.

## 2. Candidate seed robustness

The deterministic candidate rule evaluated all 58 frozen successful `(episode, eta)` pairs with seeds 2001–2016. Per episode, `eta_robust` maximizes the 16-seed success fraction and uses the registered neighbor-count, distance-to-zero, and eta-index tie-breaks.

| Robustness category | Fixed criterion | Episodes |
|---|---|---:|
| Strongly robust | `Q_max >= 0.75` | 0/38 |
| Moderately robust | `0.50 <= Q_max < 0.75` | 0/38 |
| Weakly robust | `0.25 <= Q_max < 0.50` | 19/38 |
| Seed-fragile | `Q_max < 0.25` | 19/38 |

Median `Q_max` is **0.21875**. The maximum observed `Q_max` is **0.3125**; no episode reached the `0.50` robust-center threshold. The earlier single representative's median 8-seed robustness of 0.125 was therefore not solely a poor representative-selection artifact: exhaustive testing of all frozen successful Sobol candidates raises the median best value only to 0.21875.

Full per-candidate seed outcomes are in `Q_seed.json`; deterministic centers are in `eta_robust.json`.

## 3. Local robust-basin confirmation

The protocol permits the 16-neighbor × 8-seed local test only when `Q_max >= 0.50`. There were **0/38** eligible episodes, so no local eta was generated or evaluated. Accordingly, the requested median local robust-basin fraction is **N/A**, not zero. This preserves the pre-registered stop rule and avoids post-hoc refinement of seed-fragile points.

## 4. Control preservation

Across the 38 episode-selected centers, the median success-control preservation probability is **0.1875** over 24 controls × 4 seeds. Across the seven unique centers it is 0.1458. No unique center preserved any control on all four seeds for every control; the best unique mean preservation probability was 0.3021.

Across the seven unique centers, the 672 control trials partition into 105 successes and 567 safe timeouts, with zero collisions.

Thus the eta values that weakly recover a particular timeout case also degrade frozen baseline-success behavior frequently. This is characterization, not deployment optimization.

## 5. Cross-state reuse

The best fixed-seed screen result is eta index **255**, which rescues **14/61** timeout states at seed 5001. In the required 8-seed confirmation on those 14 screen-positive states, only **3/61** states retain `Q >= 0.50`, and none reaches `Q >= 0.75`. Its mean seed success over the 14 screen-positive states is 0.2679.

The other top-four screen eta values rescue 11, 9, 8, 7 states respectively, but none has a screen-covered state with confirmed `Q >= 0.50`. The apparent shared single-seed reuse therefore contracts sharply under stochastic replication.

## 6. Projection coupling

No strongly or moderately robust center exists, so a direct strong-versus-fragile projection comparison is unavailable. For weak centers, mean raw/executable correction norms are 0.5306/0.1978 m/s, mean projection-removal norm is 0.4490 m/s, and >50% removal occurs on 98.89% of steps. For seed-fragile centers the corresponding values are 0.5245/0.2054 m/s, 0.4445 m/s, and 99.13%.

These descriptive differences do not establish causality. They show that neither category obtains a robust outcome despite the second projection producing nonzero executable corrections.

## 7. Behavioral consistency

Across the 16 seeds for each selected center, the median within-episode horizon-censored timing standard deviations are:

| Timing quantity | Median within-episode std (s) |
|---|---:|
| First-bottleneck clearance | 5.729 |
| Second-bottleneck clearance | 2.685 |
| Total waiting (agent-seconds) | 18.429 |
| Final goal entry | 1.823 |
| Completion | 1.823 |

The selected corrections do not create a consistently successful behavioral outcome across Flow samples. Episode-level timing and missing-event counts are preserved in `timing_consistency.json`.

## 8. Collision and safety sanity

All **2419** scientific rollouts in this phase were collision-free: **YES**. Wall collisions: 0; agent collisions: 0. Hard safety remains effective under every tested candidate, control, reuse-screen, and reuse-confirmation rollout.

## 9. Interpretation

The frozen common-256 search established deterministic existence in 38/61 cases, but existence under one MACFlow realization does not translate into a genuinely seed-robust P0 region under this protocol:

- all 58 already-successful candidates were tested, eliminating candidate-selection incompleteness within the frozen basin data;
- 0/38 episode-wise best candidates reach 50% success over 16 new seeds;
- half are weak (observed `0.25–0.3125`) and half are seed-fragile (`<0.25`);
- no center qualifies for local robust-neighborhood confirmation;
- cross-state reuse and control preservation both collapse under repeated Flow sampling.

The evidence therefore does **not** support training a state-to-P0 mapping yet: its labels would be based mainly on seed-sensitive episode-level successes. This phase does not determine whether the cause is fixed episode-level coefficients, missing basis directions, or stochastic MACFlow/control interaction.

## 10. Decision

**REVISE — some P0 success is repeatable, but the geometry remains too seed-fragile for a robust 3D learning target.**

The fixed taxonomy has a numerical gap here: zero episodes reach `Q_max >= 0.50`, but only 19/38—not “almost all”—fall below 0.25. Therefore the explicit REJECT condition is not met, and the conservative protocol status is REVISE. No representation expansion or G_phi training is performed in this task.

## 11. Required final numbers

1. `P0-positive episodes = 38/61`.
2. `Strongly robust count = 0/38`.
3. `Moderately robust count = 0/38`.
4. `Weakly robust count = 19/38`.
5. `Seed-fragile count = 19/38`.
6. `Median Q_max = 0.21875`.
7. `Median local robust-basin fraction among Q_max >= 0.5 cases = N/A (0 eligible cases)`.
8. `Best robust eta cross-state coverage = 3/61 at Q>=0.50 after 8-seed confirmation` (single-seed screen: 14/61).
9. `Median control-preservation rate = 0.1875`.
10. `All tested rollouts collision-free: YES`.

Stop condition satisfied. No G_phi, Agent6, Pair8, Temporal6, P0 change, eta-domain change, MACFlow change, safety change, or follow-on experiment was launched.
