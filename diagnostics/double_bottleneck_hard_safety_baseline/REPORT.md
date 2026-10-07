# Double-Bottleneck frozen hard-safety baseline

Date: 2026-09-24  
Scope: frozen S-XL-128 Stage-I MACFlow, with and without the existing one-pass four-agent hard projection. No retraining, eta evaluation, basin search, or `G_phi` was performed.

## 1. Frozen S-XL-128 reference

The experiment used the exact selected S-XL-128 checkpoint and the same two previously frozen untouched test pools. The no-safety controller was the canonical radially speed-bounded MACFlow action, with no correction. The safety controller applied exactly one existing global CBF/SOCP projection to that same action:

```text
no safety:   u_exec = u_flow
hard safety: u_exec = Pi_U(x)(u_flow)
```

There was no eta correction and no second projection. The projector used every wall constraint, all six unordered agent-pair constraints, and the unchanged per-agent speed cones. Its configuration was not tuned.

| Frozen item | SHA-256 |
|---|---|
| S-XL-128 checkpoint | `6e2ed4e31443bbb34457d3b3aabe0e7d741b904259391f8893b1104c143546bd` |
| S-XL-128 data manifest | `771b575f5641562a4b4631da02c70ac06fea993c29c1f2fb802b10e3d17c6c56` |
| `flowbc_4a_agent.py` | `02dda46aff97254c04974ade395dc7967bd15706352f7cbee1c34e2e99ad18f8` |
| `environment.py` | `3159b98f180f18d2f270d2b093e547d7d9f3c9f5b25347fb60d42b3ada149cdc` |
| `hard_projection.py` | `847f7045ffb617f403abb5af3a4edd092c70eef7734e819d42718c3391b0ff79` |

The existing set used seeds `[101,211,307,401]`; the fresh set used `[503,607,701,809]`. Each set contains 24 independent initial states and four noise seeds, for 96 rollouts. `rollout_id`, seed, initial state, and random-key construction were paired between controllers. Raw Flow samples use the same key at a matched step, although later sampled actions may differ because projection causes the states to diverge.

The complete protocol was fixed before evaluation in `PREREGISTRATION.json`. The new no-safety runs matched all 192 frozen reference outcomes, episode lengths, and collision flags exactly.

## 2. No-safety baseline

Runtime termination preserves the environment's existing priority: collision, success, strict deadlock, then timeout. Collision subtypes below are mutually exclusive; no same-step wall-and-agent collision occurred.

| Test set | Success | Wall collision | Agent collision | Runtime strict deadlock | Timeout | Failure | Mean / median steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing untouched (96) | 68 | 4 | 6 | 0 | 18 | 28 | 701.23 / 735.5 |
| Fresh untouched (96) | 76 | 1 | 3 | 0 | 16 | 20 | 734.04 / 736.5 |
| Combined (192) | **144** | **5** | **9** | **0** | **34** | **48** | **717.64 / 736.0** |

Thus the frozen no-safety success rate is `144/192 = 75.0%`; collision is `14/192 = 7.29%`; and collision-free timeout is `34/192 = 17.71%`. Minimum swept clearances over the combined set were `-0.00423 m` to walls and `-0.00451 m` between agent surfaces.

Successful no-safety rollouts contain all eight complete passage-order signatures. Their first direction is balanced: 70 LTR-first and 74 RTL-first successes.

## 3. Runtime versus shadow four-agent deadlock

### Canonical runtime result

The runtime detector was not changed. It produced zero strict-deadlock terminations under both controllers on both test sets. All incomplete collision-free horizon endpoints are therefore canonically `timeout`, not deadlock.

### Diagnostic shadow result

`double_bottleneck_resource_frontier_v1` was implemented offline from the prior audit specification and never affected rollout termination. Its thresholds remain unfrozen.

| Controller / set | Shadow triggers | On eventual successes | On runtime timeouts | Success false-positive rate |
|---|---:|---:|---:|---:|
| No safety / existing | 53 | 36 | 17 | 52.94% |
| No safety / fresh | 58 | 43 | 15 | 56.58% |
| Hard safety / existing | 7 | 4 | 3 | 6.25% |
| Hard safety / fresh | 6 | 4 | 2 | 5.97% |

The shadow detector would classify 32 of the 34 no-safety timeouts earlier, but it also triggers on 79 eventual no-safety successes. Most false positives recover after the proposed trigger. This directly demonstrates that the candidate resource/hold thresholds are not valid canonical ground truth. Shadow counts are retained for audit, but they must not be used to claim 4-agent deadlock incidence.

The hard-safety shadow triggers are all post-passage branches; no safety timeout is classified by a resource branch. Only 5 of 61 safety timeouts receive a shadow trigger, while 8 eventual safety successes are also triggered. Consequently the present experiment establishes a safe finite-horizon timeout population, not a validated deadlock population.

## 4. Hard-safety outcomes

| Test set | Success | Wall collision | Agent collision | Runtime strict deadlock | Timeout | Safe liveness failure | Mean / median steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Existing untouched (96) | 64 | 0 | 0 | 0 | 32 | 32 | 838.11 / 843.0 |
| Fresh untouched (96) | 67 | 0 | 0 | 0 | 29 | 29 | 837.78 / 842.0 |
| Combined (192) | **131** | **0** | **0** | **0** | **61** | **61** | **837.95 / 843.0** |

Hard safety eliminated every observed collision: `14 -> 0`. The combined full-task success rate remains meaningful at `131/192 = 68.23%`. Minimum swept clearances become `0.01694 m` to walls and `0.08619 m` between agent surfaces, both positive with margin.

There is no directional collapse: all 192 safety rollouts realize a first direction (95 LTR, 97 RTL), successful rollouts split 66 LTR / 65 RTL, and all eight complete order signatures remain among successes.

## 5. Projection intervention statistics

The preregistered correction is the joint eight-dimensional norm `||u_safe-u_flow||`. Projection is active above `1e-6`; a correction at least `0.1 m/s` is called large, equal to 10% of the maximum possible joint action norm.

| Metric over 160,886 executed steps | Value |
|---|---:|
| Active fraction | 66.35% |
| Large-correction fraction | 38.87% |
| Mean / median correction | 0.0826 / 0.0520 m/s |
| p90 / p95 / p99 correction | 0.2147 / 0.2312 / 0.2543 m/s |
| Maximum correction | 0.3086 m/s |
| Mean / median correction-to-Flow norm | 13.70% / 9.35% |
| p95 / maximum relative correction | 36.82% / 47.29% |
| Mean / median Flow-safe cosine | 0.9883 / 0.9977 |
| Correction at least half of Flow norm | 0 steps |

Solver statuses were 54,127 nominal-feasible, 106,108 solved, and 651 KKT-certified solved steps. No solver failure or fallback occurred.

Correction size is not selectively larger in failures: successful episodes have mean correction `0.0835 m/s`; timeouts have `0.0807 m/s`. Therefore projection is frequent and non-negligible, especially while constraints are nearby, but it does not dominate or reverse the policy action. It is more accurately described as an intervention-heavy safety filter than as either a rare mild correction or wholesale policy rewriting.

## 6. Failure-conversion matrix

Matched combined counts are:

| No-safety outcome | Hard-safety success | Hard-safety timeout | Row total |
|---|---:|---:|---:|
| Success | 100 | 44 | 144 |
| Wall collision | 2 | 3 | 5 |
| Agent collision | 4 | 5 | 9 |
| Timeout | 25 | 9 | 34 |
| **Column total** | **131** | **61** | **192** |

The direct unsafe-to-safe conversion is clear: of 14 no-safety collisions, 6 become success and 8 become safe timeout. Projection also changes long-horizon paths in both directions: 44 no-safety successes become timeouts, while 25 no-safety timeouts become successes. This is why aggregate counts alone are insufficient.

![Terminal outcomes and matched conversion](figures/outcomes_and_conversion.svg)

## 7. Safe-liveness-failure population

The report's runtime safe-liveness definition is objective: no collision, no all-agent task success, and termination by strict deadlock or timeout. Under current semantics all 61 cases are timeout.

Post-hoc trajectory analysis shows:

- all 61 timeouts have historically cleared both bottlenecks for all four agents;
- 42 end with three agents in goal, 17 with two in goal, and 2 with none in goal;
- 50/61 have a stationary subset over the last 5 s, usually agents already complete while remaining agents continue moving;
- the final 84 unfinished-agent errors have median `0.118 m`; 52 are at most `0.15 m`, 75 are at most `0.30 m`, and 9 exceed `0.30 m`;
- the last gate-clearance step has median 746 for timeouts versus 644 for successes, showing substantial accumulated coordination/safety delay;
- timeout tails still have only 29.44% projection activity, mean correction `0.0363 m/s`, and mean removed goal-directed component `0.0086 m/s`.

These facts rule out describing the entire population as bottleneck mutual deadlock. It is a mixed finite-horizon liveness population dominated by delayed post-passage/goal completion. Some cases are near-threshold terminal failures; others remain materially far from a goal. The population is scientifically useful for a success-basin question, but not yet for a paper claim specifically about deadlock.

## 8. Representative trajectory analysis

Representatives were selected only after all outcomes were frozen, using the first rollout in each available conversion cell.

- `success -> timeout`, existing id 10: three agents finish; the last agent ends at approximately the goal threshold. Tail projection is active only 12%, while unfinished agents still make large net progress. This is deadline/terminal completion, not a stationary resource deadlock.
- `agent collision -> timeout`, existing id 2: safety removes the collision; three agents finish and the fourth ends at `0.084 m` goal error. Tail projection activity is 26%.
- `wall collision -> timeout`, existing id 80: safety removes the wall collision; three agents finish and the last remains `0.178 m` from goal.
- `timeout -> timeout`, existing id 24: two agents finish; the remaining errors are `0.093 m` and `0.086 m`, just outside tolerance.
- `agent collision -> success` and `timeout -> success` cases demonstrate that the projection can produce a safe, successful alternative path rather than only slow the controller.

![Representative matched trajectories](figures/representative_trajectories.svg)

The representative and aggregate tail evidence do not show projection repeatedly canceling all forward motion at a bottleneck. They show accumulated path changes and delayed completion, with occasional post-passage stagnation. No controller change was made from this analysis.

## 9. Basin-readiness assessment

1. **Does hard safety eliminate collisions?** Yes: `14/192 -> 0/192`, with positive minimum wall and agent clearances.
2. **Does it preserve meaningful task success?** Yes: `131/192 = 68.23%`, balanced across directions and all eight successful order signatures.
3. **Does it expose safe liveness failures?** Yes: 61 objective collision-free timeouts, including 8 direct collision-to-timeout conversions.
4. **Are they nontrivial rather than evidence of a broken baseline?** Yes for a finite-horizon success-basin study. The frozen policy still succeeds in over two thirds of cases, projection is numerically certified, and failures include reproducible late coordination/goal-completion limits. They are not, however, validated bottleneck deadlocks.
5. **Is projection rewriting the policy almost everywhere?** No. It activates frequently, but the relative correction has 9.35% median, never reaches half the Flow norm, and preserves action direction closely.

The later basin target is therefore well-posed as **safe full-task failure within the frozen 850-step horizon**. A deadlock-specific interpretation remains blocked by the unfrozen shadow detector's false positives and should be handled as a separate semantics task.

## 10. Decision

**PASS — ready for basin analysis.**

The only recommended next phase is:

> proceed to 3D eta success-basin existence/capacity evaluation on the frozen hard-safety baseline.

This recommendation does not authorize a deadlock-ground-truth claim, eta dimensionality expansion, or `G_phi` training.

## Reproducibility and regression

Machine-readable artifacts:

- `no_safety_existing_untouched_test_outcomes.json`
- `no_safety_fresh_untouched_test_outcomes.json`
- `hard_safety_existing_untouched_test_outcomes.json`
- `hard_safety_fresh_untouched_test_outcomes.json`
- `no_safety_summary.json`
- `hard_safety_summary.json`
- `matched_episode_conversion_matrix.json` and `matched_episode_pairs.csv`
- `projection_statistics.json`
- `shadow_deadlock_analysis.json`
- `safe_timeout_liveness_analysis.json`
- `representative_trajectory_analysis.json`

The maintained full regression command passed 79/79 tests: 58 shared/single-integrator, 2 Toy Give-Way, and 19 Double-Bottleneck. JAX emitted its known unavailable-CUDA warning and ran on CPU; this was not a test failure. Canonical S-XL-128, environment, projection, and authoritative Toy hashes match their pre-experiment values. Only isolated diagnostic scripts/results were added; Toy and canonical MACFlow source were not modified.
