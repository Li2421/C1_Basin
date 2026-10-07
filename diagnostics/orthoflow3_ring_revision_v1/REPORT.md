# OrthoFlow3 Ring revision v1

## Executive conclusion

This phase does **not** validate the joint learned pipeline on Ring. The fair fixed eta selected solely on train/development evidence is robust on 57/60 fresh Ring states, while the frozen generator/critic pipeline reaches 39/60 with one unresolved numerical case. Ring therefore remains the blocker even though the hard-safety implementation defect was repaired.

- Ring pipeline: `RING_NEEDS_REDESIGN`
- Fixed vs learning: `FIXED_CONTROL_SUFFICIENT`
- Overall: `PIPELINE_NOT_READY`

No mode labels, critic eta optimization, eta-domain change, model retraining, or fresh-test adaptation was used.

## 1. B1 fairness

The old Double-Bottleneck B1 was fair. The old Four-Way and Ring B1 values were test-informed: their eta search/cross-state coverage populations included the same safe-failure states later scored in the old frozen test. Those old numbers are not used for the fixed-vs-learning conclusion.

New B1-FAIR values were selected before the fresh test, from train/development evidence only:

| Scenario | B1-FAIR eta | Train/dev coverage |
|---|---|---:|
| Double-Bottleneck | `[0.514630, 0.442564, 0.001143]` | 85/85 |
| Four-Way | `[0.529167, -0.011185, 0.010672]` | 80/80 |
| Ring | `[0.880612, 0.028541, 0.024503]` | 80/80 |

Full provenance is in `B1_PROVENANCE_AUDIT.md` and `fair_fixed_eta.json`.

## 2. Ring hard-safety audit

All 14 old B0 collisions were outer-boundary collisions. The Ring safety adapter encoded 48 central-obstacle edges but omitted the physical outer boundary. Every old projection satisfied its supplied constraints, so the root cause is S5: a missing geometry constraint, not tolerance drift or CBF violation.

The minimal generic fix adds a conservative 48-edge inscribed outer polygon to the same wall-CBF snapshot. Equations, tolerances, physical collision definitions, and per-state parameters were unchanged. Identical replay of all 14 precursors produced 0 collisions, and the fresh evaluation has 0 collision seeds for every Ring method. See `RING_SAFETY_AUDIT.md` and `ring_safety_collision_traces.json`.

## 3. Train/development diagnosis and frozen revision

Ring robust eta evidence is often disconnected (56/80 states), but the extra component is usually the isolated eta=0 no-op point rather than two nonzero coordination modes. There is no evidence that a mean between two CW/CCW-like eta components is the primary cause.

K scaling saturated at K=4 on train/dev:

| stochastic K, plus mean | oracle robust coverage proxy | robust proposals/state | relative score cost |
|---:|---:|---:|---:|
| 1 | 79/80 | 1.94 | 2 |
| 4 | 80/80 | 4.69 | 5 |
| 8 | 80/80 | 8.46 | 9 |
| 16 | 80/80 | 15.94 | 17 |

Adding eta=0 changed neither rescue nor break and was selected zero times by the critic, so it was not adopted. The train/dev finite-set critic had top-1 accuracy 1.0, pairwise accuracy 0.972, and zero mean regret. Therefore neither R3 nor R4 was triggered:

- `K_FINAL=4`
- eta=0 candidate: no
- generator retraining: no
- critic retraining: no
- added train/dev labels: 0

The frozen proposal set remains generator mean plus four deterministic stochastic proposals. Details are in `TRAIN_DEV_DIAGNOSIS.md`.

## 4. Fresh untouched evaluation

Manifests and complete proposals were frozen before outcomes. Ring uses the required full 60 fresh states. Because no learned weights changed and only Ring proposal/safety logic was under revision, Double/Four use 24-state confirmation cohorts. Every candidate was evaluated with all 16 seeds because exact Q and oracle regret were required.

### Robust states (`Q16 >= 15/16`)

| Scenario | B0 hard safety | B1-FAIR | B2 mean | B4 oracle K=4 | B5 critic |
|---|---:|---:|---:|---:|---:|
| Double-Bottleneck (24) | 0 | 24 | 22 | 24 | 24 |
| Four-Way (24) | 1 | 24 | 24 | 24 | 24 |
| Ring (60) | 35 | 57 | 7 | 45 | 39 + 1 unresolved |

### Ring rescue, break, Q and audits

| Method | Rescue among 25 B0 non-robust | Break among 35 B0 robust | Mean Q16 lower | Collision seeds | Numerical seeds |
|---|---:|---:|---:|---:|---:|
| B0 | 0/25 | 0/35 | 0.7479 | 0 | 0 |
| B1-FAIR | 23/25 | 1/35 | 0.9854 | 0 | 1 |
| B2 mean | 2/25 | 30/35 | 0.3552 | 0 | 2 |
| B4 oracle | 21/25 | 11/35 | 0.8531 | 0 | 1 |
| B5 critic | 16/25 | 12/35 | 0.7917 | 0 | 2 |

The fresh Ring oracle ceiling (45/60) is below the old 47/60 result, not materially above it. B5 misses six oracle-robust states: five are certified non-robust and one is numerically unresolved. Four selections are clear high-score/low-Q exploitation cases. Thus both proposal generalization and critic ranking generalization remain deficient.

For protection cohorts, B5 equals B4 at 24/24 Double; on Four both are 24/24, with B5 mean-Q gap 0.0078 and three numerical seed failures. No catastrophic regression is seen outside Ring.

## 5. Why Ring underperforms

The primary observed failure is **insufficient proposal coverage on fresh Ring states**, compounded by critic generalization error. Generator mean is especially unsuitable (7/60), and even the oracle over the finite K=4 set reaches only 45/60. This is not supported as a simple mean-of-two-nonzero-components effect, nor did train/dev show K=8/16 benefit. The mismatch is therefore a held-out generalization limitation that the current train/dev cohort and label-space proxy did not expose; it cannot be repaired using this fresh test.

The previous seven old oracle/critic misses were all ranking errors: three narrow 16/16-versus-14/16 threshold misses and four substantial robust/non-robust confusions. The fresh test reproduces this failure class. See `OLD_RING_CRITIC_MISS_AUDIT.md` and `critic_exploitation_audit.json`.

## 6. Fixed control versus learning

B1-FAIR matches B5 on Double and Four confirmation cohorts and strongly exceeds it on Ring (57 vs 39 robust states), with much lower Ring break (1 vs 12). Under this experiment's fresh populations, the correct conclusion is `FIXED_CONTROL_SUFFICIENT`.

This does not establish that one universal eta solves every future geometry; it establishes that the present learned pipeline does not add value over the preregistered fair scenario-level fixed controls.

## 7. Numerical, cache, and provenance audit

- DB integrity: `ok`
- foreign-key violations: 0
- database rollouts after run: 1,425,462
- requested fresh seed tuples: 12,096
- successful exact cache entries after run: 12,041
- remaining 55 cache misses: persisted numerical-failure tuples, not silently fabricated outcomes
- physical executions including identical retries: 12,529
- fresh collisions: 0 for all reported methods
- generator/critic and OrthoFlow3 hashes unchanged
- Double/Four environment and shared safety hashes unchanged
- Ring safety adapter hash changed intentionally for the audited outer-boundary fix

Regression results: 21 basis/Four/Ring tests plus 19 Double-Bottleneck tests passed; failures 0.

## 8. Deformation branch

The earlier Four-Way Min-Def benefit remains preserved. The Ring Min-Def diagnostic was not repeated because its explicit prerequisite—acceptable repaired B5 generalization—failed. Optimizing deformation before restoring robustness would confound the diagnosis.

## 9. Required answers

1. Old B1 fairness: Double fair; Four-Way and Ring test-informed.
2. Ring B0 collisions: omitted outer-boundary constraints in the Ring safety adapter (S5).
3. Ring generator failure: primarily insufficient fresh-state proposal coverage/held-out generalization; not demonstrated to be a mean-between-two-nonzero-modes issue.
4. Selected K: 4 stochastic proposals plus generator mean.
5. eta=0 added: no; train/dev ablation showed no material benefit.
6. Generator retraining: no.
7. Critic retraining: no.
8. Previous seven misses: three narrow strict-threshold ranking errors, four substantial robust/non-robust misrankings.
9. Fresh Ring B0/B1-FAIR/B2/B4/B5: 35/60, 57/60, 7/60, 45/60, 39/60 plus one unresolved.
10. B5 rescue/break: 16/25 and 12/35.
11. Critic exploitation: four clear high-confidence/low-Q cases; not negligible.
12. Collision/numerical: zero fresh Ring collisions; B5 has two numerical seed failures and one unresolved state.
13. Learning over fair fixed eta: no; fair fixed control is sufficient and stronger in this comparison.

## Final decisions

`RING_NEEDS_REDESIGN`

`FIXED_CONTROL_SUFFICIENT`

`PIPELINE_NOT_READY`
