# Why useful controller context has not produced stable robust selection

## Answer

Controller identity/behavior is a necessary condition in the previously demonstrated controller-swap counterexample. The current physical context contains useful information about that condition. However, these are different statements from saying this finite-horizon summary is sufficient, or that the trained network can reliably identify the best robust candidate.

The new source-only diagnostics identify three concrete issues:

1. Most probability improvement occurs outside the strong candidates competing for top-1.
2. The decisive two-candidate TRAIN contrast is weakly measured; state variation in that contrast is not reliably resolved by Q4.
3. The network has a controller-specific candidate-preference fitting bias. Correcting average preferences alone does not solve the needed state-dependent switches.

The existing seven-family benchmark also has only one certified state-adaptive success of headroom beyond a TRAIN-derived controller-fixed baseline. These findings do not establish that no transferable law exists, or that the current context is sufficient.

## Frozen scope

- Same large-H20 source data: Ring, three known controllers, 46 TRAIN and seven VAL families, 16 exact candidates per case.
- All original checkpoints and generator remain frozen. No target-controller or LOSO labels opened.
- NEW ROLLOUT = 0. Global DB reads use exact compatible, nonquarantined source keys only.
- The source VAL already selected checkpoints; this is post-hoc localization, not new confirmatory generalization.
- Candidate row indices below are file indices, not artificial canonical modes.

## 1. Where probability gains occur

For each known controller, define its top three candidates using **TRAIN counts only**. These are fixed before this decomposition and do not use VAL performance.

| Training seed | eta-only NLL | Full h+C+eta NLL | Fraction of NLL improvement on the other 13 candidates |
|---|---:|---:|---:|
| 17 | 0.34114 | 0.24790 | 94.3% |
| 23 | 0.34132 | 0.25375 | 91.2% |
| 41 | 0.34148 | 0.23749 | 92.8% |

The other 13 candidates are 81.25% of candidate cells. Therefore this accounting does not alone prove harmful weighting. It shows that the large aggregate NLL gain is not evidence of a similarly large gain in the decisive top-candidate comparisons.

A TRAIN-derived known-controller × exact-eta prior, with no state input, has NLL 0.26115 and B15 16/17. This explains much of the gap to eta-only's 0.341, though it is a different, privileged diagnostic model—not an exact causal decomposition and not a zero-shot baseline.

Thus context is doing useful work distinguishing controller-dependent feasibility over the field. Most of that work does not change which robust candidate wins.

## 2. The decisive contrast is weakly supervised

Under the third source controller, the relevant pair is dataset eta rows 1 and 3. Each has four observed seeds on each of 46 TRAIN families:

| TRAIN evidence | eta row 1 | eta row 3 |
|---|---:|---:|
| Successes/trials | 158/184 | 155/184 |
| Families with higher empirical Q than the other eta | 9 | 9 |

They tie in 28 families. The observed mean preference difference is +0.0163 for row 1, with family-bootstrap 95% interval [-0.0489, 0.0815]. Matched-seed discordances are 19 versus 16; the exploratory pooled paired test is p=0.736.

More importantly, split the four seeds into two plus two and recompute the per-state *difference between these eta*. Its split-half correlation is only 0.053, with covariance interval spanning zero. This differs from the approximately 0.600 split-half correlation of state–eta interaction aggregated across the whole panel in the previous audit.

Hence there can simultaneously be substantial learnable variation over all 16 eta and poorly resolved supervision for the particular contrast that decides top-1. Low contrast repeatability does not prove its true state dependence is absent; Q4 may simply be too imprecise here.

## 3. Frozen network inference finds actual fitting bias

The checkpoint was reloaded and run on CPU using its saved normalization. Recomputed VAL logits match the previously saved logits within 1e-5. No retraining occurred.

| Seed | TRAIN predicted mean Q, row 1 | TRAIN predicted mean Q, row 3 | Predicted average preference |
|---|---:|---:|---:|
| 17 | 0.8486 | 0.9064 | -0.0577 |
| 23 | 0.8512 | 0.8312 | +0.0200 |
| 41 | 0.8182 | 0.8701 | -0.0519 |

The observed TRAIN rates are 0.8587 and 0.8424. Seeds 17 and 41 reverse their empirical mean preference and favor row 3 in 45/46 TRAIN states. The same two models choose row 3 for every third-controller VAL state. This is not merely a mismatch that first appears on an unseen controller.

For example, at VAL state index 50, row 1 is 15/16 and row 3 is 12/16. Seed 17 assigns them 0.902 and 0.938 respectively. The model has learned useful field structure, but its local preference bias selects the non-B15 candidate.

There is also a genuine need to switch: at VAL state 51, the same pair reverses, row 1=12/16 and row 3=16/16. Recommending row 1 everywhere therefore does not solve state-conditioned selection.

Among five VAL states with complete Q16 for both candidates, predicted versus observed contrast correlations are 0.75–0.78. This tiny post-hoc subset is not evidence of reliable generalization; it illustrates how some relative state signal can coexist with a biased decision threshold. Correlation alone does not guarantee the correct sign of each candidate comparison.

## 4. Direct test: does average-preference repair suffice?

Fit one bounded logit intercept per known controller/exact eta, uniformly across all 48 groups, using **TRAIN observed-count NLL only**. No VAL outcome or ranking loss chooses an intercept. Bounds [-8,8] prevent infinite estimates for groups with all failures.

This is deliberately a privileged, fixed-panel diagnostic. It is not a new continuous critic, cannot generalize to unknown controllers or arbitrary eta, and is not promoted into the pipeline.

| Seed | TRAIN NLL before → after | VAL NLL before → after | Certified VAL B15 before → after |
|---|---:|---:|---:|
| 17 | 0.2202 → 0.2026 | 0.2479 → 0.2489 | 15 + 1 unresolved → 15 |
| 23 | 0.2493 → 0.2333 | 0.2537 → 0.2468 | 17 → 16 |
| 41 | 0.2108 → 0.2000 | 0.2375 → 0.2381 | 15 + 1 unresolved → 15 |

For seeds 17 and 41, the third controller now chooses row 1 everywhere. This repairs one known failure but loses the state requiring row 3. Other controller choices can also break. Average preference correction is therefore insufficient; the state-dependent change of preference still matters.

This test demonstrates that the original network did not fully minimize even some simple conditional TRAIN likelihood directions. It does **not** prove the original loss is inappropriate, nor does improving TRAIN NLL guarantee independent decision gains.

## What can be said about C being “key”?

1. **Controller behavior is causally relevant:** established by the previous same-h/eta controller intervention.
2. **Current C carries relevant information:** supported by NLL gains, wrong-context degradation and state-profile permutation diagnostics.
3. **Current C is sufficient for robust choice or transfer:** not established.
4. **The model reliably exploits all needed C×h×eta interactions:** not established.

The first two do not entail the last two. In an ideal population setting the richer predictor can ignore unhelpful inputs, but a strict decision improvement additionally requires changing optimal actions, enough identifiable supervision, successful estimation and a discriminative test distribution.

## Root-cause ordering for the present source-selection problem

1. **Decision-critical supervision and learned contrast mismatch — direct evidence.** The top-candidate contrast has low Q4 repeatability; two initializations impose the same wrong coarse preference across most states. Data precision and fitting/generalization are both implicated; neither is fully isolated.
2. **Probability-field benefit is not the same as decision benefit — direct evidence.** 91–94% of NLL improvement is outside the TRAIN top three, and a controller-fixed rule already achieves 16/17.
3. **Finite context completeness/support — still underresolved.** A one-second summary can distinguish controllers without revealing all relevant future 35-second interactions. No new input-aliasing counterexample or sufficiency proof was constructed here. Redundant features are not established as the cause.

The old catastrophic LOSO Ring result and this current known-controller near-tie problem must not be conflated. This audit diagnoses the latter and does not claim to resolve cross-scene transfer.

## Minimal next discriminating experiment

First resolve the supervision for decision-relevant source TRAIN contrasts, not another broad architecture sweep. For the diagnosed controller and two eta, Q4→Q16 across the same 46 TRAIN families would request at most 1,104 additional seeds before cache preflight. This is a **plan only**, not executed. All canonical signatures must be resolved from the original manifests, not rounded feature-table eta.

Keep h, C, architecture and pure NLL fixed. Separate more informative labels from extra training weight: a high-certainty label arm should preserve each pair's prior effective weight, with an old-label reweighting control if count weighting is also changed. Use source-family cross-validation and keep target-controller/LOSO data unopened.

- If the contrast becomes reproducible and held-family prediction improves, weak decision-specific supervision is supported.
- If reliable contrasts exist but cannot be fit on TRAIN, investigate representation/optimization; if TRAIN fits and held families fail, prioritize support and conditional generalization.
- If state contrasts remain unresolvable or a common candidate remains sufficient, this panel cannot strongly test state-adaptive robust selection; build independent evidence with a source-only, outcome-blind confirmation rule rather than selecting model-winning TEST cases.

No generator changes or model promotion were made.

## Reproduce

```bash
cd /home/zhihan/research/Basin_C1
JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /home/zhihan/research/02_C1_Toy_GiveWay/.venv-c1/bin/python -m diagnostics.orthoflow3_state_learning_bottleneck_audit_v1.context_value_followup
```

The script explicitly limits CPU inference to two cores. CUDA plugin discovery can emit a warning on the login environment; the script asserts that actual inference uses CPU and verifies frozen logits.
