# Mode-free generator and finite-proposal critic on the original hard cohort

The original frozen Toy WIDE-IC cohort has 200 states. We evaluated 16 matched continuations per state and candidate, with B15 defined as at least 15/16 successes. The mode-free generator and all four proposals, critic rankings, controller settings, and cohort were frozen before these outcomes. The new generator uses no manual mode ID or Toy–DB correspondence. The critic only ranked four generated proposals; it never optimized eta in continuous space.

| Controller | B15 states / 200 | Mean Q16 | Success / 3200 | Deadlock | Timeout | Collision | Mean J_def |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAC-only | 119 | 0.7316 | 2341 | 0 | 152 | 707 | — |
| Hard Safety | 114 | 0.7216 | 2309 | 133 | 758 | 0 | 0 |
| Fixed common eta | 180 | 0.9578 | 3065 | 113 | 22 | 0 | 0.5117 |
| Old selector anchor (diagnostic) | 199 | 0.9969 | 3190 | 9 | 1 | 0 | 0.4964 |
| Old mode-conditioned generator mean (diagnostic) | 197 | 0.9919 | 3174 | 19 | 7 | 0 | 0.5015 |
| New mode-free generator mean | 142 | 0.7947 | 2543 | 357 | 300 | 0 | 0.5036 |
| One frozen random sample | 131 | 0.7219 | 2310 | 517 | 373 | 0 | 0.4842 |
| Critic-selected among four | 167 | 0.8997 | 2879 | 171 | 150 | 0 | 0.5508 |
| Oracle best among four | 173 | 0.9225 | 2952 | 116 | 132 | 0 | 0.5578 |

Across all four random draws per state, a single draw was B15 in 533/800 state–draw cases (66.6%), below the mean's 142/200 (71%). Yet the best of four was B15 on 173/200; this is a genuine proposal-diversity opportunity, not evidence that an unselected sample is safer.

Paired state-level bootstrap: generator mean minus Safety was +14.0 B15 percentage points (95% CI +4.5 to +23.0), with 61 rescued and 33 broken states. Against fixed common eta, the same mean was −19.0 points (CI −26.0 to −12.0), with 11 rescues and 49 breaks. Critic selection improved on the generator mean by +12.5 points (CI +7.5 to +18.0): 28 rescues and 3 breaks. The critic remained 6.5 points below fixed common eta (CI −13.0 to 0.0). Oracle best-of-four was only 3.0 points above the critic (CI +1.0 to +5.5), and the critic selected B15 on 167/173 oracle-coverable states. The requested critic recovery fraction versus the mean is 28/173 = 16.2%; versus the first random sample it is 39/173 = 22.5%.

The hard cohort does expose baseline failures: Safety is non-B15 on 86 states. The generator mean rescues 61 of those, while critic selection rescues 72. But fixed common eta is already B15 on 180/200 and the old selector anchor on 199/200, so the new method does not improve the strongest existing correction. Even the four-sample oracle reaches only 173/200. On 20 states where fixed common eta fails, the critic rescues 15, but it breaks 28 states where fixed common eta succeeds; a hypothetical hybrid policy was not tested.

The simplest mode-free diagonal Gaussian is therefore not adequate as a replacement. Its predicted scale nearly saturates the configured maximum of 0.30 in every coordinate (median approximately 0.30), consistent with a broad unimodal density trying to cover separated successful eta regions. The mean's average Euclidean eta distance from the fixed common point is 0.616. These are descriptive diagnostics, not proof that distance itself causes failure. This experiment falsifies this particular unimodal architecture and training rule on unseen hard states; it does not prove that manual canonical mode IDs are necessary or that every mode-free multimodal generator would fail.

Verdicts: **NO_GAIN_OVER_FIXED_BASELINE** for the generator mean; **STOCHASTIC_PROPOSALS_ADD_VALUE_WITH_SELECTION_ONLY** for four draws; **CRITIC_SELECTION_ADDS_VALUE_WITHIN_GENERATOR_PROPOSALS**. The critic is near the proposal oracle, so proposal coverage is now the larger bottleneck. A learned mode-free mixture or flow is a justified next capacity test, but multimodality is not yet proven necessary. The default deployed method should not be replaced by this generator/critic combination.

All 31,152 unique requested continuations are in the global rollout database and the final postflight reports 31,152 EXACT_REUSE, zero missing, zero ambiguous and zero incompatible. The first 336 GPU/CPU exact-float-check false alarms were quarantined and never used; valid rows were recovered, and the final matrix contains only scientifically valid outcomes. This report concerns Toy WIDE-IC only; no DB hard-cohort transfer claim is made.
