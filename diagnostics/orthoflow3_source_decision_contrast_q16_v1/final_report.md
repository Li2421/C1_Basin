# Source decision-contrast Q16 audit

The controller context is causally relevant to rollout outcomes, but this
experiment did **not** establish that the current critic learns its effect on
eta choice. This is a source-only, family-disjoint diagnostic, not a new LOSO
result or an independent held-controller confirmation.

## Frozen protocol and data integrity

- Source TRAIN only: 46 independent source families, the previously frozen
  third Flow controller, two exact eta probes per family. No held-controller,
  source VAL or LOSO target labels entered fitting, normalization or checkpoint
  selection. The generator and critic architecture remained unchanged.
- Cache preflight: 1,472 requested continuation keys; 368 prior compatible
  outcomes reused; 1,104 truly missing outcomes run. Six shard journals added
  1,104 records to the global rollout DB, including six distinct numerical
  failures. There were zero merge conflicts. All 1,472 requested keys are now
  in the DB; 1,466 have valid outcomes and six remain numerical rather than
  fabricated Q16 successes/failures. The postflight cache therefore reports
  six *non-exact-reusable* numerical keys, not six absent DB records.
- Ninety-two state–eta pairs were upgraded: 87 have full valid Q16; five
  retain their actual partial counts. The first four seed outcomes match the
  frozen original Q4 labels exactly.
- The outer evaluation is a deterministic three-fold **source-family** split.
  Each family is held out once. A disjoint six-family inner validation split
  chooses checkpoints. Three initialization/order seeds were used per arm.

## Was the original decision target noisy or weak?

Only 16/46 families distinguish a B15 candidate from a known non-B15
candidate among these two probes. Only 9/46 have an empirical Q contrast of
at least 0.25. Across families, the original Q4 and upgraded Q16 eta contrasts
correlate at 0.518. Independent first-eight versus last-eight seed contrasts
correlate at 0.337; among the nine larger full-Q16 contrasts, seven retain the
same sign between halves. Both limited decision headroom and finite-seed noise
are present. These statistics concern this frozen two-probe comparison, not
the entire three-dimensional Success Basin.

## Matched pure-NLL models

All three arms share the same physical inputs, architecture, optimizer,
minibatch draws, number of steps, fold split and checkpoint criterion. Only
the 92 source TRAIN labels differ. `Q16-rate / weight 4` changes the observed
rate while preserving each pair's four-trial effective weight. `Q4 / weight
16` retains the original rate while increasing its effective weight.

| Training arm | Outer Q NLL | Outer Q MAE | Predicted vs actual eta-Q contrast Pearson | B15 selected on 16 eligible families, seeds 17/23/41 |
|---|---:|---:|---:|---:|
| Original Q4 | 0.4260 | 0.1381 | −0.098 | 7 / 7 / 7 |
| Q16 rate, weight 4 | 0.4263 | 0.1400 | −0.116 | 8 / 10 / 8 |
| Original Q4, weight 16 | 0.4405 | 0.1442 | −0.135 | 7 / 7 / 6 |
| Fit-fold eta-only preference | — | — | cannot model state contrast | 9 / 16 |

The Q16-rate arm rescues nine and breaks four eligible seed×family decisions
relative to the original Q4 arm (pooled across three correlated training
seeds). The gain is small and not robust enough to establish state-conditioned
selection; the arm does not beat the fold-fit eta-only choice, and neither NLL
nor Q-contrast prediction improves. Simply increasing old Q4 weight is worse,
so the modest B15 change is not explained by extra label weight.

## Interpretation and next discriminating test

This result weakens the hypothesis that **Q4 label precision alone** was the
main reason the critic ignored h and controller context. It does not prove that
the physical inputs lack the required information, nor that the shared
feasibility law is absent. The present two-probe panel offers too few reliable
state-dependent decisions to distinguish model-fit failure from weak
decision supervision.

The smallest useful next test is a source-only, source-family-disjoint panel
of naturally stronger eta contrasts. The panel must be selected from TRAIN
evidence, confirmed with compatible Q16, and evaluated on untouched source
families with eta-only, additive, and full interaction models. Only a model
that wins this source test merits another strict LOSO confirmation. Do not use
LOSO target labels to choose probes or alter the frozen generator.

Reproduction: `pipeline.py` (preflight, journals, merge/postflight),
`labels.py` (matched labels), `train_contrast.py` (27 source-family crossfit
models), and `summarize.py` (all tables). Structured results are in
`final_decision.json`, `crossfit_summary.json`, and the adjacent CSV files.
