# Success-basin oracle label stability

## Scope

This audit uses existing trajectories only. No rollout, training, controller change, or new objective was introduced. Feasibility is the requested provisional rule `success_count >= 63/64`. Cells without a valid 64-run cohort are **not evaluated**, not labeled infeasible.

When the canonical 64 seeds (`95106001..32` plus `95107001..32`) exist, they are used. An eta with exactly 64 other existing valid seeds is also audited, with its cohort disclosed; paired near-optimal tests then use only the seed intersection.

## 1. B_63(z)

- D1: (0.5703125, -0.375, -0.125), (0.578125, -0.40625, -0.125), (0.578125, -0.34375, -0.125)
- D2: EMPTY
- D4: EMPTY

D2 and D4 have no currently evaluated eta meeting 63/64. This is an empty empirical feasible set under the provisional rule, not evidence that no feasible controller exists.

## 2–3. Minimum deformation and E_near

| state | strict empirical best eta | mean J_def over matched 64 | success | E_near size |
|---|---|---:|---:|---:|
| D1 | (0.5703125, -0.375, -0.125) | 0.230463 | 63/64 | 2 |
| D2 | EMPTY | — | — | 0 |
| D4 | EMPTY | — | — | 0 |

Percentage bands (1%, 2.5%, 5%) and every paired confidence interval are in `near_optimal_eta.csv` and `label_statistics.json`; no percentage threshold is frozen.

## 4. First-step executed-label stability

The label is always `g_exec = u_exec - u_safe` after the second hard projection, flattened from a 2×2 joint action to four components.

| state | result | E_near | mean same-seed eta distance | max distance | mean / vmax | mean / typical label norm | eta variation / Flow-seed variation | executed/raw pair distance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D1 | LABEL_STABLE | 2 | 0.000987704 | 0.00111996 | 0.0020 | 0.0232 | 0.4643 | 0.1759 |
| D2 | NOT_TESTABLE_EMPTY_B63 | 0 | — | — | — | — | — | — |
| D4 | NOT_TESTABLE_EMPTY_B63 | 0 | — | — | — | — | — | — |

Projection-collapse fractions, cosine similarities, componentwise variances, and raw/executed corrections are machine-readable in `label_statistics.json`, `first_step_labels.csv`, and `representative_cases.json`.

## 5. Deterministic supervision decision

- D1: deterministic first-step supervision is justified at the resolution of this audit. The numerical classification is based on disagreement relative to `vmax=0.5`, typical executed-correction norm, actual Flow-seed variability, and projection compression—not eta-space connectedness.
- D2/D4: deterministic supervision is **not currently justified under the 63/64 oracle rule**, because `B_63` is empty. This is lack of an eligible target, not evidence of multivalued action labels.

## 6. Remaining ambiguity

For D1, any remaining ambiguity is action-space variation among statistically equivalent feasible eta after hard projection and is quantified above. For D2/D4, the blocker precedes identifiability: no current eta passes 63/64, so `E_near` and its executed-label distribution do not exist in the available data. Forcing either state into `LABEL_STABLE`, `LABEL_MILDLY_AMBIGUOUS`, or `LABEL_MULTIVALUED` would conflate empty feasibility with label geometry.
