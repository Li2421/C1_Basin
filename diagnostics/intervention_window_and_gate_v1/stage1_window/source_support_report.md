# Window-target source-support audit

This is a zero-cost, artifact-only audit of the frozen Stage-1 candidate
horizons. No rollout, model training, label modification, feature change, or
controller/oracle execution was performed.

## Inputs and label reconstruction

- Source artifact: `paired_success_differences.csv`.
- Population: 277 oracle-stable states from 117 source groups.
- Positive (`y_H=1`): `SUPPORTED_RECOVERABILITY_LOSS`.
- Zero (`y_H=0`): `RESOLVED_NO_LOSS_EXACT` or
  `SUPPORTED_RECOVERABILITY_IMPROVEMENT`.
- Excluded from resolved-class support: `H_AMBIGUOUS`.

The reconstructed zero/positive/ambiguous counts exactly match the frozen
candidate inventory for H=4, H=8, and H=16.

## Compact source-support results

| H | Positive states | Positive source groups | Groups with both classes | Top-3 source share | Positive-group fraction | Classification |
|---:|---:|---:|---:|---:|---:|---|
| 4 | 28 | 11 | 11 | 60.71% | 9.40% | `HIGHLY_SOURCE_CONCENTRATED` |
| 8 | 24 | 10 | 10 | 62.50% | 8.55% | `HIGHLY_SOURCE_CONCENTRATED` |
| 16 | 35 | 11 | 11 | 54.29% | 9.40% | `HIGHLY_SOURCE_CONCENTRATED` |

The classification is descriptive: at every horizon, only 10–11 of 117
groups contain any positive state, while the top three positive-containing
groups contribute 54–63% of all positives. This joint pattern indicates
strong source concentration rather than broad independent-group support.

## Positive states per positive-containing source group

| H | Median | Mean | Max | Top-1 share | Top-3 share | Top-5 share |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.0 | 2.545 | 7 | 25.00% | 60.71% | 78.57% |
| 8 | 1.5 | 2.400 | 7 | 29.17% | 62.50% | 79.17% |
| 16 | 3.0 | 3.182 | 8 | 22.86% | 54.29% | 74.29% |

Exact distributions:

- **H=4:** `anchor_D4_pair227:7`, `anchor_D1_pair231:6`,
  `anchor_D2_pair228:4`, `baseline_r096:3`, `qual_pair225:2`, and six
  groups with one positive each.
- **H=8:** `anchor_D1_pair231:7`, `anchor_D4_pair227:5`,
  `qual_pair225:3`, `anchor_D2_pair228:2`, `baseline_r096:2`, and five
  groups with one positive each.
- **H=16:** `anchor_D1_pair231:8`, `anchor_D4_pair227:7`,
  `anchor_D2_pair228:4`, `qual_pair225:4`, `baseline_r096:3`,
  `baseline_r182:3`, `baseline_r198:2`, and four groups with one positive
  each.

The complete group/count strings are preserved in `source_support_audit.csv`.

## Zero support and strict-LOGO implications

| H | Groups with zero | Groups with both classes | Zero-only/single-class groups | Test folds with a positive | Test folds without a positive |
|---:|---:|---:|---:|---:|---:|
| 4 | 117 | 11 | 106 | 11 | 106 |
| 8 | 117 | 10 | 107 | 10 | 107 |
| 16 | 117 | 11 | 106 | 11 | 106 |

All 117 groups contain at least one resolved zero. Consequently every
positive-containing group contains both classes, and all remaining groups are
resolved-zero-only.

Strict LOGO training retains positive support in every fold: holding out the
largest positive group still leaves 21, 17, and 27 positive training states at
H=4, H=8, and H=16 respectively. Therefore **no training fold lacks positive
examples**. However, 106/117, 107/117, and 106/117 held-out test groups contain
no positive example. Per-fold balanced accuracy or AUROC is therefore undefined
for most individual LOGO folds; any later evaluation must pool leakage-free
OOF predictions and separately disclose positive-containing held-out groups.

## Interpretation

The frozen targets have enough multi-group positive support to make strict
LOGO training mechanically possible, and no single source group owns all
positives. They do **not** have broad source coverage: positives occupy only
8.5–9.4% of source groups and are dominated by a small leading subset.

This audit does not establish or refute learnability. It establishes that a
future Stage-2 target-validity experiment will have highly concentrated
positive test support and must interpret cross-source results accordingly.
