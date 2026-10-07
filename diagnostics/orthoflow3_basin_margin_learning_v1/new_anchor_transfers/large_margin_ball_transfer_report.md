# OrthoFlow3 large-margin ball transfer pilot v1

## Decision

**LARGE_MARGIN_BALL_TRANSFER_STRONGLY_SUPPORTED**

- Anchor radii: N_r002_m120=0.353621, N_r049_s159=0.249688, N_r126_s461=0.353621, N_r132_s238=0.265625.
- Margin-eligible anchors: 4/6; exact real neighbors: 16.
- Center transfer B63: 16/16.
- Training-usable transferred balls: 16/16.
- Accepted shrink factors: {'1.0': 16}; failures: 0.
- Mandatory inside B63: 64/64; confirmed false inclusions: 0.
- Transferred radius mean/median/min/max: 0.3056388125467918 / 0.3096231875467918 / 0.2496875 / 0.3536213750935836.
- Retained radius mean/median/min: 1.0 / 1.0 / 1.0; retained-volume mean: 1.0.
- Outside shell 8/8: 0/96.
- Usable new labels per eligible expensive anchor: 4.000; label multiplier: 5.000.

## Dataset interpretation

Unique expensive anchor source groups remain 4; transferred neighbors increase local coverage, not source diversity. The empirical shrink factor was 1.00 for every accepted neighbor, so no distance/shrink correlation is estimable. Estimated source-separated 24/8/8 construction is recorded in `dataset_expansion_estimate.json`.

No G, H, Q, J, center regression, or margin learner was trained.
