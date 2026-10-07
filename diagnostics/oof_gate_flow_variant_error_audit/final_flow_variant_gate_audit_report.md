# Final OOF gate Flow-variant error audit

## Stage A — missing fold

`LOGO_anchor_D1_pair231` passed every leakage/integrity check. Train/validation/test contained 199/47/31 states and 12736/3008/1984 saved samples. All three pre-registered seeds trained successfully and their checkpoints were saved.

- seed 17: 64/64 correct; p=0.883493±0.003289, range [0.879746, 0.894697].
- seed 23: 60/64 correct; p=0.856882±0.002986, range [0.844361, 0.859535].
- seed 41: 64/64 correct; p=0.782395±0.002811, range [0.775207, 0.786425].
- seed SEED_MEAN: 64/64 correct; p=0.840924±0.001319, range [0.837940, 0.842961].

## Stage B — exact reproduction

The three existing required folds reproduced their saved per-seed and seed-mean OOF state probabilities/thresholds within tolerance 2e-06. No test threshold was retuned.

## Primary state persistence

- `RBV_Q_pair228_m080_s95401003_p030`: 0/64 correct, p=0.902928±0.006064, range [0.889123, 0.921269]; **SYSTEMATIC_STATE_LEVEL_ERROR**.
- `R_D4_s95101006_p60`: 64/64 correct, p=0.745113±0.002254, range [0.737884, 0.752026]; **MOSTLY_CORRECT_WITH_ISOLATED_ERRORS**.
- `RB_Q_pair226_m080_s95400802_p073`: 0/64 correct, p=0.750910±0.003421, range [0.743333, 0.756912]; **SYSTEMATIC_STATE_LEVEL_ERROR**.
- `R_D1_s95106004_p40`: 64/64 correct, p=0.840924±0.001319, range [0.837940, 0.842961]; **MOSTLY_CORRECT_WITH_ISOLATED_ERRORS**.

## Pair ranking and thresholds

- `collision_pair_1`: ranking/AUC=0.000000, probability margin(nonzero-zero)=-0.157815, logit margin=-1.159419, threshold accuracy=50.00%; **RANKING_WRONG**.
- `collision_pair_2`: ranking/AUC=1.000000, probability margin(nonzero-zero)=0.090013, logit margin=0.561593, threshold accuracy=50.00%; **RANKING_CORRECT_THRESHOLD_WRONG**.

## Control and decision

The fixed control pair has ranking/AUC=0.999268, probability margin=0.043194, and threshold accuracy=50.00%.

Descriptive source/local-shortcut evidence on the erroneous states: supported (2/2 systematically wrong states are closer in mean score to opposite-label than true-label training states).

Final classification: **MIXED_OOF_FAILURE**.

Smallest justified next experiment: perform one inference-only, frozen-checkpoint attribution-and-margin audit on these same variants, comparing static feature-group logit contributions for the two systematic zero-state errors against their validation score/threshold distributions and fixed training neighbors. This addresses the pair-1 ranking reversal and pair-2 calibration shift without retraining or redesign.
