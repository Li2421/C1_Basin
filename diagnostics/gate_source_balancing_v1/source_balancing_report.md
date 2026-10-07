# Source-group-balanced gate training

Controlled sampler comparison only: no state, rollout, oracle, feature, architecture, correction-head, or closed-loop change.

## Integrity and sampler weights

The sample-uniform seven-fold baseline was exactly reproduced. Every stable state has 64 Flow variants, so state-balanced-only has exactly the same expected state and source weights as sample-uniform; it only changes epoch resampling noise. Sample-uniform source-group mass spans 0.0046–0.1505, whereas source-group-balanced is approximately 0.0088–0.0094 per train group.

Normalization is train-only; thresholds are validation-only; held-out source groups are excluded from training, normalization, validation, and threshold selection. Raw scores from separate LOGO models are not pooled for cross-fold AUROC/AUPRC.

## Seed-mean OOF classification: BAcc / FPR / FNR

| condition | all evaluated stable (N=130) | recovery (N=95) | pre-registered difficult stable (N=13) |
|---|---:|---:|---:|
| sample-uniform | 0.8588 / 0.1200 / 0.1625 | 0.8067 / 0.1200 / 0.2667 | 0.3929 / 0.5000 / 0.7143 |
| state-balanced only | 0.8588 / 0.1200 / 0.1625 | 0.8067 / 0.1200 / 0.2667 | 0.3929 / 0.5000 / 0.7143 |
| source-group balanced | 0.9025 / 0.1200 / 0.0750 | 0.8844 / 0.1200 / 0.1111 | 0.3929 / 0.5000 / 0.7143 |

The three fixed robust stable-zero false-positive clouds remain 64/64 intervention under source balancing, and their positive fold-relative margins increase. Group-margin spread improves (zero 1.411→0.339; nonzero 0.571→0.494), but integrated-gradient contributions from shortcut-prone groups do not disappear.

## Conclusion

**SOURCE_BALANCING_HELPS_BUT_INSUFFICIENT.** It improves broad stable/recovery FNR and group-margin stability, but does not repair the primary hard-boundary false positives. The smallest justified next training experiment is a source-group-robust objective (such as group-DRO) with a fresh source-group OOF evaluation; source-frequency equalization alone is insufficient.
