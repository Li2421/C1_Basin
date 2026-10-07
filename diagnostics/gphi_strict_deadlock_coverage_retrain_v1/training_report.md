# Strict-deadlock coverage retraining

The model was trained from scratch on the immutable startup-complete dataset
plus exactly 11 historical earliest-robust strict-deadlock states (64 Flow
variants per state).  No state was duplicated or specially weighted.  The six
fresh-unseen strict-deadlock states were absent from training, normalization,
validation, and checkpoint selection.

## Validation-only checkpoint

- Seed: 41
- Epoch: 1177
- SHA256: `340b81d5c4ad2cea7bee16931fa00d095708f5aa6ca6f255e0a7fc5a35873700`
- Startup validation mean L2: 0.07338723
- Warm validation mean L2: 0.03972464
- Aggregate validation mean L2: 0.04829330

## Frozen matched test retention

| Cohort | Old mean L2 | New mean L2 |
|---|---:|---:|
| Startup | 0.08885766 | 0.08009410 |
| Warm V3 | 0.01207518 | 0.01585419 |
| All | 0.02887135 | 0.02990667 |

Closed-loop outcomes are deliberately not used anywhere in this selection.
