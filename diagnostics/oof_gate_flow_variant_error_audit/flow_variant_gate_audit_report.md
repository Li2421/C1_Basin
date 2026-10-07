# OOF gate Flow-variant error audit — preflight stop

## Outcome

The requested inference audit was **not launched** because one of the four primary states has no valid pre-existing OOF fold.

`R_D1_s95106004_p40` belongs to source group `anchor_D1_pair231`. The existing hard-stable cross-validation contains outer folds only for:

- `anchor_D2_pair228`
- `anchor_D4_pair227`
- `baseline_r175`
- `baseline_r198`
- `qual_pair225`
- `qual_pair226`

Consequently, no saved/reproducible existing gate exists for which `anchor_D1_pair231` was absent from training, normalization fitting, validation threshold selection, and model selection. In `LOGO_00_anchor_D2_pair228` this group is an inner-validation group, so using that gate would explicitly violate the requested OOF condition. In the other relevant folds it is training data.

No fold checkpoints were saved. Three primary states have a recorded exact outer fold and could be reproduced, but pair 2 cannot receive the requested valid pair-level ranking/AUC without creating a new seventh outer fold. Creating that fold would be a new evaluation, not reproduction of an existing OOF gate.

Per the task's explicit instruction, “If exact reproduction fails, STOP and report why,” no training or per-variant inference was performed. No final failure-mode classification is scientifically valid from an incomplete pair audit.

## Smallest justified next experiment

Pre-register and run one additional `LOGO_anchor_D1_pair231` fold using the unchanged deterministic split/training recipe and seeds 17/23/41, then rerun this audit. This supplies the single missing OOF gate without changing architecture, features, oracle, states, or rollouts.
