# Frozen asset audit (2026-10-02)

- The original joint mode-free generator/critic was trained on the v1 Double-Bottleneck, Four-Way and Ring dataset. Its Ring training labels predate the corrected safety semantics, so its Ring predictions are a frozen historical baseline, not a corrected-safety trained model.
- `datasets/orthoflow3_basin_dataset_v2_audited` contains corrected Ring labels (`ring_current_safety_v2`) on a dense repeated-state×eta matrix. Four-Way is also dense, whereas DB has a much smaller repeated-eta panel. The audited train/validation split reports no source-parent, trajectory-family or exact-conditioning cross-split overlap.
- The Ring K16 diagnostic reuses 60 source-diverse states under corrected safety: oracle 60/60, frozen critic 53/60. It has been examined already and is therefore diagnostic only, not a new untouched confirmation set.
- The separate `orthoflow3_k16_critic_canonicalization_v1` task is actively collecting audited K16 Q16 labels and independent fresh confirmation states. This task will not duplicate or overwrite that work. Planned work is not counted as completed evidence.
- The Toy 200-state K16 original 181/200 versus oracle 194/200 remains a frozen historical result. The 30-pair local-repair result also admits a 10/10 eta-only explanation and is not evidence of state-conditioned generalization.

## Immediate discriminating test

Freeze an eta-only continuous predictor using audited TRAIN Ring labels and validation-only hyperparameter selection. On the existing frozen Ring K16 proposals, compare it with the original critic on identical candidates. Also select eta pairs using TRAIN only and test whether their preference reverses across validation source families. This separates global eta preference from evidence that state matters without launching a duplicate rollout.
