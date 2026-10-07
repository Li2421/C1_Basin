# Confidence-aware gate retraining

## Decision

**STABLE_BOUNDARY_STILL_NOT_LEARNABLE**  
**NOT_READY_FOR_GATE_PLUS_CORRECTION_PILOT**

Hard BCE used only statistically stable original B_63 labels. Ambiguous states were excluded from normalization, training, validation selection, and primary scoring; they were retained for probability diagnostics.

## Confidence dataset

- States: 324 total = 157 stable zero / 120 stable nonzero / 47 ambiguous.
- Category counts by confidence class: {'ORACLE_STABLE_ZERO': {'NORMAL': 90, 'RECOVERY': 67}, 'ORACLE_AMBIGUOUS': {'NORMAL': 15, 'RECOVERY': 32}, 'ORACLE_STABLE_NONZERO': {'NORMAL': 15, 'PRE_DEADLOCK': 60, 'RECOVERY': 45}}.
- Existing enlarged audit reproduced exactly for all 42 prior states. New continuation rollouts: 73476.
- Original Dataset V4 source-group split was preserved with no cross-split group leakage.

## Selected gate

- Architecture/seed: [214, 64, 64, 1] / 41.
- Validation-stable-selected threshold: 0.847717.
- Stable test balanced accuracy / AUROC / AUPRC: 0.9722 / 0.9954 / 0.9966.
- Stable test FPR / FNR / Brier: 0.0556 / 0.0000 / 0.0322.
- Difficult-stable test balanced accuracy / AUROC / FPR / FNR: 0.7500 / 0.5000 / 0.5000 / 0.0000 (4 states).

## Matched previous-gate comparison

- Previous -> new stable-test balanced accuracy: 0.9722 -> 0.9722; AUROC: 1.0000 -> 0.9954; Brier: 0.0192 -> 0.0322.
- Previous -> new difficult-stable balanced accuracy: 0.7500 -> 0.7500; AUROC: 1.0000 -> 0.5000.

## Ambiguous states and scaling

- Ambiguous p_gate median / 5-95%: 0.5547 / [0.0001, 0.9965].
- Mean gate confidence (ambiguous vs stable): 0.8181 vs 0.9436.
- Ambiguous p_gate vs Q0 Pearson/Spearman: -0.3853 / -0.4576; the sensible direction is negative.
- Stable training scaling rows are recorded in `scaling_analysis.csv`; difficult-subset improvement from 50% to 100%: False.

No correction head was trained and no learned closed-loop control was run.
