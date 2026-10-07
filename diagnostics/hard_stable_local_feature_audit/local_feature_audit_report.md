# Hard stable local feature audit

## Decision

**MIXED_LOCAL_FAILURE_MODES**

The exact six pre-defined robust mistakes were reproduced from the three-seed source-group-held-out predictions. No model was retrained, no state or rollout was generated, and the 214-D representation and oracle were unchanged.

## State-level evidence

| State | Oracle | cross-group d_same | cross-group d_opp | cross ratio | own-group ratio | k=5 true fraction | support groups | Classification |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `RBV_Q_pair228_m080_s95401003_p030` | 0 | 0.649 | 0.505 | 0.78 | 0.63 | 0.00 | 0 | REPRESENTATION_COLLISION |
| `RB_P_r175_m080_s95400305_p132` | 1 | 0.026 | 0.198 | 7.69 | 2.07 | 0.80 | 2 | INFORMATION_PRESENT_BUT_NOT_LEARNED |
| `RB_P_r175_m080_s95400308_p144` | 1 | 0.115 | 0.320 | 2.78 | 1.77 | 0.20 | 1 | INFORMATION_PRESENT_BUT_UNDERCOVERED |
| `RB_Q_pair225_m080_s95400707_p166` | 1 | 0.114 | 0.280 | 2.47 | 1.11 | 0.20 | 1 | INFORMATION_PRESENT_BUT_UNDERCOVERED |
| `RB_Q_pair226_m080_s95400802_p073` | 0 | 0.686 | 0.253 | 0.37 | 0.86 | 0.00 | 0 | REPRESENTATION_COLLISION |
| `RB_Q_pair228_m080_s95401001_p050` | 0 | 0.663 | 0.609 | 0.92 | 2.70 | 0.00 | 0 | INFORMATION_PRESENT_BUT_UNDERCOVERED |

Distances use each state's original cross-validation fold normalization and only that fold's training source groups. The held-out source group is excluded from every primary neighbor, kNN, support, and probe calculation.

## Existing information and full snapshot

- Diagnostic group/top-eight logistic probes, selected and fitted only on training groups: episode_time 3/6, history_monitor 3/6, TOP8_UNIVARIATE_TRAIN_ONLY 2/6, projection_safety_diagnostics 2/6, current_control 1/6, current_geometry_observation 1/6.
- Full augmented snapshots contain 14 stored fields. Every varying physical, monitor, timer, latch, and ordered error-history field is already encoded in the 214-D input.
- The only omitted stored fields are four first-terminal-event latches plus `done`; none differs in any robust-mistake/nearest-opposite comparison, so they provide no missing decision information.
- State classifications: {'REPRESENTATION_COLLISION': 2, 'INFORMATION_PRESENT_BUT_NOT_LEARNED': 1, 'INFORMATION_PRESENT_BUT_UNDERCOVERED': 3}.

## Frozen gate sensitivity

Sensitivity is the input gradient of the frozen confidence-aware gate logit with respect to its normalized 214-D input, averaged over the state's 64 Flow variants. It is diagnostic only. Per-state comparisons between the most sensitive and most locally separating groups are recorded in `gate_sensitivity.csv` and `per_state_failure_classification.csv`.

The frozen gate classifies 6/6 of these states correctly at its validation-selected threshold 0.847717, whereas the strict source-group-held-out gates fail under all three seeds. Because the frozen checkpoint's original train/validation split contains some of these groups, its gradients diagnose learned attention but are not independent OOF attribution.

## Smallest next experiment

Using only the saved 64 Flow variants, compare sample-distribution overlap for the representation-collision states against their nearest opposite-label neighbors; do not retrain or collect data.

No controller redesign is proposed by this audit.
