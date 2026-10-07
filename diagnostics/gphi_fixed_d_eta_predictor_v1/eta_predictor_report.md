# Fixed-D structured eta predictor audit

## Result

**STRUCTURED_ETA_STRONGLY_SUPPORTED**

The eta predictor was trained on exactly the same 424 states / 27,136 Flow-variant samples and the same state-grouped split as the coverage direct-g model.  It predicts one three-dimensional eta at the queried state, clips only to the frozen full domain, and then holds eta fixed while the known basis feedback is recomputed every physical step.  There was no DAgger data, gate, online eta search, or per-step eta re-prediction.

## Frozen model

- Architecture: 214 -> 128 -> 128 -> 3, SiLU (44,419 parameters; direct-g has 44,548)
- Seed / epoch: 17 / 777
- Checkpoint: `/home/zhihan/research/Basin_C1/diagnostics/gphi_fixed_d_eta_predictor_v1/best_fixed_d_eta_checkpoint.npz`
- SHA256: `2481028b7e2c4ef14fb14016c138e0d00dd83e40cc2997fd1ad1ee9b5028b095`
- Validation startup/warm normalized eta L2: 0.104640 / 0.126386
- Test all normalized/physical eta L2: 0.087594 / 0.094295
- Fresh-6 physical eta L2: 0.331817
- Test clipping fraction: 17.2363%

## Strict-deadlock robust64

| Controller | Success/1088 | Q | Deadlock | Timeout | Collision | B63/17 | Mean J_def |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle fixed eta | 1088/1088 | 1.0000 | 0 | 0 | 0 | 17 | 0.40625 |
| Direct-g coverage | 841/1088 | 0.7730 | 0 | 247 | 0 | 0 | 0.13192 |
| Direct-g k1 | 298/1088 | 0.2739 | 0 | 790 | 0 | 0 | 0.28291 |
| Structured fixed-D eta | 1018/1088 | 0.9357 | 69 | 1 | 0 | 15 | 0.36772 |

Exact original-Flow success was 11/11 historical, 5/6 fresh diagnostic, and 16/17 combined.  Robust structured-eta success split was 704/704 historical and 314/384 fresh.

Flow-variant stability: across all 424 states, mean/max within-state maximum pairwise eta distance was 0.009768/0.156530; on the aligned strict-deadlock 17 it was 0.013254/0.052361.  See `temporal_structural_error.csv` for the same-visited-state structured-vs-oracle correction comparison at k=0,1,2,4,8,16,32, and `teacher_takeover_structured_eta.csv` for optional k=8,16,32 takeover recoverability.

Hard safety intact: **True**.  Collisions=0, invalid/solver failures=0.

Residual limitation: fresh_r133 remains outside a robust predicted-eta basin (2/64), while fresh_r058 is usable but not B63 (56/64).  This is localized fresh-cohort eta prediction/generalization error, not a failure of the fixed-D structured representation.

Basis-as-set decision: The fixed-D result supports a later variable-D Basis-as-set experiment, but only after one fresh full-WIDE confirmation.

Smallest next experiment: Freeze this checkpoint and evaluate one fixed, untouched 200-episode WIDE cohort against Safety and direct-g using the one-shot persistent eta deployment.
