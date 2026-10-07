# OrthoFlow3 CENTER vs MARGIN learning v1

## Decisions

- Point target: **LOWJ_POINT_TARGET_REMAINS_COMPETITIVE**.
- Basin supervision: **BASIN_MARGIN_SUPERVISION_HARMS**.

Verified set-valued supervision did **not** add value beyond robust-center regression. On the held-out intermediate states both were robust despite failing to reproduce the verified-ball geometry; on the mandatory fresh t=0 extrapolation, MARGIN caused materially more Safety-success breaks than CENTER.

## Frozen dataset and geometry

- Dataset unchanged: TRAIN/VAL/TEST = 25/10/9; independent source families = 5/2/2; leakage = 0.
- New labels/anchors or center/radius modifications: 0/0/none.
- True t=0 labels: 0/44. All supervised examples are intermediate states.
- All 25 TRAIN retained 0.8-radius balls share the exact physical eta `(0.625, 0, 0.375)`; common-intersection fraction = 25/25 and minimum retained radius = 0.19975. A state-independent zero-loss MARGIN solution therefore exists.

## Training and VAL selection

Both arms used the same 214→128→128→3 SiLU model (44,419 parameters), unconstrained normalized-eta output, AdamW 1e-3/1e-5, and seeds 17/23/41.

| arm | seed | best epoch | VAL loss | TRAIN rho | TRAIN retained | VAL rho | VAL retained | VAL true success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CENTER | 17 | 1017 | 0.920464 | 0.054 | 1.00 | 2.760 | 0.50 | 160/160 |
| CENTER | 23 | 7 | 0.542762 | 0.516 | 0.92 | 2.294 | 0.30 | 80/160 |
| CENTER | 41 | 2 | 0.598870 | 0.749 | 0.52 | 2.372 | 0.40 | 80/160 |
| MARGIN | 17 | 99 | 14.630221 | 0.617 | 1.00 | 3.272 | 0.50 | 160/160 |
| MARGIN | 23 | 10 | 5.202090 | 0.651 | 0.44 | 2.419 | 0.20 | 80/160 |
| MARGIN | 41 | 2 | 4.176321 | 0.763 | 0.64 | 2.128 | 0.50 | 80/160 |

Selected CENTER: seed 17, SHA256 `8a79fc4e198cba76562b90197a8a6f52fff2d20c2f536b000301d482ec5c2779`. Selected MARGIN: seed 17, SHA256 `ba02aeddccb5533b1fea7e3a5da8039cb9226cc2196168d1e40a897c177146f8`. Both achieved VAL 160/160 and 10/10 states at 16/16. MARGIN's selected checkpoint had TRAIN zero-loss fraction 1.00, output mean pairwise distance 0.189; CENTER's was 0.026. Thus MARGIN did not fully collapse numerically, despite an exact universal solution existing.

## Held-out intermediate TEST

| model | inside ball | retained | mean/median rho | B63 states | mean Q64 | successes | deadlock/timeout/collision | mean successful J_def |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CENTER | 0/9 | 0/9 | 2.396/2.947 | 9/9 | 1.000 | 576/576 | 0/0/0 | 0.384574 |
| MARGIN | 0/9 | 0/9 | 2.773/3.686 | 9/9 | 1.000 | 576/576 | 0/0/0 | 0.394977 |
| LOWJ | n/a | n/a | n/a | 9/9 | 1.000 | 576/576 | 0/0/0 | 0.022994 |

Paired CENTER→MARGIN transitions: BOTH_B63 9; CENTER_FAIL→MARGIN_B63 0; CENTER_B63→MARGIN_FAIL 0; BOTH_FAIL 0. All 18 CENTER/MARGIN predictions were in the `rho>1` bin yet all were Q64=1, so neither rho nor raw center error is predictive within this nondiscriminative TEST cohort. The conservative balls are valid inner sets, not complete success basins.

Perturbations that remained in E_bridge were universally successful: CENTER 0.25r 232/232 and 0.50r 168/168; MARGIN 0.25r 152/152 and 0.50r 136/136. MARGIN shows no measurable tolerance gain. Fewer MARGIN perturbations were eligible because its nominal outputs were less often in-domain.

LOWJ is feature-compatible and was evaluated read-only. Among states where all policies were B63, mean statewise J ratios were CENTER/LOWJ 22.23, MARGIN/LOWJ 23.70, and MARGIN/CENTER 0.987.

## Fresh-WIDE t=0 extrapolation (300 matched episodes)

| controller | success | deadlock | timeout | collision | rescue | break | net vs Safety | mean successful J_def |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Safety | 211 | 6 | 83 | 0 | – | – | – | 0 |
| LOWJ | 220 | 46 | 34 | 0 | 74 | 65 | +9 | 0.273 |
| CENTER | 175 | 115 | 10 | 0 | 36 | 72 | -36 | 0.691 |
| MARGIN | 166 | 123 | 11 | 0 | 61 | 106 | -45 | 0.625 |

MARGIN recovered 14 CENTER breaks and created 48 new breaks; it preserved 33 CENTER rescues and added 28 new rescues. Its success-rate difference from CENTER was -3.0 percentage points (95% bootstrap CI -9.33 to +3.33), while its break-rate difference was +11.33 points (95% CI +6.33 to +16.33).

Fresh h0 is far outside labeled support: nearest TRAIN distance mean/median 15.537/15.206; every nearest neighbor was a transferred intermediate state. Failures were not associated with larger distance within this uniformly far-OOD cohort (CENTER failure mean 15.517 vs success 15.551; MARGIN failure 15.431 vs success 15.623). The t0 distribution mismatch remains a major limitation, but does not explain away MARGIN's paired increase in nominal breaks.

## Direct answers

1. Robust-center regression does **not** improve on LOWJ regression: TEST robustness tied, while LOWJ strongly outperformed on fresh-WIDE and had much lower J_def.
2. Basin-margin supervision does **not** improve on center MSE; it materially increases fresh nominal breakage.
3. Normalized error relative to verified radius is **not shown to be more predictive** than point error here: every TEST prediction was outside the balls and still Q64=1.
4. Margin supervision provides **no measurable perturbation-tolerance gain**: both arms retained 100% success for every in-domain perturbation tested.

## Runtime and next step

Six training runs took 25.9s critical-path. Evaluation used 4,576 new continuations and 1,358,643 physical steps; rollout critical path was 7.56 minutes. Maximum allocation was 6 GPU shards, 12 CPU threads, and 48 GiB RAM; GPU peak memory was not captured. No new ball label was generated.

The smallest justified next experiment is **not** another margin loss. First add a small, source-diverse set of verified true-t0 ball labels (with complete source-family separation), then rerun the same CENTER-vs-MARGIN comparison. This directly tests whether the current failure is caused by the documented t0 support mismatch before considering richer losses, Q, J, or selectors.
